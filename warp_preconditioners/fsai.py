"""Row-parallel, diagonally equilibrated adaptive FSAI, entirely in Warp."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator


@lru_cache(None)
def _kernels(dtype, width):
    indices = wp.types.vector(length=width, dtype=wp.int32)
    vector = wp.types.vector(length=width, dtype=dtype)
    matrix = wp.types.matrix(shape=(width, width), dtype=dtype)

    @wp.func
    def entry(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        i: int,
        j: int,
    ):
        lo = offsets[i]
        hi = offsets[i + 1]
        end = hi
        while lo < hi:
            mid = (lo + hi) // 2
            if columns[mid] < j:
                lo = mid + 1
            else:
                hi = mid
        result = dtype(0)
        if lo < end:
            if columns[lo] == j:
                result = values[lo]
        return result

    @wp.kernel(enable_backward=False)
    def diagonal(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        scale: wp.array(dtype=dtype),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        d = entry(offsets, columns, values, i, i)
        if d > dtype(0) and wp.isfinite(d):
            scale[i] = dtype(1) / wp.sqrt(d)
        else:
            scale[i] = dtype(0)
            wp.atomic_add(status, 0, 1)

    @wp.kernel(enable_backward=False)
    def build(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        scale: wp.array(dtype=dtype),
        tolerance: dtype,
        pivot_floor: dtype,
        rows_out: wp.array(dtype=int),
        cols_out: wp.array(dtype=int),
        vals_out: wp.array(dtype=dtype),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        pattern = indices(-1)
        chol = matrix(dtype(0))
        z = vector(dtype(0))
        pattern[0] = i
        chol[0, 0] = dtype(1)
        z[0] = dtype(1)
        size = int(1)
        # Search the frontier of the current support. Duplicates are harmless:
        # candidate scores are evaluated exactly, without a truncated hash table.
        for step in range(wp.static(width - 1)):
            best = int(-1)
            best_score = dtype(0)
            for p in range(size):
                row = pattern[p]
                for e in range(offsets[row], offsets[row + 1]):
                    c = columns[e]
                    selected = bool(c >= i)
                    for q in range(size):
                        if pattern[q] == c:
                            selected = True
                    if not selected:
                        residual = dtype(0)
                        for q in range(size):
                            j = pattern[q]
                            residual += (
                                entry(offsets, columns, values, c, j) * scale[c] * scale[j] * z[q]
                            )
                        score = wp.abs(residual)
                        if score > best_score or (score == best_score and c < best):
                            best = c
                            best_score = score
            if best < 0 or best_score == dtype(0):
                break
            # Border the Cholesky factor of the selected principal submatrix.
            w = vector(dtype(0))
            pivot = dtype(1)
            for p in range(size):
                a = (
                    entry(offsets, columns, values, best, pattern[p])
                    * scale[best]
                    * scale[pattern[p]]
                )
                for q in range(p):
                    a -= chol[p, q] * w[q]
                w[p] = a / chol[p, p]
                pivot -= w[p] * w[p]
            if pivot <= pivot_floor or not wp.isfinite(pivot):
                wp.atomic_add(status, 1, 1)
                break
            pattern[size] = best
            for p in range(size):
                chol[size, p] = w[p]
            chol[size, size] = wp.sqrt(pivot)
            size += 1
            old_z0 = z[0]
            # A[S,S] z = e_0; no global factorization or triangular dependency.
            v = vector(dtype(0))
            for p in range(size):
                a = dtype(0)
                if p == 0:
                    a = dtype(1)
                for q in range(p):
                    a -= chol[p, q] * v[q]
                v[p] = a / chol[p, p]
            for rev in range(size):
                p = size - 1 - rev
                a = v[p]
                for q in range(p + 1, size):
                    a -= chol[q, p] * z[q]
                z[p] = a / chol[p, p]
            # psi=1/z[0]. Relative reduction is 1-old_z0/new_z0.
            if dtype(1) - old_z0 / z[0] <= tolerance:
                break
        norm = wp.sqrt(z[0])
        for p in range(wp.static(width)):
            e = i * wp.static(width) + p
            rows_out[e] = i
            cols_out[e] = -1
            vals_out[e] = dtype(0)
            if p < size:
                cols_out[e] = pattern[p]
                vals_out[e] = z[p] / norm * scale[pattern[p]]

    return diagonal, build


class FSAI(LinearOperator):
    """Approximate inverse ``G.T @ G`` of an SPD BSR matrix.

    ``max_row_size`` includes the diagonal. Each step adds the largest residual
    entry from the lower-triangular graph frontier (one entry per step).
    ``kap_tolerance`` stops rows whose relative energy improvement stagnates.
    Square blocks are scalarized on device. Float32 and float64 are supported.
    Setup synchronizes once to validate diagonals; apply is graph-capturable.
    Matrix symmetry/positive definiteness are caller preconditions.

    Instances own scratch storage; do not apply one instance concurrently on
    independent streams. Rebuild after changing the source matrix.
    """

    def __init__(self, A, max_row_size=8, kap_tolerance=1.0e-3, pivot_floor=None):
        if (
            not isinstance(A, sp.BsrMatrix)
            or A.shape[0] != A.shape[1]
            or A.block_shape[0] != A.block_shape[1]
        ):
            raise ValueError("FSAI requires a square BSR matrix with square blocks")
        if A.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("FSAI supports float32 and float64")
        if not isinstance(max_row_size, int) or not 1 <= max_row_size <= 64:
            raise ValueError("max_row_size must be an integer in [1, 64]")
        if not 0 <= kap_tolerance < 1:
            raise ValueError("kap_tolerance must be in [0, 1)")
        if pivot_floor is None:
            pivot_floor = 1.0e-6 if A.scalar_type == wp.float32 else 1.0e-12
        if not 0 < pivot_floor < 1:
            raise ValueError("pivot_floor must be in (0, 1)")
        self.source = A
        # Also canonicalizes padded storage. No host matrix staging.
        scalar = sp.bsr_copy(A, block_shape=(1, 1))
        n = scalar.nrow
        dtype, device = scalar.scalar_type, scalar.device
        scale = wp.empty(n, dtype=dtype, device=device)
        status = wp.zeros(2, dtype=int, device=device)
        diagonal, build = _kernels(dtype, max_row_size)
        wp.launch(
            diagonal,
            n,
            [scalar.offsets, scalar.columns, scalar.values, scale, status],
            device=device,
        )
        rows = wp.empty(n * max_row_size, dtype=int, device=device)
        cols = wp.empty_like(rows)
        vals = wp.empty(n * max_row_size, dtype=dtype, device=device)
        wp.launch(
            build,
            n,
            [
                scalar.offsets,
                scalar.columns,
                scalar.values,
                scale,
                dtype(kap_tolerance),
                dtype(pivot_floor),
                rows,
                cols,
                vals,
                status,
            ],
            device=device,
        )
        diagnostics = status.numpy()
        if diagnostics[0]:
            raise ValueError(
                f"FSAI requires finite positive diagonals ({diagnostics[0]} invalid rows)"
            )
        self.truncated_rows = int(diagnostics[1])
        self.G = sp.bsr_from_triplets(n, n, rows, cols, vals)
        self.GT = sp.bsr_transposed(self.G)
        self._tmp = wp.empty(n, dtype=dtype, device=device)
        super().__init__(A.shape, A.dtype, device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        x = x.view(self.scalar_type).flatten()
        z = z.view(self.scalar_type).flatten()
        sp.bsr_mv(self.G, x, self._tmp)
        # x is fully consumed before writing z, including when x, y, z alias.
        if beta != 0.0 and z.ptr != y.ptr:
            wp.copy(dest=z, src=y.view(self.scalar_type).flatten())
        sp.bsr_mv(self.GT, self._tmp, z, alpha=alpha, beta=beta)

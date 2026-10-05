"""Row-parallel, diagonally equilibrated adaptive FSAI, entirely in Warp."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator

from .sparse_operator import _matvec


@lru_cache(None)
def _kernels(dtype, width, step_size=1):
    best_indices = wp.types.vector(length=step_size, dtype=wp.int32)
    best_scores = wp.types.vector(length=step_size, dtype=dtype)
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

    @wp.kernel(enable_backward=False, module="unique")
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

    @wp.kernel(enable_backward=False, module="unique")
    def build(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        scale: wp.array(dtype=dtype),
        tolerance: dtype,
        pivot_floor: dtype,
        cols_out: wp.array(dtype=int),
        vals_out: wp.array(dtype=dtype),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        pattern = indices(-1)
        chol = matrix(dtype(0))
        z = vector(dtype(0))
        v = vector(dtype(0))
        pattern[0] = i
        chol[0, 0] = dtype(1)
        z[0] = dtype(1)
        v[0] = dtype(1)
        size = int(1)
        # Search the frontier of the current support. Duplicates are harmless:
        # candidate scores are evaluated exactly, without a truncated hash table.
        while size < wp.static(width):
            candidates = best_indices(-1)
            scores = best_scores(dtype(0))
            for p in range(size):
                row = pattern[p]
                for e in range(offsets[row], offsets[row + 1]):
                    c = columns[e]
                    selected = bool(c >= i)
                    for q in range(size):
                        if pattern[q] == c:
                            selected = True
                    for q in range(wp.static(step_size)):
                        if candidates[q] == c:
                            selected = True
                    if not selected:
                        residual = dtype(0)
                        for q in range(size):
                            j = pattern[q]
                            residual += (
                                entry(offsets, columns, values, c, j) * scale[c] * scale[j] * z[q]
                            )
                        score = wp.abs(residual)
                        for rank in range(wp.static(step_size)):
                            if score > scores[rank] or (
                                score == scores[rank] and c < candidates[rank]
                            ):
                                for rev in range(wp.static(step_size - 1)):
                                    dest = wp.static(step_size - 1) - rev
                                    if dest > rank:
                                        candidates[dest] = candidates[dest - 1]
                                        scores[dest] = scores[dest - 1]
                                candidates[rank] = c
                                scores[rank] = score
                                break
            if candidates[0] < 0:
                break
            old_z0 = z[0]
            stopped = bool(False)
            for rank in range(wp.static(step_size)):
                best = candidates[rank]
                if best < 0 or size == wp.static(width):
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
                    stopped = True
                    break
                pattern[size] = best
                for p in range(size):
                    chol[size, p] = w[p]
                chol[size, size] = wp.sqrt(pivot)
                # Existing forward-solve entries are unchanged by bordering L.
                rhs = dtype(0)
                for p in range(size):
                    rhs -= w[p] * v[p]
                v[size] = rhs / chol[size, size]
                size += 1
            # A[S,S] z = e_0; no global factorization or triangular dependency.
            for rev in range(size):
                p = size - 1 - rev
                a = v[p]
                for q in range(p + 1, size):
                    a -= chol[q, p] * z[q]
                z[p] = a / chol[p, p]
            # psi=1/z[0]. Relative reduction is 1-old_z0/new_z0.
            if stopped or (tolerance > dtype(0) and dtype(1) - old_z0 / z[0] <= tolerance):
                break
        norm = wp.sqrt(z[0])
        valid = bool(True)
        for p in range(wp.static(width)):
            e = i * wp.static(width) + p
            cols_out[e] = -1
            vals_out[e] = dtype(0)
            if p < size:
                cols_out[e] = pattern[p]
                vals_out[e] = z[p] / norm * scale[pattern[p]]
                if not wp.isfinite(vals_out[e]):
                    valid = False
                if p == 0 and vals_out[e] <= dtype(0):
                    valid = False
        if not valid:
            wp.atomic_add(status, 2, 1)

    return diagonal, build


@wp.kernel(enable_backward=False, module="unique")
def _validate_float32_factor(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=wp.float32),
    status: wp.array(dtype=int),
):
    row = wp.tid()
    invalid = bool(False)
    for e in range(offsets[row], offsets[row + 1]):
        if not wp.isfinite(values[e]):
            invalid = True
        if columns[e] == row and values[e] <= wp.float32(0):
            invalid = True
    if invalid:
        wp.atomic_add(status, 0, 1)


class FSAI(LinearOperator):
    """Approximate inverse ``G.T @ G`` of an SPD BSR matrix.

    ``max_row_size`` includes the diagonal. Each search selects up to
    ``max_step_size`` distinct largest-residual entries from the lower-triangular
    graph frontier (default one). Larger batches reduce search work but change
    the selected supports. ``kap_tolerance`` tests relative energy improvement
    after each complete batch; zero disables this test. Retune it when changing batch size.
    Square blocks are scalarized on device. Float32 and float64 are supported.
    ``apply_lanes`` selects cooperative CUDA factor products (1 keeps CSR).
    It changes application arithmetic/order, not the factor construction.
    ``factor_dtype=wp.float32`` optionally compresses the completed factor;
    products still accumulate in the matrix scalar type. The transpose is
    formed from the same rounded factor, retaining the Gram form.
    Setup synchronizes to validate diagonals and optional factor compression; apply is graph-capturable.
    Matrix symmetry/positive definiteness are caller preconditions.

    Instances own scratch storage; do not apply one instance concurrently on
    independent streams. With ``reuse_pattern=True``, ``update(A)`` refits the
    original supports for new values on identical compact BSR topology; otherwise
    rebuild after changing the source. Updates must not overlap application.
    """

    def __init__(
        self,
        A,
        max_row_size=8,
        kap_tolerance=1.0e-3,
        pivot_floor=None,
        apply_lanes=1,
        factor_dtype=None,
        reuse_pattern=False,
        max_step_size=1,
    ):
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
        if not isinstance(max_step_size, int) or not 1 <= max_step_size <= 64:
            raise ValueError("max_step_size must be an integer in [1, 64]")
        if not 0 <= kap_tolerance < 1:
            raise ValueError("kap_tolerance must be in [0, 1)")
        if pivot_floor is None:
            pivot_floor = 1.0e-6 if A.scalar_type == wp.float32 else 1.0e-12
        if not 0 < pivot_floor < 1:
            raise ValueError("pivot_floor must be in (0, 1)")
        if not isinstance(apply_lanes, int) or apply_lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError("apply_lanes must be one of 1, 2, 4, 8, 16, 32")
        if factor_dtype is None:
            factor_dtype = A.scalar_type
        if factor_dtype not in (A.scalar_type, wp.float32):
            raise ValueError("factor_dtype must be the matrix scalar type or wp.float32")
        if reuse_pattern and A.row_counts is not None:
            raise ValueError(
                "reuse_pattern requires compact BSR storage; canonicalize with bsr_copy"
            )
        self.max_row_size = max_row_size
        self.max_step_size = max_step_size
        self.factor_dtype = factor_dtype
        self.apply_lanes = apply_lanes
        self.source = A
        # Settle the source count before scalarization: nnz can otherwise be a
        # very loose assembly upper bound, causing oversized temporary buffers.
        A.nnz_sync()
        # Also canonicalizes padded storage. No host matrix staging.
        from .fsai_pack import scalarize_compact

        if A.row_counts is not None:
            scalar = sp.bsr_copy(A, block_shape=(1, 1))
        elif A.dtype == A.scalar_type and not reuse_pattern:
            scalar = A
        else:
            scalar = scalarize_compact(A)
        n = scalar.nrow
        dtype, device = scalar.scalar_type, scalar.device
        scale = wp.empty(n, dtype=dtype, device=device)
        status = wp.zeros(3, dtype=int, device=device)
        diagonal, build = _kernels(dtype, max_row_size, max_step_size)
        wp.launch(
            diagonal,
            n,
            [scalar.offsets, scalar.columns, scalar.values, scale, status],
            device=device,
        )
        cols = wp.empty(n * max_row_size, dtype=int, device=device)
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
        if diagnostics[2]:
            raise ValueError("FSAI factor lost finite entries or a positive diagonal")
        from .fsai_pack import pack_factor

        self.G = pack_factor(n, max_row_size, cols, vals)
        if factor_dtype != dtype:
            self.G = sp.bsr_copy(self.G, scalar_type=factor_dtype)
            status.zero_()
            wp.launch(
                _validate_float32_factor,
                n,
                [self.G.offsets, self.G.columns, self.G.values, status],
                device=device,
            )
            if status.numpy()[0]:
                raise ValueError(
                    "factor_dtype conversion lost finite entries or a positive diagonal"
                )
        self.GT = sp.bsr_transposed(self.G)
        self._tmp = wp.empty(n, dtype=dtype, device=device)
        super().__init__(A.shape, A.dtype, device, self._apply)
        self._refit = None
        if reuse_pattern:
            from .fsai_reuse import RefitPlan

            self._refit = RefitPlan(self, A, scalar, max_row_size, pivot_floor)

    def update(self, A=None):
        """Refit on the original factor supports, preserving captured apply buffers.

        Requires ``reuse_pattern=True`` and identical compact BSR topology/storage.
        This is not a new adaptive pattern search. Rebuild when convergence worsens.
        Failed validation leaves the previous factors intact. Updates synchronize.
        """
        if self._refit is None:
            raise ValueError("Construct FSAI with reuse_pattern=True before updating")
        A = self.source if A is None else A
        if not isinstance(A, sp.BsrMatrix):
            raise ValueError("update requires a Warp BSR matrix")
        self._refit.update(
            self, A, _kernels(self.scalar_type, self.max_row_size, self.max_step_size)[0]
        )
        self.source = A
        return self

    def quality(self, A=None, probes=4, seed=17):
        """Estimate normalized Frobenius defect of G A G.T using fixed random probes.

        A diagnostic heuristic, not a bound on conditioning or solve iterations.
        Includes GPU products, allocations and host synchronization; not capturable.
        """
        from .fsai_reuse import quality

        return quality(self, self.source if A is None else A, probes, seed)

    def _apply(self, x, y, z, alpha, beta):
        x = x.view(self.scalar_type).flatten()
        z = z.view(self.scalar_type).flatten()
        y = y.view(self.scalar_type).flatten()
        if alpha != 0.0:
            _matvec(self.G, x, self._tmp, self._tmp, 1.0, 0.0, self.apply_lanes)
        # x is fully consumed before writing z, including when x, y, z alias.
        _matvec(self.GT, self._tmp, y, z, alpha, beta, self.apply_lanes)

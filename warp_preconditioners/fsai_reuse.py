"""Cached topology and fixed-support FSAI numerical refits, entirely in Warp."""

from functools import lru_cache

import warp as wp

from .sparse_operator import _matvec


@wp.kernel(enable_backward=False, module="unique")
def compare(a: wp.array(dtype=int), b: wp.array(dtype=int), bad: wp.array(dtype=int)):
    i = wp.tid()
    if a[i] != b[i]:
        wp.atomic_add(bad, 0, 1)


@wp.func
def find(offsets: wp.array(dtype=int), columns: wp.array(dtype=int), i: int, j: int):
    lo = offsets[i]
    hi = offsets[i + 1]
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if columns[mid] < j:
            lo = mid + 1
        else:
            hi = mid
    result = int(-1)
    if lo < end:
        if columns[lo] == j:
            result = lo
    return result


@wp.kernel(enable_backward=False, module="unique")
def source_map(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    source_offsets: wp.array(dtype=int),
    source_columns: wp.array(dtype=int),
    block_size: int,
    mapping: wp.array(dtype=int),
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        slot = find(source_offsets, source_columns, i // block_size, j // block_size)
        mapping[e] = slot * block_size * block_size + (i % block_size) * block_size + j % block_size


@wp.kernel(enable_backward=False, module="unique")
def transpose_map(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    toff: wp.array(dtype=int),
    tcol: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        mapping[e] = find(toff, tcol, columns[e], i)


@lru_cache(None)
def kernels(dtype, storage, width):
    vec = wp.types.vector(length=width, dtype=dtype)
    mat = wp.types.matrix(shape=(width, width), dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def gather(
        source: wp.array(dtype=dtype), mapping: wp.array(dtype=int), out: wp.array(dtype=dtype)
    ):
        e = wp.tid()
        out[e] = source[mapping[e]]

    @wp.kernel(enable_backward=False, module="unique")
    def refit(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        goff: wp.array(dtype=int),
        gcol: wp.array(dtype=int),
        scale: wp.array(dtype=dtype),
        floor: dtype,
        out: wp.array(dtype=storage),
        bad: wp.array(dtype=int),
    ):
        i = wp.tid()
        start = goff[i]
        size = goff[i + 1] - start
        chol = mat(dtype(0))
        valid = bool(True)
        for p in range(size):
            r = gcol[start + p]
            for q in range(p + 1):
                c = gcol[start + q]
                slot = find(offsets, columns, r, c)
                a = dtype(0)
                if slot >= 0:
                    a = values[slot] * scale[r] * scale[c]
                for k in range(q):
                    a -= chol[p, k] * chol[q, k]
                if p == q:
                    if a <= floor or not wp.isfinite(a):
                        valid = False
                        a = dtype(1)
                    chol[p, q] = wp.sqrt(a)
                else:
                    chol[p, q] = a / chol[q, q]
        z = vec(dtype(0))
        for p in range(size):
            a = dtype(0)
            if gcol[start + p] == i:
                a = dtype(1)
            for q in range(p):
                a -= chol[p, q] * z[q]
            z[p] = a / chol[p, p]
        for rev in range(size):
            p = size - 1 - rev
            a = z[p]
            for q in range(p + 1, size):
                a -= chol[q, p] * z[q]
            z[p] = a / chol[p, p]
        # Canonical lower-triangular support: the diagonal is last.
        norm = wp.sqrt(z[size - 1])
        for p in range(size):
            value = storage(z[p] / norm * scale[gcol[start + p]])
            out[start + p] = value
            if not wp.isfinite(value):
                valid = False
            if p == size - 1 and value <= storage(0):
                valid = False
        if not valid:
            wp.atomic_add(bad, 0, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def transpose(
        values: wp.array(dtype=storage), mapping: wp.array(dtype=int), out: wp.array(dtype=storage)
    ):
        e = wp.tid()
        out[mapping[e]] = values[e]

    @wp.kernel(enable_backward=False, module="unique")
    def probe(seed: int, out: wp.array(dtype=dtype)):
        i = wp.tid()
        state = wp.rand_init(seed, i)
        out[i] = dtype(2 * wp.randi(state, 0, 2) - 1)

    @wp.kernel(enable_backward=False, module="unique")
    def subtract(a: wp.array(dtype=dtype), b: wp.array(dtype=dtype)):
        i = wp.tid()
        a[i] -= b[i]

    return gather, refit, transpose, probe, subtract


class RefitPlan:
    def __init__(self, owner, A, scalar, width, floor):
        self.signature = (A.shape, A.dtype, A.device, A.row_counts is not None)
        self.topology = [wp.clone(a) for a in self.arrays(A)]
        self.scalar = scalar
        self.floor = floor
        self.kernels = kernels(A.scalar_type, owner.factor_dtype, width)
        self.mapping = wp.empty(scalar.nnz_sync(), dtype=int, device=A.device)
        # Padded source rows need compact search endpoints, not unused slots.
        # source_map's search is safe for compact input; padded input is rejected
        # explicitly for opt-in reuse, while ordinary construction supports it.
        wp.launch(
            source_map,
            scalar.nrow,
            [scalar.offsets, scalar.columns, A.offsets, A.columns, A.block_shape[0], self.mapping],
            device=A.device,
        )
        self.transpose = wp.empty(owner.G.nnz_sync(), dtype=int, device=A.device)
        wp.launch(
            transpose_map,
            owner.G.nrow,
            [owner.G.offsets, owner.G.columns, owner.GT.offsets, owner.GT.columns, self.transpose],
            device=A.device,
        )
        self.values = wp.empty_like(scalar.values)
        self.candidate = wp.empty_like(owner.G.values)
        self.scale = wp.empty(scalar.nrow, dtype=A.scalar_type, device=A.device)
        self.bad = wp.zeros(2, dtype=int, device=A.device)

    @staticmethod
    def arrays(A):
        return [A.offsets[: A.nrow + 1], A.columns[: A.nnz_sync()]] + (
            [A.row_counts] if A.row_counts is not None else []
        )

    def update(self, owner, A, diagonal):
        if (A.shape, A.dtype, A.device, A.row_counts is not None) != self.signature:
            raise ValueError("FSAI update requires the original shape, dtype, device and topology")
        incoming = self.arrays(A)
        if any(a.shape != b.shape for a, b in zip(incoming, self.topology)):
            raise ValueError("FSAI update requires identical topology storage")
        self.bad.zero_()
        for a, b in zip(incoming, self.topology):
            wp.launch(compare, a.size, [a, b, self.bad], device=A.device)
        if self.bad.numpy()[0]:
            raise ValueError("FSAI update requires identical sparsity; rebuild to change it")
        if A.values.size < self.topology[1].size:
            raise ValueError("Insufficient source values")
        wp.launch(
            self.kernels[0],
            self.mapping.size,
            [A.values.view(A.scalar_type).flatten(), self.mapping, self.values],
            device=A.device,
        )
        s = self.scalar
        wp.launch(
            diagonal,
            s.nrow,
            [s.offsets, s.columns, self.values, self.scale, self.bad],
            device=A.device,
        )
        wp.launch(
            self.kernels[1],
            s.nrow,
            [
                s.offsets,
                s.columns,
                self.values,
                owner.G.offsets,
                owner.G.columns,
                self.scale,
                A.scalar_type(self.floor),
                self.candidate,
                self.bad,
            ],
            device=A.device,
        )
        if self.bad.numpy()[0]:
            raise ValueError("FSAI refit requires finite positive diagonals and SPD local supports")
        wp.copy(owner.G.values, self.candidate)
        wp.launch(
            self.kernels[2],
            self.transpose.size,
            [owner.G.values, self.transpose, owner.GT.values],
            device=A.device,
        )


def quality(owner, A, probes=4, seed=17):
    """Rademacher estimate of ||G A G.T - I||_F / sqrt(n), not a condition number."""
    if not isinstance(probes, int) or probes < 1:
        raise ValueError("probes must be a positive integer")
    if (A.shape, A.dtype, A.device) != (owner.shape, owner.dtype, owner.device):
        raise ValueError("quality requires matching shape, dtype and device")
    from warp.optim.linear import aslinearoperator

    op = aslinearoperator(A)
    vector_type = (
        wp.types.vector(length=A.block_shape[0], dtype=A.scalar_type)
        if A.block_shape[0] > 1
        else A.scalar_type
    )
    k = kernels(A.scalar_type, owner.factor_dtype, owner.max_row_size)
    n = A.shape[0]
    x, y, z, w = [wp.empty(n, dtype=A.scalar_type, device=A.device) for _ in range(4)]
    total = 0.0
    for p in range(probes):
        wp.launch(k[3], n, [seed + p, x], device=A.device)
        _matvec(owner.GT, x, y, y, 1.0, 0.0, owner.apply_lanes)
        op.matvec(
            y.reshape((A.nrow, A.block_shape[0])).view(vector_type).flatten(),
            z.reshape((A.nrow, A.block_shape[0])).view(vector_type).flatten(),
            z.reshape((A.nrow, A.block_shape[0])).view(vector_type).flatten(),
            1.0,
            0.0,
        )
        _matvec(owner.G, z, w, w, 1.0, 0.0, owner.apply_lanes)
        wp.launch(k[4], n, [w, x], device=A.device)
        total += float(wp.utils.array_inner(w, w))
    return (total / (probes * max(n, 1))) ** 0.5

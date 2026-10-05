"""Experimental float64 3x3 SPD block Jacobi using local Cholesky solves.

The public BlockJacobi also handles indefinite blocks. This experiment uses the
stronger SPD assumption to avoid determinant thresholds and cofactor cancellation.
"""

import warp as wp
from warp.optim.linear import LinearOperator


@wp.kernel(enable_backward=False)
def invert(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    padded: bool,
    values: wp.array(dtype=wp.mat33d),
    out: wp.array(dtype=wp.mat33d),
    bad: wp.array(dtype=int),
):
    i = wp.tid()
    lo = offsets[i]
    hi = offsets[i + 1]
    if padded:
        hi = lo + counts[i]
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if columns[mid] < i:
            lo = mid + 1
        else:
            hi = mid
    if lo >= end:
        wp.atomic_add(bad, 0, 1)
        return
    if columns[lo] != i:
        wp.atomic_add(bad, 0, 1)
        return
    a = values[lo]
    chol = wp.mat33d(wp.float64(0))
    for r in range(3):
        for c in range(r + 1):
            v = a[r, c]
            for k in range(c):
                v -= chol[r, k] * chol[c, k]
            if r == c:
                if v <= wp.float64(0) or not wp.isfinite(v):
                    wp.atomic_add(bad, 0, 1)
                    return
                chol[r, c] = wp.sqrt(v)
            else:
                chol[r, c] = v / chol[c, c]
    inv = wp.mat33d(wp.float64(0))
    for c in range(3):
        z = wp.vec3d(wp.float64(0))
        for r in range(3):
            v = wp.float64(0)
            if r == c:
                v = wp.float64(1)
            for k in range(r):
                v -= chol[r, k] * z[k]
            z[r] = v / chol[r, r]
        for rev in range(3):
            r = 2 - rev
            v = z[r]
            for k in range(r + 1, 3):
                v -= chol[k, r] * z[k]
            z[r] = v / chol[r, r]
            inv[r, c] = z[r]
    out[i] = (inv + wp.transpose(inv)) * wp.float64(0.5)


@wp.kernel(enable_backward=False)
def apply(
    inv: wp.array(dtype=wp.mat33d),
    x: wp.array(dtype=wp.vec3d),
    y: wp.array(dtype=wp.vec3d),
    z: wp.array(dtype=wp.vec3d),
    alpha: wp.float64,
    beta: wp.float64,
):
    i = wp.tid()
    value = alpha * (inv[i] * x[i])
    if beta != wp.float64(0):
        value += beta * y[i]
    z[i] = value


class SpdBlockJacobi(LinearOperator):
    def __init__(self, a):
        if a.block_shape != (3, 3) or a.scalar_type != wp.float64:
            raise ValueError("Experimental SPD block Jacobi requires float64 3x3 blocks")
        self.inverse_diagonal = wp.empty(a.nrow, dtype=wp.mat33d, device=a.device)
        self._candidate = wp.empty_like(self.inverse_diagonal)
        self._bad = wp.zeros(1, dtype=int, device=a.device)
        super().__init__(a.shape, a.dtype, a.device, self._apply)
        self.update(a)

    def update(self, a):
        if (a.shape, a.dtype, a.device) != (self.shape, self.dtype, self.device):
            raise ValueError("update requires identical shape, dtype and device")
        self._bad.zero_()
        wp.launch(
            invert,
            a.nrow,
            [
                a.offsets,
                a.columns,
                a.row_counts if a.row_counts is not None else a.offsets,
                a.row_counts is not None,
                a.values,
                self._candidate,
                self._bad,
            ],
            device=a.device,
        )
        if self._bad.numpy()[0]:
            raise ValueError("Missing or nonpositive SPD block Jacobi pivot")
        wp.copy(self.inverse_diagonal, self._candidate)
        return self

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(wp.vec3d).flatten() for v in (x, y, z))
        wp.launch(apply, len(x), [self.inverse_diagonal, x, y, z, alpha, beta], device=self.device)

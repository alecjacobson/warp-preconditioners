"""Cooperative CSR products for short rows, implemented with Warp tiles."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator


@lru_cache(None)
def _grouped_kernel(dtype, lanes, storage_type):
    rows = 128 // lanes

    @wp.kernel(enable_backward=False, module="unique")
    def mv(
        n: int,
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=storage_type),
        x: wp.array(dtype=dtype),
        y: wp.array(dtype=dtype),
        z: wp.array(dtype=dtype),
        alpha: dtype,
        beta: dtype,
    ):
        row, lane = wp.tid()
        result = dtype(0)
        if row < n and alpha != dtype(0):
            for e in range(offsets[row] + lane, offsets[row + 1], wp.static(lanes)):
                result += dtype(values[e]) * x[columns[e]]
        t = wp.tile_reshape(wp.tile(result), shape=(wp.static(rows), wp.static(lanes)))
        sums = wp.tile_sum(t, axis=1)
        result = wp.tile_extract(sums, row % wp.static(rows))
        if lane == 0 and row < n:
            result *= alpha
            if beta != dtype(0):
                result += beta * y[row]
            z[row] = result

    return mv


def _grouped_mv(a, x, y, z, alpha, beta, lanes):
    rows = 128 // lanes
    wp.launch(
        _grouped_kernel(x.dtype, lanes, a.scalar_type),
        dim=(((a.nrow + rows - 1) // rows) * rows, lanes),
        inputs=[
            a.nrow,
            a.offsets,
            a.columns,
            a.values,
            x,
            y,
            z,
            x.dtype(alpha),
            x.dtype(beta),
        ],
        device=a.device,
        block_dim=128,
    )


@lru_cache(None)
def _mixed_kernel(storage_type, dtype):
    @wp.kernel(enable_backward=False, module="unique")
    def mv(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=storage_type),
        x: wp.array(dtype=dtype),
        y: wp.array(dtype=dtype),
        z: wp.array(dtype=dtype),
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        result = dtype(0)
        if alpha != dtype(0):
            for e in range(offsets[row], offsets[row + 1]):
                result += dtype(values[e]) * x[columns[e]]
            result *= alpha
        if beta != dtype(0):
            result += beta * y[row]
        z[row] = result

    return mv


def _matvec(a, x, y, z, alpha, beta, lanes):
    """Canonical scalar CSR; x and z must not alias for the tiled path."""
    if a.device.is_cuda and lanes > 1:
        _grouped_mv(a, x, y, z, alpha, beta, lanes)
    elif a.scalar_type != x.dtype:
        wp.launch(
            _mixed_kernel(a.scalar_type, x.dtype),
            a.nrow,
            [a.offsets, a.columns, a.values, x, y, z, x.dtype(alpha), x.dtype(beta)],
            device=a.device,
        )
    else:
        if beta != 0.0 and z.ptr != y.ptr:
            wp.copy(z, y)
        sp.bsr_mv(a, x, z, alpha=alpha, beta=beta)


class SparseOperator(LinearOperator):
    """BSR LinearOperator with cooperative short-row products on CUDA.

    ``row_lanes`` threads cooperate on each scalar row; 1 selects Warp's
    ordinary sparse product. 4 or 8 are useful for short mesh stencils.
    CPU uses Warp's ordinary CSR implementation. Construction scalarizes
    and canonicalizes on device. No host matrix staging is used.

    Owns scratch for x/z aliasing; do not apply concurrently on independent
    streams. Rebuild when the input matrix changes.
    """

    def __init__(self, A, row_lanes=4):
        if not isinstance(A, sp.BsrMatrix):
            raise TypeError("SparseOperator requires a Warp BSR matrix")
        if A.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("SparseOperator supports float32 and float64")
        if not isinstance(row_lanes, int) or row_lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError("row_lanes must be one of 1, 2, 4, 8, 16, 32")
        self.source = A
        self.scalar = sp.bsr_copy(A, block_shape=(1, 1))
        self.row_lanes = row_lanes
        self._alias = wp.empty(self.scalar.ncol, dtype=A.scalar_type, device=A.device)
        super().__init__(A.shape, A.dtype, A.device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        x = x.view(self.scalar_type).flatten()
        y = y.view(self.scalar_type).flatten()
        z = z.view(self.scalar_type).flatten()
        if x.ptr == z.ptr:
            wp.copy(self._alias, x)
            x = self._alias
        _matvec(self.scalar, x, y, z, alpha, beta, self.row_lanes)

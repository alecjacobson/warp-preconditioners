"""Factored application of a squared Laplacian with exact variable elimination."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator


@wp.kernel(enable_backward=False, module="unique")
def _index_free(free: wp.array(dtype=int), index: wp.array(dtype=int), status: wp.array(dtype=int)):
    i = wp.tid()
    j = free[i]
    if j < 0 or j >= index.shape[0]:
        wp.atomic_add(status, 0, 1)
    else:
        old = wp.atomic_cas(index, j, -1, i)
        if old != -1:
            wp.atomic_add(status, 1, 1)


@lru_cache(None)
def _kernels(dtype, lanes):
    rows = 128 // lanes

    @wp.kernel(enable_backward=False, module="unique")
    def mass_inverse(
        mass: wp.array(dtype=dtype), inverse: wp.array(dtype=dtype), status: wp.array(dtype=int)
    ):
        i = wp.tid()
        if mass[i] > dtype(0) and wp.isfinite(mass[i]):
            inverse[i] = dtype(1) / mass[i]
        else:
            wp.atomic_add(status, 2, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        offsets: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        vals: wp.array(dtype=dtype),
        inverse: wp.array(dtype=dtype),
        index: wp.array(dtype=int),
        x: wp.array(dtype=dtype),
        tmp: wp.array(dtype=dtype),
        boundary: bool,
    ):
        row, lane = wp.tid()
        acc = dtype(0)
        if row < tmp.shape[0]:
            for e in range(offsets[row] + lane, offsets[row + 1], wp.static(lanes)):
                col = cols[e]
                c = index[col]
                if boundary:
                    if c < 0:
                        acc += vals[e] * x[col]
                else:
                    if c >= 0:
                        acc += vals[e] * x[c]
        if wp.static(lanes > 1):
            t = wp.tile_reshape(wp.tile(acc), shape=(wp.static(rows), wp.static(lanes)))
            sums = wp.tile_sum(t, axis=1)
            acc = wp.tile_extract(sums, row % wp.static(rows))
        if lane == 0 and row < tmp.shape[0]:
            tmp[row] = inverse[row] * acc

    @wp.kernel(enable_backward=False, module="unique")
    def backward(
        offsets: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        vals: wp.array(dtype=dtype),
        free: wp.array(dtype=int),
        tmp: wp.array(dtype=dtype),
        y: wp.array(dtype=dtype),
        z: wp.array(dtype=dtype),
        alpha: dtype,
        beta: dtype,
    ):
        i, lane = wp.tid()
        acc = dtype(0)
        if i < free.shape[0] and alpha != dtype(0):
            row = free[i]
            for e in range(offsets[row] + lane, offsets[row + 1], wp.static(lanes)):
                acc += vals[e] * tmp[cols[e]]
        if wp.static(lanes > 1):
            t = wp.tile_reshape(wp.tile(acc), shape=(wp.static(rows), wp.static(lanes)))
            sums = wp.tile_sum(t, axis=1)
            acc = wp.tile_extract(sums, i % wp.static(rows))
        if lane == 0 and i < free.shape[0]:
            acc *= alpha
            if beta != dtype(0):
                acc += beta * y[i]
            z[i] = acc

    return mass_inverse, forward, backward


class SquaredLaplacianOperator(LinearOperator):
    """Apply ``(L diag(1/mass) L)[free,free]`` without multiplying sparse matrices.

    ``L`` must be a symmetric scalar BSR matrix. ``mass`` contains positive
    lumped masses. ``free_indices`` contains distinct free scalar indices;
    any omitted indices are constrained. All arrays must be on L.device.
    The energy includes all rows of L, including constrained vertices.
    Positive definiteness depends on the constraints removing L's nullspace.

    ``rhs(boundary_values)`` constructs the eliminated RHS with the same
    factored arithmetic. Values at free indices in that full vector are ignored.
    Setup and products run in Warp; setup reads only validation counters.
    Instances own scratch and must not be used on concurrent streams.
    """

    def __init__(self, L, mass, free_indices, row_lanes=1):
        if not isinstance(L, sp.BsrMatrix) or L.nrow != L.ncol or L.block_shape != (1, 1):
            raise ValueError("L must be a square scalar BSR matrix")
        if L.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("L must use float32 or float64")
        if mass.device != L.device or free_indices.device != L.device:
            raise ValueError("L, mass, and free_indices must be on the same device")
        if mass.dtype != L.scalar_type or mass.ndim != 1 or mass.size != L.nrow:
            raise ValueError("mass must match the scalar type and size of L")
        if free_indices.dtype != wp.int32 or free_indices.ndim != 1:
            raise ValueError("free_indices must be a one-dimensional int32 array")
        if not isinstance(row_lanes, int) or row_lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError("row_lanes must be one of 1, 2, 4, 8, 16, 32")
        self.L = sp.bsr_copy(L)
        self.mass = mass
        self.free = wp.clone(free_indices)
        self.index = wp.full(L.nrow, -1, dtype=int, device=L.device)
        self.inverse_mass = wp.empty_like(mass)
        self.row_lanes = row_lanes if L.device.is_cuda else 1
        validate_mass, self._forward, self._backward = _kernels(L.scalar_type, self.row_lanes)
        status = wp.zeros(3, dtype=int, device=L.device)
        wp.launch(_index_free, self.free.size, [self.free, self.index, status], device=L.device)
        wp.launch(validate_mass, L.nrow, [mass, self.inverse_mass, status], device=L.device)
        errors = status.numpy()
        if errors[0] or errors[1]:
            raise ValueError("free_indices must be distinct and in bounds")
        if errors[2]:
            raise ValueError("mass entries must be finite and positive")
        self._tmp = wp.empty_like(mass)
        super().__init__((self.free.size, self.free.size), L.dtype, L.device, self._apply)

    def _dim(self, n):
        rows = 128 // self.row_lanes
        return (((n + rows - 1) // rows) * rows, self.row_lanes)

    def _product(self, x, y, z, alpha, beta, boundary):
        L = self.L
        wp.launch(
            self._forward,
            dim=self._dim(L.nrow),
            inputs=[
                L.offsets,
                L.columns,
                L.values,
                self.inverse_mass,
                self.index,
                x,
                self._tmp,
                boundary,
            ],
            device=self.device,
            block_dim=128,
        )
        wp.launch(
            self._backward,
            dim=self._dim(self.free.size),
            inputs=[
                L.offsets,
                L.columns,
                L.values,
                self.free,
                self._tmp,
                y,
                z,
                self.scalar_type(alpha),
                self.scalar_type(beta),
            ],
            device=self.device,
            block_dim=128,
        )

    def _apply(self, x, y, z, alpha, beta):
        self._product(x, y, z, alpha, beta, False)

    def rhs(self, boundary_values, out=None):
        if (
            boundary_values.dtype != self.scalar_type
            or boundary_values.device != self.device
            or boundary_values.ndim != 1
            or boundary_values.size != self.L.nrow
        ):
            raise ValueError("boundary_values must match the scalar type, device, and size of L")
        if out is None:
            out = wp.empty(self.free.size, dtype=self.scalar_type, device=self.device)
        elif (
            out.dtype != self.scalar_type
            or out.device != self.device
            or out.ndim != 1
            or out.size != self.free.size
        ):
            raise ValueError("out must match the reduced operator")
        self._product(boundary_values, out, out, -1.0, 0.0, True)
        return out

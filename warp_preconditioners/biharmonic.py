"""Equivalent first-order system for M + tau K M^{-1} K (K positive semidefinite)."""

import math
from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator

from .fsai import FSAI


@lru_cache(None)
def _kernels(dtype):
    mat2 = wp.types.matrix(shape=(2, 2), dtype=dtype)
    vec2 = wp.types.vector(length=2, dtype=dtype)

    @wp.kernel
    def lift(
        n: int,
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        mass: wp.array(dtype=dtype),
        root_tau: dtype,
        rows: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        blocks: wp.array(dtype=mat2),
        hvals: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        for e in range(offsets[i], offsets[i + 1]):
            j = columns[e]
            k = root_tau * values[e]
            rows[n + e] = i
            cols[n + e] = j
            blocks[n + e] = mat2(dtype(0), -k, k, dtype(0))
            hvals[n + e] = k
        rows[i] = i
        cols[i] = i
        blocks[i] = mat2(mass[i], dtype(0), dtype(0), mass[i])
        hvals[i] = mass[i]

    @wp.kernel
    def duplicate(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        rows: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        blocks: wp.array(dtype=mat2),
    ):
        i = wp.tid()
        for e in range(offsets[i], offsets[i + 1]):
            rows[e] = i
            cols[e] = columns[e]
            g = values[e]
            blocks[e] = mat2(g, dtype(0), dtype(0), g)

    @wp.kernel
    def embed(b: wp.array(dtype=dtype), out: wp.array(dtype=vec2)):
        i = wp.tid()
        out[i] = vec2(b[i], dtype(0))

    @wp.kernel
    def extract(x: wp.array(dtype=vec2), out: wp.array(dtype=dtype)):
        i = wp.tid()
        out[i] = x[i][0]

    @wp.kernel
    def validate(mass: wp.array(dtype=dtype), bad: wp.array(dtype=int)):
        i = wp.tid()
        if mass[i] <= dtype(0) or not wp.isfinite(mass[i]):
            wp.atomic_add(bad, 0, 1)

    return lift, duplicate, embed, extract, validate


class RepeatedFSAI(LinearOperator):
    """Apply the same scalar FSAI independently to both components of vec2 arrays."""

    def __init__(self, factor):
        self.factor = factor
        g = factor.G
        g.nnz_sync()
        self.vector_type = wp.types.vector(length=2, dtype=g.scalar_type)
        mat2 = wp.types.matrix(shape=(2, 2), dtype=g.scalar_type)
        rows = wp.empty(g.nnz, dtype=int, device=g.device)
        cols = wp.empty_like(rows)
        vals = wp.empty(g.nnz, dtype=mat2, device=g.device)
        wp.launch(
            _kernels(g.scalar_type)[1],
            g.nrow,
            [g.offsets, g.columns, g.values, rows, cols, vals],
            device=g.device,
        )
        self.G = sp.bsr_from_triplets(g.nrow, g.ncol, rows, cols, vals)
        self.GT = sp.bsr_transposed(self.G)
        self._tmp = wp.empty(g.nrow, dtype=self.vector_type, device=g.device)
        super().__init__((2 * g.nrow, 2 * g.ncol), mat2, g.device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        sp.bsr_mv(self.G, x, self._tmp)
        if beta != 0.0 and z.ptr != y.ptr:
            wp.copy(dest=z, src=y)
        sp.bsr_mv(self.GT, self._tmp, z, alpha=alpha, beta=beta)


class BiharmonicSystem:
    """Lift a biharmonic solve to a real 2x2-block, first-order BSR system.

    Input: symmetric PSD stiffness ``K`` (positive Laplacian), positive diagonal
    ``mass`` (a scalar Warp array), and positive ``tau``. Eliminating the second
    component of ``matrix @ (u,v) = (b,0)`` gives exactly
    ``(M + tau*K*M^-1*K) u = b``. No boundary conditions are changed.

    ``matrix = [[M, -sqrt(tau)*K], [sqrt(tau)*K, M]]`` in interleaved ordering.
    It is nonsymmetric: use BiCGSTAB or GMRES, never CG/CR. The SPD matrix
    ``H = M + sqrt(tau)*K`` provides the scalar FSAI target.

    Always check the original biharmonic residual after extracting u: the
    lifted system has different residual scaling. Float64 is recommended.
    """

    def __init__(self, K, mass, tau=1.0):
        if not isinstance(K, sp.BsrMatrix) or K.shape[0] != K.shape[1] or K.block_shape != (1, 1):
            raise ValueError("K must be a square scalar BSR matrix")
        if K.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("K must use float32 or float64")
        if mass.dtype != K.scalar_type or mass.device != K.device or mass.shape != (K.nrow,):
            raise ValueError("mass must match K's scalar dtype, device, and row count")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        # Compact topology avoids holes in COO output for padded inputs.
        K = sp.bsr_copy(K)
        K.nnz_sync()
        self.K, self.mass, self.tau = K, wp.clone(mass), tau
        self.vector_type = wp.types.vector(length=2, dtype=K.scalar_type)
        kernels = _kernels(K.scalar_type)
        bad = wp.zeros(1, dtype=int, device=K.device)
        wp.launch(kernels[4], K.nrow, [mass, bad], device=K.device)
        if bad.numpy()[0]:
            raise ValueError("mass must contain finite positive entries")
        rows = wp.empty(K.nrow + K.nnz, dtype=int, device=K.device)
        cols = wp.empty_like(rows)
        blocks = wp.empty(
            rows.size,
            dtype=wp.types.matrix(shape=(2, 2), dtype=K.scalar_type),
            device=K.device,
        )
        hvals = wp.empty(rows.size, dtype=K.scalar_type, device=K.device)
        wp.launch(
            kernels[0],
            K.nrow,
            [
                K.nrow,
                K.offsets,
                K.columns,
                K.values,
                mass,
                K.scalar_type(math.sqrt(tau)),
                rows,
                cols,
                blocks,
                hvals,
            ],
            device=K.device,
        )
        self.matrix = sp.bsr_from_triplets(K.nrow, K.ncol, rows, cols, blocks)
        self.H = sp.bsr_from_triplets(K.nrow, K.ncol, rows, cols, hvals)

    def preconditioner(self, **fsai_options):
        return RepeatedFSAI(FSAI(self.H, **fsai_options))

    def rhs(self, b):
        """Embed a scalar RHS as interleaved (b, 0)."""
        if b.shape != self.mass.shape or b.dtype != self.mass.dtype or b.device != self.mass.device:
            raise ValueError("b must match mass's shape, dtype and device")
        out = wp.empty(b.size, dtype=self.vector_type, device=b.device)
        wp.launch(_kernels(b.dtype)[2], b.size, [b, out], device=b.device)
        return out

    def solution(self, x, out=None):
        """Extract u from a lifted solution; optionally reuse a scalar output."""
        if (
            x.dtype != self.vector_type
            or x.shape != self.mass.shape
            or x.device != self.mass.device
        ):
            raise ValueError("x must be a vec2 solution on the system device")
        if out is None:
            out = wp.empty_like(self.mass)
        if (
            out.dtype != self.mass.dtype
            or out.shape != self.mass.shape
            or out.device != self.mass.device
        ):
            raise ValueError("out must match mass's shape, dtype and device")
        wp.launch(_kernels(self.mass.dtype)[3], x.size, [x, out], device=x.device)
        return out

"""Mixed harmonic benchmark systems and a matching Schur preconditioner."""

import math
from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator

from .fsai import FSAI
from .sparse_operator import SparseOperator


@lru_cache(None)
def _kernels(dtype, order):
    mat = wp.types.matrix(shape=(order, order), dtype=dtype)
    vec = wp.types.vector(length=order, dtype=dtype)

    @wp.kernel(enable_backward=False)
    def build(
        n: int,
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        mass: wp.array(dtype=dtype),
        weight: dtype,
        shift: dtype,
        rows: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        blocks: wp.array(dtype=mat),
        hvalues: wp.array(dtype=dtype),
        bad: wp.array(dtype=int),
    ):
        i = wp.tid()
        if mass[i] <= dtype(0) or not wp.isfinite(mass[i]):
            wp.atomic_add(bad, 0, 1)
        rows[i] = i
        cols[i] = i
        block = mat(dtype(0))
        block[0, 0] = weight * mass[i]
        if wp.static(order == 2):
            block[1, 1] = -mass[i]
        else:
            block[1, 2] = -mass[i]
            block[2, 1] = -mass[i]
        blocks[i] = block
        hvalues[i] = shift * mass[i]
        for e in range(offsets[i], offsets[i + 1]):
            rows[n + e] = i
            cols[n + e] = columns[e]
            block = mat(dtype(0))
            if wp.static(order == 2):
                block[0, 1] = -values[e]
                block[1, 0] = -values[e]
            else:
                block[0, 2] = -values[e]
                block[2, 0] = -values[e]
                block[1, 1] = -values[e]
            blocks[n + e] = block
            hvalues[n + e] = values[e]

    @wp.kernel(enable_backward=False)
    def embed(b: wp.array(dtype=dtype), out: wp.array(dtype=vec)):
        i = wp.tid()
        v = vec(dtype(0))
        v[0] = b[i]
        out[i] = v

    @wp.kernel(enable_backward=False)
    def component(x: wp.array(dtype=vec), c: int, out: wp.array(dtype=dtype)):
        i = wp.tid()
        out[i] = x[i][c]

    @wp.kernel(enable_backward=False)
    def scaled_component(
        x: wp.array(dtype=vec), c: int, mass: wp.array(dtype=dtype), out: wp.array(dtype=dtype)
    ):
        i = wp.tid()
        out[i] = x[i][c] / mass[i]

    @wp.kernel(enable_backward=False)
    def multiply_mass(
        x: wp.array(dtype=dtype), mass: wp.array(dtype=dtype), out: wp.array(dtype=dtype)
    ):
        i = wp.tid()
        out[i] = mass[i] * x[i]

    @wp.kernel(enable_backward=False)
    def rhs(
        x: wp.array(dtype=vec),
        k1: wp.array(dtype=dtype),
        k2: wp.array(dtype=dtype),
        mass: wp.array(dtype=dtype),
        out: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        out[i] = x[i][0] - k1[i]
        if wp.static(order == 3):
            out[i] += k2[i]

    @wp.kernel(enable_backward=False)
    def auxiliary(
        x: wp.array(dtype=vec),
        c: int,
        ku: wp.array(dtype=dtype),
        mass: wp.array(dtype=dtype),
        out: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        out[i] = -(ku[i] + x[i][c]) / mass[i]

    @wp.kernel(enable_backward=False)
    def divide(x: wp.array(dtype=dtype), mass: wp.array(dtype=dtype), out: wp.array(dtype=dtype)):
        i = wp.tid()
        out[i] = x[i] / mass[i]

    @wp.kernel(enable_backward=False)
    def finish(
        u: wp.array(dtype=dtype),
        a: wp.array(dtype=dtype),
        lam: wp.array(dtype=dtype),
        y: wp.array(dtype=vec),
        z: wp.array(dtype=vec),
        alpha: dtype,
        beta: dtype,
    ):
        i = wp.tid()
        value = vec(dtype(0))
        value[0] = u[i]
        value[1] = a[i]
        if wp.static(order == 3):
            value[2] = lam[i]
        if alpha == dtype(0):
            value = vec(dtype(0))
        else:
            value *= alpha
        if beta != dtype(0):
            value += beta * y[i]
        z[i] = value

    return build, embed, component, scaled_component, multiply_mass, rhs, auxiliary, divide, finish


@lru_cache(None)
def _scaling_kernels(dtype, order):
    vec = wp.types.vector(length=order, dtype=dtype)

    @wp.kernel(enable_backward=False)
    def mass_scale(mass: wp.array(dtype=dtype), out: wp.array(dtype=dtype), scale: dtype):
        i = wp.tid()
        out[i] = scale * mass[i]

    @wp.kernel(enable_backward=False)
    def vector_scale(x: wp.array(dtype=vec), out: wp.array(dtype=vec), scale: vec):
        i = wp.tid()
        out[i] = wp.cw_mul(x[i], scale)

    return mass_scale, vector_scale


class MixedHarmonicSystem:
    """Preserve the upstream benchmark's symmetric indefinite 2/3-field systems.

    K is symmetric positive semidefinite (minus the cotangent Laplacian),
    mass is positive, and unknowns are interleaved per vertex. For order=2:
    [[alpha M,-K],[-K,-M]]. For order=3:
    [[alpha M,0,-K],[0,-K,-M],[-K,-M,0]].

    IMPORTANT: order=3 has Schur complement alpha M - K M^-1 K M^-1 K,
    not the positive triharmonic energy. These are the actual benchmark
    signs; they are not silently corrected. Use GMRES, not CG.

    ``equilibrate=True`` constructs D A D with D=(alpha^-1/4,alpha^1/4)
    for order=2, or D=(alpha^-1/3,1,alpha^1/3) for order=3. ``rhs`` and
    ``solution`` perform the corresponding transforms; ``transform`` applies
    D to a full vector. This is a change of variables, not regularization.
    In that case ``mass`` stores alpha^(1/order) times the input mass and
    ``data_weight`` is 1; ``input_data_weight`` retains the original value.
    Operator construction, transformations and preconditioning run in Warp.
    """

    def __init__(self, K, mass, order=2, data_weight=1.0, equilibrate=False):
        if not isinstance(K, sp.BsrMatrix) or K.nrow != K.ncol or K.block_shape != (1, 1):
            raise ValueError("K must be square scalar BSR")
        if K.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("K must use float32 or float64")
        if mass.dtype != K.scalar_type or mass.device != K.device or mass.shape != (K.nrow,):
            raise ValueError("mass must match K")
        if order not in (2, 3) or not math.isfinite(data_weight) or data_weight <= 0:
            raise ValueError("order must be 2 or 3 and data_weight finite positive")
        self.K = sp.bsr_copy(K)
        self.K.nnz_sync()
        self.mass = wp.clone(mass)
        self.order, self.data_weight = order, data_weight
        self.vector_type = wp.types.vector(length=order, dtype=K.scalar_type)
        self.input_data_weight = data_weight
        self.equilibrate = equilibrate
        self.scale = self.vector_type(1.0)
        self._scaling = _scaling_kernels(K.scalar_type, order)
        if equilibrate:
            root = data_weight ** (1 / order)
            self.scale = (
                self.vector_type(data_weight**-0.25, data_weight**0.25)
                if order == 2
                else self.vector_type(1 / root, 1.0, root)
            )
            wp.launch(
                self._scaling[0], K.nrow, [mass, self.mass, K.scalar_type(root)], device=K.device
            )
            self.data_weight = 1.0
            data_weight = 1.0
        self._kernels = _kernels(K.scalar_type, order)
        count = K.nrow + self.K.nnz
        rows = wp.empty(count, dtype=int, device=K.device)
        cols = wp.empty_like(rows)
        blocks = wp.empty(
            count, dtype=wp.types.matrix(shape=(order, order), dtype=K.scalar_type), device=K.device
        )
        hvalues = wp.empty(count, dtype=K.scalar_type, device=K.device)
        bad = wp.zeros(1, dtype=int, device=K.device)
        wp.launch(
            self._kernels[0],
            K.nrow,
            [
                K.nrow,
                self.K.offsets,
                self.K.columns,
                self.K.values,
                self.mass,
                K.scalar_type(data_weight),
                K.scalar_type(data_weight ** (1 / order)),
                rows,
                cols,
                blocks,
                hvalues,
                bad,
            ],
            device=K.device,
        )
        if bad.numpy()[0]:
            raise ValueError("mass must be finite positive")
        self.matrix = sp.bsr_from_triplets(K.nrow, K.ncol, rows, cols, blocks)
        self.H = sp.bsr_from_triplets(K.nrow, K.ncol, rows, cols, hvalues)

    def rhs(self, b):
        if b.dtype != self.mass.dtype or b.shape != self.mass.shape or b.device != self.mass.device:
            raise ValueError("b must match mass")
        out = wp.empty(self.K.nrow, dtype=self.vector_type, device=self.K.device)
        wp.launch(self._kernels[1], self.K.nrow, [b, out], device=self.K.device)
        if self.equilibrate:
            self.transform(out, out)
        return out

    def transform(self, x, out=None):
        """Apply D: maps original RHS to balanced RHS, or balanced solution to original."""
        if (
            x.dtype != self.vector_type
            or x.shape != self.mass.shape
            or x.device != self.mass.device
        ):
            raise ValueError("x must be an interleaved system vector")
        if out is None:
            out = wp.empty_like(x)
        if out.dtype != x.dtype or out.shape != x.shape or out.device != x.device:
            raise ValueError("out must match x")
        wp.launch(self._scaling[1], self.K.nrow, [x, out, self.scale], device=self.K.device)
        return out

    def solution(self, x):
        if (
            x.dtype != self.vector_type
            or x.shape != self.mass.shape
            or x.device != self.mass.device
        ):
            raise ValueError("x must be an interleaved system vector")
        if self.equilibrate:
            x = self.transform(x)
        out = wp.empty_like(self.mass)
        wp.launch(self._kernels[2], self.K.nrow, [x, 0, out], device=self.K.device)
        return out


class MatchingSchur(LinearOperator):
    """Block elimination using repeated approximate inverses of H=K+alpha^(1/p)M.

    ``H_inverse`` must be a fixed linear operator approximating H^-1. Defaults
    to FSAI(H). The biharmonic Schur inverse approximation is H^-1 M H^-1;
    for the benchmark's order=3 signs it is -H^-1 M H^-1 M H^-1.
    Exact H solves give a spectral bound only for the positive order=2 case.
    For order=3, near-resonant modes remain difficult. No inner adaptive CG
    is hidden in this operator: ordinary right GMRES is valid.
    """

    def __init__(self, system, H_inverse=None, **fsai_options):
        self.system = system
        self.inverse = FSAI(system.H, **fsai_options) if H_inverse is None else H_inverse
        if (
            self.inverse.shape != system.H.shape
            or self.inverse.dtype != system.H.dtype
            or self.inverse.device != system.H.device
        ):
            raise ValueError("H_inverse must match H")
        self.K = SparseOperator(system.K, row_lanes=4)
        self._buffers = [wp.empty_like(system.mass) for _ in range(7)]
        super().__init__(system.matrix.shape, system.matrix.dtype, system.K.device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        s = self.system
        kernels = s._kernels
        n, mass = s.K.nrow, s.mass
        t, v, w, g, u, a, lam = self._buffers

        def launch(index, args):
            wp.launch(kernels[index], n, args, device=self.device)

        launch(3, [x, 1, mass, t])
        self.K.matvec(t, v, v, 1.0, 0.0)
        if s.order == 3:
            launch(3, [x, 2, mass, t])
            self.K.matvec(t, w, w, 1.0, 0.0)
            launch(7, [w, mass, t])
            self.K.matvec(t, w, w, 1.0, 0.0)
        launch(5, [x, v, w, mass, g])
        self.inverse.matvec(g, u, u, 1.0, 0.0)
        for step in range(s.order - 1):
            launch(4, [u, mass, t])
            self.inverse.matvec(t, u, u, -1.0 if s.order == 3 and step == 1 else 1.0, 0.0)
        self.K.matvec(u, v, v, 1.0, 0.0)
        launch(6, [x, 1 if s.order == 2 else 2, v, mass, a])
        if s.order == 3:
            self.K.matvec(a, v, v, 1.0, 0.0)
            launch(6, [x, 1, v, mass, lam])
        launch(8, [u, a, lam, y, z, self.scalar_type(alpha), self.scalar_type(beta)])


@lru_cache(None)
def _shifted_kernels(dtype, order):
    vec = wp.types.vector(length=order, dtype=dtype)

    @wp.kernel(enable_backward=False)
    def unpack(
        x: wp.array(dtype=vec),
        a: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        c: wp.array(dtype=dtype),
        scale: dtype,
    ):
        i = wp.tid()
        if wp.static(order == 2):
            a[i] = (x[i][0] / scale - scale * x[i][1]) * dtype(0.5)
            b[i] = (-x[i][0] / scale - scale * x[i][1]) * dtype(0.5)
        else:
            a[i] = -x[i][2]
            b[i] = -x[i][1]
            c[i] = -x[i][0]

    @wp.kernel(enable_backward=False)
    def finish(
        a: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        c: wp.array(dtype=dtype),
        y: wp.array(dtype=vec),
        z: wp.array(dtype=vec),
        scale: dtype,
        alpha: dtype,
        beta: dtype,
    ):
        i = wp.tid()
        value = vec(dtype(0))
        if wp.static(order == 2):
            value[0] = a[i] / scale
            value[1] = b[i] * scale
        else:
            value[0] = a[i]
            value[1] = b[i]
            value[2] = c[i]
        if alpha == dtype(0):
            value = vec(dtype(0))
        else:
            value *= alpha
        if beta != dtype(0):
            value += beta * y[i]
        z[i] = value

    return unpack, finish


class ShiftedBlock(LinearOperator):
    """First-order block approximation using H=K+alpha^(1/order)M.

    For order=2, a balanced block rotation avoids squaring errors in the
    approximate H inverse. With exact H inverses, the balanced preconditioned
    eigenvalues are (1 +/- i*t)/2, |t|<=1. Approximate inverses have no such
    guarantee. For order=3, approximate the inverse of the leading K blocks;
    the resonance at generalized eigenvalue alpha^(1/3) remains.
    Instances own scratch; apply is capturable and supports input/output
    aliases, but must not be invoked concurrently on independent streams.
    """

    def __init__(self, system, H_inverse=None, **fsai_options):
        self.system = system
        self.inverse = FSAI(system.H, **fsai_options) if H_inverse is None else H_inverse
        if (
            self.inverse.shape != system.H.shape
            or self.inverse.dtype != system.H.dtype
            or self.inverse.device != system.H.device
        ):
            raise ValueError("H_inverse must match H")
        self._buffers = [wp.empty_like(system.mass) for _ in range(6)]
        self._kernels = _shifted_kernels(system.mass.dtype, system.order)
        super().__init__(system.matrix.shape, system.matrix.dtype, system.K.device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        s = self.system
        a, b, c, u, v, w = self._buffers
        scale = self.scalar_type(s.data_weight**0.25)
        wp.launch(self._kernels[0], s.K.nrow, [x, a, b, c, scale], device=self.device)
        self.inverse.matvec(a, u, u, 1.0, 0.0)
        self.inverse.matvec(b, v, v, 1.0, 0.0)
        if s.order == 3:
            self.inverse.matvec(c, w, w, 1.0, 0.0)
        wp.launch(
            self._kernels[1],
            s.K.nrow,
            [u, v, w, y, z, scale, self.scalar_type(alpha), self.scalar_type(beta)],
            device=self.device,
        )

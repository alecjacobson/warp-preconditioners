"""Parallel block Jacobi and block ILU(0), built and applied entirely in Warp."""

import math
from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator


@lru_cache(None)
def _kernels(dtype, size):
    mat = wp.types.matrix(shape=(size, size), dtype=dtype)
    vec = wp.types.vector(length=size, dtype=dtype)

    @wp.func
    def safe_inverse(a: mat, tolerance: dtype, bad: wp.array(dtype=int)):
        scale = dtype(0)
        for i in range(size):
            for j in range(size):
                scale = wp.max(scale, wp.abs(a[i, j]))
        inverse = mat(dtype(0))
        if scale > dtype(0) and wp.isfinite(scale):
            normalized = a / scale
            det = dtype(0)
            if wp.static(size == 1):
                det = normalized[0, 0]
            else:
                det = wp.determinant(normalized)
            if wp.isfinite(det) and wp.abs(det) > tolerance:
                if wp.static(size == 1):
                    inverse[0, 0] = dtype(1) / a[0, 0]
                else:
                    inverse = wp.inverse(normalized) / scale
            else:
                wp.atomic_add(bad, 0, 1)
        else:
            wp.atomic_add(bad, 0, 1)
        for i in range(size):
            for j in range(size):
                if not wp.isfinite(inverse[i, j]):
                    wp.atomic_add(bad, 0, 1)
        return inverse

    @wp.kernel(enable_backward=False)
    def topology(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        rows: wp.array(dtype=int),
        diagonal: wp.array(dtype=int),
        bad: wp.array(dtype=int),
    ):
        i = wp.tid()
        found = int(-1)
        for e in range(offsets[i], offsets[i + 1]):
            rows[e] = i
            if columns[e] == i:
                found = e
        diagonal[i] = found
        if found < 0:
            wp.atomic_add(bad, 0, 1)

    @wp.kernel(enable_backward=False)
    def invert(
        values: wp.array(dtype=mat),
        diagonal: wp.array(dtype=int),
        inverse: wp.array(dtype=mat),
        tolerance: dtype,
        bad: wp.array(dtype=int),
    ):
        i = wp.tid()
        if diagonal[i] >= 0:
            inverse[i] = safe_inverse(values[diagonal[i]], tolerance, bad)

    @wp.kernel(enable_backward=False)
    def diagonal_only(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        padded: bool,
        values: wp.array(dtype=mat),
        inverse: wp.array(dtype=mat),
        tolerance: dtype,
        bad: wp.array(dtype=int),
    ):
        i = wp.tid()
        lo = offsets[i]
        end = offsets[i + 1]
        if padded:
            end = lo + counts[i]
        hi = end
        while lo < hi:
            mid = (lo + hi) // 2
            if columns[mid] < i:
                lo = mid + 1
            else:
                hi = mid
        if lo < end:
            if columns[lo] == i:
                inverse[i] = safe_inverse(values[lo], tolerance, bad)
            else:
                wp.atomic_add(bad, 0, 1)
        else:
            wp.atomic_add(bad, 0, 1)

    @wp.kernel(enable_backward=False)
    def initialize(
        rows: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=mat),
        inverse: wp.array(dtype=mat),
        factors: wp.array(dtype=mat),
    ):
        e = wp.tid()
        value = values[e]
        if columns[e] < rows[e]:
            value = value * inverse[columns[e]]
        factors[e] = value

    @wp.kernel(enable_backward=False)
    def sweep(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        rows: wp.array(dtype=int),
        values: wp.array(dtype=mat),
        old: wp.array(dtype=mat),
        inverse: wp.array(dtype=mat),
        out: wp.array(dtype=mat),
        relaxation: dtype,
    ):
        e = wp.tid()
        i, j = rows[e], columns[e]
        value = values[e]
        for ik in range(offsets[i], offsets[i + 1]):
            k = columns[ik]
            if k >= wp.min(i, j):
                break
            lo = int(offsets[k])
            hi = int(offsets[k + 1])
            while lo < hi:
                mid = (lo + hi) // 2
                if columns[mid] < j:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < offsets[k + 1] and columns[lo] == j:
                value -= old[ik] * old[lo]
        if j < i:
            value = value * inverse[j]
        out[e] = (dtype(1) - relaxation) * old[e] + relaxation * value

    @wp.kernel(enable_backward=False)
    def triangular(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        factors: wp.array(dtype=mat),
        inverse: wp.array(dtype=mat),
        rhs: wp.array(dtype=vec),
        old: wp.array(dtype=vec),
        out: wp.array(dtype=vec),
        lower: bool,
    ):
        i = wp.tid()
        value = rhs[i]
        for e in range(offsets[i], offsets[i + 1]):
            j = columns[e]
            if (lower and j < i) or (not lower and j > i):
                value -= factors[e] * old[j]
        if not lower:
            value = inverse[i] * value
        out[i] = value

    @wp.kernel(enable_backward=False)
    def combine(
        x: wp.array(dtype=vec),
        y: wp.array(dtype=vec),
        z: wp.array(dtype=vec),
        alpha: dtype,
        beta: dtype,
    ):
        i = wp.tid()
        value = alpha * x[i]
        if beta != dtype(0):
            value += beta * y[i]
        z[i] = value

    @wp.kernel(enable_backward=False)
    def diagonal_apply(
        inverse: wp.array(dtype=mat),
        x: wp.array(dtype=vec),
        y: wp.array(dtype=vec),
        z: wp.array(dtype=vec),
        alpha: dtype,
        beta: dtype,
    ):
        i = wp.tid()
        value = vec(dtype(0))
        if alpha != dtype(0):
            value = alpha * (inverse[i] * x[i])
        if beta != dtype(0):
            value += beta * y[i]
        z[i] = value

    @wp.kernel(enable_backward=False)
    def validate_factors(values: wp.array(dtype=mat), bad: wp.array(dtype=int)):
        e = wp.tid()
        for i in range(size):
            for j in range(size):
                if not wp.isfinite(values[e][i, j]):
                    wp.atomic_add(bad, 0, 1)

    return (
        topology,
        invert,
        initialize,
        sweep,
        triangular,
        combine,
        diagonal_apply,
        validate_factors,
        diagonal_only,
    )


class BlockJacobi(LinearOperator):
    """Invert diagonal scalar, 2x2, 3x3 or 4x4 BSR blocks on device.

    Blocks may be indefinite but must be nonsingular. The input must have
    canonical sorted topology. Singular/near-singular local blocks raise;
    no shift or silent fallback changes the preconditioner. Use GMRES or
    BiCGSTAB for indefinite/nonsymmetric inputs, not CG.
    """

    def __init__(self, A, pivot_tolerance=None):
        if not isinstance(A, sp.BsrMatrix) or A.nrow != A.ncol:
            raise ValueError("A must be a square Warp BSR matrix")
        size = A.block_shape[0]
        if A.block_shape != (size, size) or size not in (1, 2, 3, 4):
            raise ValueError("BlockJacobi requires square blocks of size 1 through 4")
        if A.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("A must use float32 or float64")
        tolerance = (
            (1e-12 if A.scalar_type == wp.float64 else 1e-6)
            if pivot_tolerance is None
            else pivot_tolerance
        )
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("pivot_tolerance must be finite and nonnegative")
        if type(self) is BlockJacobi:
            self.A = A
            self.vector_type = wp.types.vector(length=size, dtype=A.scalar_type)
            self._value_type = wp.types.matrix(shape=(size, size), dtype=A.scalar_type)
            self._kernels = _kernels(A.scalar_type, size)
            self.inverse_diagonal = wp.empty(A.nrow, dtype=self._value_type, device=A.device)
            self._candidate = wp.empty_like(self.inverse_diagonal)
            self._bad = wp.zeros(1, dtype=int, device=A.device)
            self.pivot_tolerance = tolerance
            super().__init__(A.shape, A.dtype, A.device, self._apply)
            self.update(A)
            return
        self.A = sp.bsr_copy(A)
        self.A.nnz_sync()
        self.vector_type = wp.types.vector(length=size, dtype=A.scalar_type)
        self._value_type = wp.types.matrix(shape=(size, size), dtype=A.scalar_type)
        self._values = self.A.values.view(self._value_type)
        self._kernels = _kernels(A.scalar_type, size)
        self._rows = wp.empty(self.A.nnz, dtype=int, device=A.device)
        self._diagonal = wp.empty(A.nrow, dtype=int, device=A.device)
        self.inverse_diagonal = wp.zeros(A.nrow, dtype=self._value_type, device=A.device)
        self._bad = wp.zeros(1, dtype=int, device=A.device)
        self.pivot_tolerance = tolerance
        wp.launch(
            self._kernels[0],
            A.nrow,
            [self.A.offsets, self.A.columns, self._rows, self._diagonal, self._bad],
            device=A.device,
        )
        self._invert(self._values)
        self._check()
        super().__init__(A.shape, A.dtype, A.device, self._apply)

    def update(self, A=None):
        """Recompute diagonal inverses in existing buffers; failed updates are atomic.

        Accepts any canonical topology with the same shape/type/device. Off-diagonal
        values are never copied or inspected. Setup/update synchronize; apply does not.
        """
        if type(self) is not BlockJacobi:
            raise NotImplementedError("Rebuild BlockILU0 to update its factors")
        if A is None:
            A = self.A
        if not isinstance(A, sp.BsrMatrix) or (A.shape, A.dtype, A.device) != (
            self.shape,
            self.dtype,
            self.device,
        ):
            raise ValueError("update requires the same shape, dtype, and device")
        self._bad.zero_()
        wp.launch(
            self._kernels[8],
            A.nrow,
            [
                A.offsets,
                A.columns,
                A.row_counts if A.row_counts is not None else A.offsets,
                A.row_counts is not None,
                A.values.view(self._value_type),
                self._candidate,
                A.scalar_type(self.pivot_tolerance),
                self._bad,
            ],
            device=A.device,
        )
        self._check()
        wp.copy(self.inverse_diagonal, self._candidate)
        self.A = A
        return self

    def _invert(self, values):
        wp.launch(
            self._kernels[1],
            self.A.nrow,
            [
                values,
                self._diagonal,
                self.inverse_diagonal,
                self.A.scalar_type(self.pivot_tolerance),
                self._bad,
            ],
            device=self.A.device,
        )

    def _check(self):
        if self._bad.numpy()[0]:
            raise ValueError("Missing, nonfinite or singular/near-singular block pivot")

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(self.vector_type) for v in (x, y, z))
        wp.launch(
            self._kernels[6],
            self.A.nrow,
            [self.inverse_diagonal, x, y, z, self.scalar_type(alpha), self.scalar_type(beta)],
            device=self.device,
        )


class BlockILU0(BlockJacobi):
    """Synchronous parallel block ILU(0) with fixed Jacobi triangular sweeps.

    Each setup sweep updates all stored L/U blocks from the previous sweep,
    using the original BSR pattern (no fill). L has identity diagonal.
    Application approximates each triangular solve with ``solve_sweeps``
    Jacobi steps from zero. Counts are fixed, so this is a linear, generally
    nonsymmetric preconditioner compatible with ordinary right GMRES.

    There is no pivoting across blocks: failure raises ValueError. This is
    an experimental local preconditioner, not a robust general indefinite
    factorization. Setup reads validation counters only; apply is capturable
    and supports x/z and y/z aliases. Owns scratch; do not apply concurrently.
    """

    def __init__(self, A, factor_sweeps=12, solve_sweeps=4, relaxation=1.0, pivot_tolerance=None):
        if not isinstance(factor_sweeps, int) or factor_sweeps < 0:
            raise ValueError("factor_sweeps must be a nonnegative integer")
        if not isinstance(solve_sweeps, int) or solve_sweeps < 1:
            raise ValueError("solve_sweeps must be a positive integer")
        if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
            raise ValueError("relaxation must be in (0,1]")
        super().__init__(A, pivot_tolerance)
        self.factor_sweeps, self.solve_sweeps = factor_sweeps, solve_sweeps
        self.relaxation = relaxation
        self.factors = wp.empty(self.A.nnz, dtype=self._value_type, device=A.device)
        other = wp.empty_like(self.factors)
        wp.launch(
            self._kernels[2],
            self.A.nnz,
            [self._rows, self.A.columns, self._values, self.inverse_diagonal, self.factors],
            device=A.device,
        )
        for _ in range(factor_sweeps):
            wp.launch(
                self._kernels[3],
                self.A.nnz,
                [
                    self.A.offsets,
                    self.A.columns,
                    self._rows,
                    self._values,
                    self.factors,
                    self.inverse_diagonal,
                    other,
                    A.scalar_type(relaxation),
                ],
                device=A.device,
            )
            self.factors, other = other, self.factors
            self._invert(self.factors)
        wp.launch(self._kernels[7], self.A.nnz, [self.factors, self._bad], device=A.device)
        self._check()
        self._buffers = [
            wp.empty(A.nrow, dtype=self.vector_type, device=A.device) for _ in range(4)
        ]

    def _apply(self, x, y, z, alpha, beta):
        if alpha == 0.0:
            return BlockJacobi._apply(self, x, y, z, alpha, beta)
        x, y, z = (v.view(self.vector_type) for v in (x, y, z))
        a, b, c, d = self._buffers
        a.zero_()
        for _ in range(self.solve_sweeps):
            wp.launch(
                self._kernels[4],
                self.A.nrow,
                [
                    self.A.offsets,
                    self.A.columns,
                    self.factors,
                    self.inverse_diagonal,
                    x,
                    a,
                    b,
                    True,
                ],
                device=self.device,
            )
            a, b = b, a
        c.zero_()
        for _ in range(self.solve_sweeps):
            wp.launch(
                self._kernels[4],
                self.A.nrow,
                [
                    self.A.offsets,
                    self.A.columns,
                    self.factors,
                    self.inverse_diagonal,
                    a,
                    c,
                    d,
                    False,
                ],
                device=self.device,
            )
            c, d = d, c
        wp.launch(
            self._kernels[5],
            self.A.nrow,
            [c, y, z, self.scalar_type(alpha), self.scalar_type(beta)],
            device=self.device,
        )

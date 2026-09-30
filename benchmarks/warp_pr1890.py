# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unmodified block-Jacobi implementation from NVIDIA/warp PR #1890.

Pinned source: 0b58bcf2320b77e548bbbac0292f7f4415455eec, warp/_src/optim/linear.py.
Only the imports are adapted to reuse the installed Warp 1.15 solver types.
This benchmark-only snapshot does not patch the installed Warp package.
See third_party/warp-LICENSE.txt.
"""

import functools
from typing import Any

import warp as wp
import warp.sparse as sparse
from warp._src.logger import log_warning
from warp._src.types import type_size_in_bytes
from warp._src.optim.linear import (
    LinearOperator, _Matrix, _as_scalar_array, _make_jacobi_preconditioner,
)

_BLOCK_JACOBI_STRATEGIES = {
    "block_jacobi_auto": "auto",
    "block_jacobi_direct": "direct",
    "block_jacobi_sequential": "sequential",
    "block_jacobi_tile": "tile",
}

# Block-size thresholds used by the "auto" block-Jacobi strategy: blocks of size
# [2, _BLOCK_JACOBI_AUTO_DIRECT_MAX] use "direct", (..., _BLOCK_JACOBI_AUTO_SEQUENTIAL_MAX] use
# "sequential", and anything larger uses "tile". These thresholds are chosen heuristically, for
# dtype/robustness coverage rather than measured performance.
_BLOCK_JACOBI_AUTO_DIRECT_MAX = 6
_BLOCK_JACOBI_AUTO_SEQUENTIAL_MAX = 11

# "block_jacobi_direct"'s QR-based kernel has a one-time (per-process) compile cost that grows
# sharply with block size. Requesting "direct" above this size instead falls back to "tile",
# with a warning.
_BLOCK_JACOBI_DIRECT_MAX_BLOCK_SIZE = 8

# Relative pivot tolerance below which a diagonal block is treated as singular, per dtype
# (~eps**0.75). float16's is already ~1 ulp (eps 9.8e-4) and cannot go lower without missing
# genuinely rank-deficient blocks, so a float16 block scaled below ~1e-3 falls back to the
# identity -- see test_block_jacobi_preconditioner_well_conditioned_scaled_block.
_BLOCK_JACOBI_PIVOT_REL_TOL = {wp.float16: 1.0e-3, wp.float32: 1.0e-6, wp.float64: 1.0e-12}


def _make_block_jacobi_preconditioner(A: _Matrix, strategy: str) -> LinearOperator:
    """Build a block-Jacobi preconditioner from the diagonal blocks of a BsrMatrix.

    Validates ``A`` once, then dispatches to the ``"direct"``, ``"sequential"``, or ``"tile"``
    builder for ``strategy`` (resolving ``"auto"`` by block size first). Falls back to scalar
    Jacobi for 1x1-block (CSR) matrices; see :func:`preconditioner` for the full contract.
    """
    if not isinstance(A, sparse.BsrMatrix):
        raise ValueError("Block-Jacobi preconditioner requires a warp.sparse.BsrMatrix input")

    block_shape = A.block_shape
    if block_shape[0] != block_shape[1]:
        raise ValueError(f"Block-Jacobi preconditioner requires square matrix blocks, got shape {block_shape}")

    block_size = block_shape[0]
    if block_size == 1:
        # A CSR matrix; block-Jacobi degenerates to standard (scalar) Jacobi.
        return _make_jacobi_preconditioner(A, use_abs=False)

    tile_supports_dtype = A.scalar_type in (wp.float32, wp.float64)

    if strategy == "auto":
        if block_size <= _BLOCK_JACOBI_AUTO_DIRECT_MAX:
            strategy = "direct"
        elif block_size <= _BLOCK_JACOBI_AUTO_SEQUENTIAL_MAX or not tile_supports_dtype:
            # "tile" only supports float32/float64; for a dtype it can't handle (e.g. float16)
            # at a block size that would otherwise dispatch to "tile", use "sequential" instead
            # (it supports every floating scalar type) rather than dispatching to a strategy
            # that will raise.
            strategy = "sequential"
        else:
            strategy = "tile"

    if strategy == "direct" and block_size > _BLOCK_JACOBI_DIRECT_MAX_BLOCK_SIZE and tile_supports_dtype:
        log_warning(
            f"'block_jacobi_direct' preconditioner requested for block size {block_size}, which "
            f"is larger than {_BLOCK_JACOBI_DIRECT_MAX_BLOCK_SIZE}. Its QR-based kernel has a "
            f"steep one-time compile-time cost at larger block sizes (empirically, on the order "
            f"of minutes by block size 16); falling back to 'block_jacobi_tile' instead, which "
            f"was also faster to apply at every block size benchmarked.",
            category=UserWarning,
            stacklevel=3,
        )
        strategy = "tile"
    # If the dtype isn't one "tile" supports, keep "direct" even above the size cap: falling
    # back to "tile" would just trade a slow compile for an immediate ValueError, and "direct"
    # documents float16 support that this size cap must not silently break.

    if strategy == "direct":
        return _make_block_jacobi_direct(A, block_size)
    if strategy == "sequential":
        return _make_block_jacobi_sequential(A, block_size)
    return _make_block_jacobi_tile(A, block_size)


def _make_block_jacobi_direct(A: _Matrix, block_size: int) -> LinearOperator:
    """Build a block-Jacobi preconditioner via a dense Householder-QR block inverse.

    Not the fastest strategy to apply (benchmarking found ``"tile"`` faster at every block size
    tested); its value is ``float16`` support and zero-safe behavior (see
    :func:`_block_inverse_qr`) via plain scalar arithmetic rather than tile hardware. Callers
    should use :func:`_make_block_jacobi_preconditioner`, which caps ``block_size`` for this
    strategy and falls back to ``"tile"`` above the cap.
    """
    device = A.device
    scalar_type = A.scalar_type

    A_diag = sparse.bsr_get_diag(A)
    dim = A_diag.shape[0]
    inv_diag = wp.empty_like(A_diag)
    rel_tol = scalar_type(_BLOCK_JACOBI_PIVOT_REL_TOL.get(scalar_type, 1.0e-6))
    wp.launch(_invert_diagonal_blocks_qr, dim=dim, device=device, inputs=[A_diag, rel_tol, inv_diag])

    def block_jacobi_direct_mv(x, y, z, alpha, beta):
        """Compute ``z = alpha * (blockdiag(A)^-1 @ x) + beta * y`` via the QR block inverse."""
        wp.launch(
            _block_diag_mv_inverse,
            dim=dim,
            device=device,
            inputs=[
                inv_diag,
                _as_vector_array(_as_scalar_array(x), block_size),
                _as_vector_array(_as_scalar_array(y), block_size),
                _as_vector_array(_as_scalar_array(z), block_size),
                scalar_type(alpha),
                scalar_type(beta),
            ],
        )

    return LinearOperator((dim * block_size, dim * block_size), A.dtype, device, matvec=block_jacobi_direct_mv)


def _make_block_jacobi_sequential(A: _Matrix, block_size: int) -> LinearOperator:
    """Build a block-Jacobi preconditioner via a scalar LDL^T block factorization.

    Not the fastest strategy to apply (benchmarking found ``"tile"`` faster at every block size
    tested); its value is ``float16`` support and zero-safe behavior on non-SPD input (see
    :func:`_block_ldlt`) via plain scalar arithmetic rather than tile hardware. Requires blocks
    to be symmetric positive-definite (falls back to identity on failure, see
    :func:`_block_ldlt`).
    """
    device = A.device
    scalar_type = A.scalar_type

    A_diag = sparse.bsr_get_diag(A)
    dim = A_diag.shape[0]
    ldlt_diag = wp.empty_like(A_diag)
    rel_tol = scalar_type(_BLOCK_JACOBI_PIVOT_REL_TOL.get(scalar_type, 1.0e-6))
    wp.launch(_ldlt_diagonal_blocks, dim=dim, device=device, inputs=[A_diag, rel_tol, ldlt_diag])

    def block_jacobi_sequential_mv(x, y, z, alpha, beta):
        """Compute ``z = alpha * (blockdiag(A)^-1 @ x) + beta * y`` via the packed LDL^T solve."""
        wp.launch(
            _block_diag_mv_ldlt,
            dim=dim,
            device=device,
            inputs=[
                ldlt_diag,
                _as_vector_array(_as_scalar_array(x), block_size),
                _as_vector_array(_as_scalar_array(y), block_size),
                _as_vector_array(_as_scalar_array(z), block_size),
                scalar_type(alpha),
                scalar_type(beta),
            ],
        )

    return LinearOperator((dim * block_size, dim * block_size), A.dtype, device, matvec=block_jacobi_sequential_mv)


def _make_block_jacobi_tile(A: _Matrix, block_size: int) -> LinearOperator:
    """Build a block-Jacobi preconditioner via tile-parallel Cholesky factorization (runs on
    both CPU and GPU).

    Best suited to large blocks. Requires blocks to be genuinely symmetric positive-definite (no
    identity fallback, unlike the ``"direct"``/``"sequential"`` strategies), and ``A``'s scalar
    type to be ``float32`` or ``float64`` (the types supported by :func:`warp.tile_cholesky`).
    """
    scalar_type = A.scalar_type
    if scalar_type not in (wp.float32, wp.float64):
        raise ValueError(
            f"'block_jacobi_tile' preconditioner requires a float32 or float64 scalar type, got {scalar_type.__name__}"
        )
    device = A.device

    # One matrix block per row, laid out contiguously; reinterpret as a flat
    # (dim * block_size, block_size) scalar array so the Cholesky kernels below
    # can address each block with a simple tile_load/tile_store offset.
    A_diag = sparse.bsr_get_diag(A)
    dim = A_diag.shape[0]
    A_flat = A_diag.view(scalar_type).reshape((dim * block_size, block_size))
    L_flat = wp.empty(shape=A_flat.shape, dtype=scalar_type, device=device)

    factorize_kernel, solve_kernel = _create_block_jacobi_tile_kernels(block_size)
    wp.launch_tiled(factorize_kernel, dim=[dim], inputs=[A_flat, L_flat], block_dim=32, device=device)

    def block_jacobi_tile_mv(x, y, z, alpha, beta):
        """Compute ``z = alpha * (blockdiag(A)^-1 @ x) + beta * y``, the block-Jacobi matvec."""
        wp.launch_tiled(
            solve_kernel,
            dim=[dim],
            inputs=[
                L_flat,
                _as_scalar_array(x),
                _as_scalar_array(y),
                _as_scalar_array(z),
                scalar_type(alpha),
                scalar_type(beta),
            ],
            block_dim=32,
            device=device,
        )

    return LinearOperator(
        (dim * block_size, dim * block_size),
        A.dtype,
        device,
        matvec=block_jacobi_tile_mv,
    )


@functools.cache
def _create_block_jacobi_tile_kernels(block_size: int):
    """Build and cache the tile-Cholesky factorize/solve kernel pair for one block size.

    Used by the ``block_jacobi_tile`` preconditioner strategy.
    """

    @wp.kernel(module="unique")
    def factorize_kernel(
        A_flat: wp.array2d[Any],
        L_flat: wp.array2d[Any],
    ):
        """Cholesky-factorize one ``block_size x block_size`` diagonal block per tile.

        Computes ``L L^T = A_flat``'s block.
        """
        i = wp.tid()
        off = i * block_size
        A_tile = wp.tile_load(A_flat, shape=(block_size, block_size), offset=(off, 0))
        L_tile = wp.tile_cholesky(A_tile)
        wp.tile_store(L_flat, L_tile, offset=(off, 0))

    @wp.kernel(module="unique")
    def solve_kernel(
        L_flat: wp.array2d[Any],
        x: wp.array[Any],
        y: wp.array[Any],
        z: wp.array[Any],
        alpha: Any,
        beta: Any,
    ):
        """Apply one block-diagonal inverse per tile and write the result directly into ``z``.

        Computes ``z = alpha * (A_diag^-1 x) + beta * y`` via forward/backward substitution on
        ``L_flat``, writing straight into the caller-provided ``z`` tile with no shared scratch
        buffer, so concurrent calls on different streams cannot race on intermediate state.
        """
        i = wp.tid()
        off = i * block_size
        L_tile = wp.tile_load(L_flat, shape=(block_size, block_size), offset=(off, 0))
        zero = type(alpha)(0.0)
        if alpha == zero:
            out = wp.tile_zeros(shape=(block_size,), dtype=x.dtype)
        else:
            rhs = wp.tile_load(x, shape=(block_size,), offset=(off,))
            wp.tile_cholesky_solve_inplace(L_tile, rhs)
            out = rhs * alpha

        if beta != zero:
            out += wp.tile_load(y, shape=(block_size,), offset=(off,)) * beta
        wp.tile_store(z, out, offset=(off,))

    return factorize_kernel, solve_kernel


# --- Direct (QR) and sequential (LDL^T) block-Jacobi helpers -----------------
#
# Local re-implementations of QR-based dense inversion and LDL^T factorization for small,
# fixed-size blocks. Kept as plain wp.funcs operating on fixed-size wp.matrix/wp.vector types
# (no wp.tile_* involved), so they work for any Warp floating scalar type and don't require
# tile hardware.


@wp.func
def _qr_decomposition(A: Any):
    """Compute a Householder QR factorization of a square fixed-size matrix.

    Returns ``(Q, R)`` such that ``A = Q R``, with ``Q`` orthonormal and ``R`` upper triangular.
    """
    x = type(A[0])()
    Q = wp.identity(n=x.length, dtype=A.dtype)

    zero = x.dtype(0.0)
    two = x.dtype(2.0)

    for i in range(x.length):
        for k in range(x.length):
            x[k] = wp.where(k < i, zero, A[k, i])

        alpha = wp.length(x) * wp.sign(x[i])
        x[i] += alpha
        two_over_x_sq = wp.where(alpha == zero, zero, two / wp.length_sq(x))

        A -= wp.outer(two_over_x_sq * x, x * A)
        Q -= wp.outer(Q * x, two_over_x_sq * x)

    return Q, A


@wp.func
def _solve_upper(R: Any, b: Any):
    """Solve ``R x = b`` for an upper-triangular ``R``, zero-safe on a zero diagonal entry."""
    zero = b.dtype(0)
    x = type(b)(zero)
    for i in range(b.length, 0, -1):
        j = i - 1
        r = b[j] - wp.dot(R[j], x)
        x[j] = wp.where(R[j, j] == zero, zero, r / R[j, j])
    return x


@wp.func
def _block_inverse_qr(A: Any, rel_tol: Any):
    """Invert a square fixed-size matrix via Householder QR.

    Falls back to the identity when the block is numerically singular, mirroring the zero-safe
    convention of the scalar Jacobi preconditioner, so the result stays well-defined for
    non-invertible blocks. A pivot is treated as singular when it is non-finite or small
    relative to the largest pivot magnitude in the block, rather than only when it is exactly
    zero: for a genuinely rank-deficient block, roundoff during the QR factorization typically
    leaves a tiny but nonzero diagonal entry on ``R`` rather than an exact zero, which an
    exact-zero check misses and then divides by in the triangular solve below.

    ``rel_tol`` is the dtype's relative pivot tolerance (``_BLOCK_JACOBI_PIVOT_REL_TOL``); a
    single constant across dtypes would reject well-conditioned, merely differently-scaled
    blocks such as ``diag(1, 1e-4)``.
    """
    Q, R = _qr_decomposition(A)
    row = type(A[0])()
    zero = A.dtype(0)
    one = A.dtype(1)

    max_pivot = zero
    for j in range(row.length):
        max_pivot = wp.max(max_pivot, wp.abs(R[j, j]))

    singular = wp.bool(False)
    for j in range(row.length):
        pivot = R[j, j]
        singular = singular or (not wp.isfinite(pivot)) or (wp.abs(pivot) <= rel_tol * max_pivot)

    A_inv = type(A)()
    for i in range(row.length):
        A_inv[i] = _solve_upper(R, Q[i])  # i-th column of Q^T
    inv = wp.transpose(A_inv)

    for i in range(row.length):
        for j in range(row.length):
            inv[i, j] = wp.where(singular, wp.where(i == j, one, zero), inv[i, j])
    return inv


@wp.func
def _block_ldlt(A: Any, rel_tol: Any):
    """Factorize a symmetric fixed-size matrix as ``A = L D L^T``.

    ``L`` is unit lower triangular and ``D`` is diagonal; both are packed into a single returned
    matrix ``M`` with ``M[i, i] = D[i]`` and ``M[i, j] = L[i, j]`` for ``i > j`` (``L``'s implicit
    unit diagonal is not stored). Falls back to the identity on non-SPD input, so
    :func:`_apply_ldlt` becomes a pass-through on that block and the preconditioner stays
    well-defined. A pivot is treated as non-SPD when it is non-finite or small relative to the
    block's diagonal scale, rather than only when it is ``<= 0``: for a genuinely rank-deficient
    block, LDL^T roundoff can leave a tiny positive residual pivot instead of an exact
    non-positive value, which a ``<= 0`` check misses and then divides by.

    ``rel_tol`` is the dtype's relative pivot tolerance -- see :func:`_block_inverse_qr`.
    """
    row = type(A[0])()
    zero = A.dtype(0.0)

    max_diag = zero
    for j in range(row.length):
        max_diag = wp.max(max_diag, wp.abs(A[j, j]))

    M = type(A)(zero)
    spd = wp.bool(True)
    for j in range(row.length):
        d = A[j, j]
        for k in range(j):
            d -= M[j, k] * M[j, k] * M[k, k]

        if (not wp.isfinite(d)) or (d <= rel_tol * max_diag):
            spd = False
            break

        M[j, j] = d
        for i in range(j + 1, row.length):
            t = A[i, j]
            for k in range(j):
                t -= M[i, k] * M[j, k] * M[k, k]
            M[i, j] = t / d

    if spd:
        return M
    return wp.identity(n=row.length, dtype=A.dtype)


@wp.func
def _apply_ldlt(M: Any, x: Any):
    """Solve ``(L D L^T) z = x`` given the packed LDL^T form returned by :func:`_block_ldlt`."""
    zero = x.dtype(0)

    # Forward: L y = x (unit lower).
    y = type(x)(zero)
    for i in range(x.length):
        r = x[i]
        for j in range(i):
            r -= M[i, j] * y[j]
        y[i] = r

    # Backward with D^{-1} folded in: solve L^T z = D^{-1} y (unit upper).
    z = type(x)(zero)
    for i in range(x.length, 0, -1):
        idx = i - 1
        r = y[idx] / M[idx, idx]
        for j in range(idx + 1, x.length):
            r -= M[j, idx] * z[j]
        z[idx] = r
    return z


@wp.kernel(module="unique")
def _invert_diagonal_blocks_qr(
    diag: wp.array[Any],
    rel_tol: Any,
    inv_diag: wp.array[Any],
):
    """Invert one diagonal block per thread via Householder QR."""
    i = wp.tid()
    inv_diag[i] = _block_inverse_qr(diag[i], rel_tol)


@wp.kernel(module="unique")
def _ldlt_diagonal_blocks(
    diag: wp.array[Any],
    rel_tol: Any,
    ldlt_diag: wp.array[Any],
):
    """Factorize one diagonal block per thread into packed LDL^T form."""
    i = wp.tid()
    ldlt_diag[i] = _block_ldlt(diag[i], rel_tol)


@wp.kernel(module="unique")
def _block_diag_mv_inverse(
    inv: wp.array[Any],
    x: wp.array[Any],
    y: wp.array[Any],
    z: wp.array[Any],
    alpha: Any,
    beta: Any,
):
    """Apply one precomputed block inverse per thread and write ``z = alpha * (inv x) + beta * y``."""
    i = wp.tid()
    zero = type(alpha)(0)
    s = z.dtype(zero)
    if alpha != zero:
        s = alpha * (inv[i] * x[i])
    if beta != zero:
        s += beta * y[i]
    z[i] = s


@wp.kernel(module="unique")
def _block_diag_mv_ldlt(
    ldlt: wp.array[Any],
    x: wp.array[Any],
    y: wp.array[Any],
    z: wp.array[Any],
    alpha: Any,
    beta: Any,
):
    """Apply one packed LDL^T solve per thread and write ``z = alpha * (M^-1 x) + beta * y``."""
    i = wp.tid()
    zero = type(alpha)(0)
    s = z.dtype(zero)
    if alpha != zero:
        s = alpha * _apply_ldlt(ldlt[i], x[i])
    if beta != zero:
        s += beta * y[i]
    z[i] = s



def _as_vector_array(x: wp.array, length: int):
    """View a 1-D scalar array as a 1-D array of fixed-length vectors.

    The underlying storage is shared (no copy); ``x`` must be contiguous along its last
    dimension, and that dimension's length must be divisible by ``length``.
    """
    if length == 1:
        return x
    if x.strides[-1] != type_size_in_bytes(x.dtype):
        raise ValueError("Array is not contiguous along its last dimension")
    if x.shape[-1] % length != 0:
        raise ValueError(f"Array length {x.shape[-1]} is not divisible by block size {length}")

    vec_type = wp.types.vector(length=length, dtype=x.dtype)
    arr = wp.array(
        ptr=x.ptr,
        shape=(*x.shape[:-1], x.shape[-1] // length),
        strides=(*x.strides[:-1], x.strides[-1] * length),
        dtype=vec_type,
        device=x.device,
        grad=None if x.grad is None else _as_vector_array(x.grad, length),
    )
    arr._ref = x
    return arr

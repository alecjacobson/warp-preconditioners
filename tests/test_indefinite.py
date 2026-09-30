import numpy as np
import pytest
import warp as wp
import warp.sparse as sp
from test_fsai import DEVICES, dense, from_scipy
from warp.optim import linear

from warp_preconditioners.block_ilu import BlockILU0, BlockJacobi
from warp_preconditioners.mixed import MatchingSchur, MixedHarmonicSystem, ShiftedBlock


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_block_factors_and_aliases(device, dtype, size):
    rng = np.random.default_rng(61)
    n = 5
    blocks = rng.normal(size=(n, n, size, size)) * 0.03
    for i in range(n):
        blocks[i, i] = np.diag(np.arange(1, size + 1) * np.where(np.arange(size) % 2, -1.0, 1.0))
    a = blocks.transpose(0, 2, 1, 3).reshape(n * size, n * size)
    A = sp.bsr_copy(from_scipy(a, device, dtype), block_shape=(size, size))
    pre = BlockILU0(A, factor_sweeps=2 * n + 2, solve_sweeps=n)
    jac = BlockJacobi(A)
    expected = np.linalg.inv(a)
    vec = wp.types.vector(length=size, dtype=dtype)
    for alias in ["none", "xz", "yz", "xyz"]:
        x = wp.array(rng.normal(size=(n, size)), dtype=vec, device=device)
        y = x if alias == "xyz" else wp.array(rng.normal(size=(n, size)), dtype=vec, device=device)
        z = x if alias in ["xz", "xyz"] else y if alias == "yz" else wp.empty_like(x)
        target = 0.7 * (expected @ x.numpy().ravel()) - 0.2 * y.numpy().ravel()
        pre.matvec(x, y, z, 0.7, -0.2)
        np.testing.assert_allclose(
            z.numpy().ravel(),
            target,
            rtol=2e-5 if dtype == wp.float32 else 1e-12,
            atol=1e-6 if dtype == wp.float32 else 1e-12,
        )
    x = wp.array(rng.normal(size=(n, size)), dtype=vec, device=device)
    y = wp.empty_like(x)
    jac.matvec(x, y, y, 1.0, 0.0)
    target = np.stack([np.linalg.solve(blocks[i, i], x.numpy()[i]) for i in range(n)])
    np.testing.assert_allclose(y.numpy(), target, rtol=1e-6, atol=1e-6)
    if device == "cuda:0":
        with wp.ScopedCapture(device=device) as cap:
            pre.matvec(x, y, y, 1.0, 0.0)
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(
            y.numpy().ravel(),
            expected @ x.numpy().ravel(),
            rtol=2e-5 if dtype == wp.float32 else 1e-12,
            atol=1e-6 if dtype == wp.float32 else 1e-12,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [2, 3])
def test_mixed_signs_matching_and_gmres(device, order):
    rng = np.random.default_rng(7)
    n = 7
    edges = rng.uniform(0.1, 0.3, size=(n, n))
    edges = (edges + edges.T) / 2
    k = np.diag(edges.sum(axis=1)) - edges
    mass = rng.uniform(0.7, 1.2, size=n)
    alpha = 0.04
    with wp.ScopedDevice(device):
        system = MixedHarmonicSystem(
            from_scipy(k, device), wp.array(mass, dtype=wp.float64), order=order, data_weight=alpha
        )
        M = np.diag(mass)
        zero = np.zeros_like(M)
        expected = (
            np.block([[alpha * M, -k], [-k, -M]])
            if order == 2
            else np.block([[alpha * M, zero, -k], [zero, -k, -M], [-k, -M, zero]])
        )
        permutation = np.arange(n * order).reshape(order, n).T.ravel()
        expected = expected[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(dense(sp.bsr_copy(system.matrix, block_shape=(1, 1))), expected)
        primary = np.arange(0, n * order, order)
        auxiliary = np.setdiff1d(np.arange(n * order), primary)
        schur = expected[np.ix_(primary, primary)] - expected[
            np.ix_(primary, auxiliary)
        ] @ np.linalg.solve(
            expected[np.ix_(auxiliary, auxiliary)], expected[np.ix_(auxiliary, primary)]
        )
        invmass = np.diag(1 / mass)
        target_schur = (
            alpha * M + k @ invmass @ k if order == 2 else alpha * M - k @ invmass @ k @ invmass @ k
        )
        np.testing.assert_allclose(schur, target_schur, rtol=1e-12, atol=1e-12)
        pre = MatchingSchur(system, max_row_size=n, kap_tolerance=0)
        x = wp.array(rng.normal(size=(n, order)), dtype=system.vector_type)
        y = wp.empty_like(x)
        pre.matvec(x, y, y, 1.0, 0.0)
        r = x.numpy()
        H = k + alpha ** (1 / order) * M
        invH = np.linalg.inv(H)
        if order == 2:
            g = r[:, 0] - k @ (r[:, 1] / mass)
            u = invH @ M @ invH @ g
            result = np.column_stack([u, -(k @ u + r[:, 1]) / mass])
        else:
            g = r[:, 0] - k @ (r[:, 1] / mass) + k @ ((k @ (r[:, 2] / mass)) / mass)
            u = -invH @ M @ invH @ M @ invH @ g
            aux = -(k @ u + r[:, 2]) / mass
            result = np.column_stack([u, aux, -(k @ aux + r[:, 1]) / mass])
        np.testing.assert_allclose(y.numpy(), result, rtol=2e-11, atol=2e-11)
        shifted = ShiftedBlock(system, max_row_size=n, kap_tolerance=0)
        shifted.matvec(x, y, y, 1.0, 0.0)
        if order == 2:
            scale = alpha**0.25
            target = np.column_stack(
                [
                    invH @ ((r[:, 0] / scale - scale * r[:, 1]) * 0.5) / scale,
                    invH @ ((-r[:, 0] / scale - scale * r[:, 1]) * 0.5) * scale,
                ]
            )
        else:
            target = np.column_stack([-invH @ r[:, 2], -invH @ r[:, 1], -invH @ r[:, 0]])
        np.testing.assert_allclose(y.numpy(), target, rtol=1e-11, atol=1e-11)
        if device == "cuda:0":
            with wp.ScopedCapture(device=device) as cap:
                shifted.matvec(x, y, y, 1.0, 0.0)
            wp.capture_launch(cap.graph)
            np.testing.assert_allclose(y.numpy(), target, rtol=1e-11, atol=1e-11)
        truth = rng.normal(size=(n, order))
        b = wp.array((expected @ truth.ravel()).reshape(n, order), dtype=system.vector_type)
        solution = wp.zeros_like(b)
        linear.gmres(
            system.matrix,
            b,
            solution,
            M=BlockILU0(system.matrix, factor_sweeps=20, solve_sweeps=n),
            tol=1e-11,
            maxiter=60,
            restart=15,
        )
        np.testing.assert_allclose(solution.numpy(), truth, rtol=1e-8, atol=1e-8)


def test_bad_block_pivot():
    a = sp.bsr_copy(from_scipy(np.diag([1.0, 0.0]), "cpu"), block_shape=(2, 2))
    with pytest.raises(ValueError, match="pivot"):
        BlockJacobi(a)


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_ilu_pattern_and_fixed_sweep_linearity(device):
    rng = np.random.default_rng(83)
    n, size = 9, 2
    blocks = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                blocks[i, j] = np.diag([3.0, -2.0])
            elif abs(i - j) == 1 or (i + j) % 5 == 0:
                blocks[i, j] = rng.normal(size=(size, size)) * 0.08
    expected = {key: value.copy() for key, value in blocks.items()}
    for i in range(n):
        for j in sorted(j for row, j in blocks if row == i):
            value = blocks[i, j].copy()
            for k in range(min(i, j)):
                if (i, k) in expected and (k, j) in expected:
                    value -= expected[i, k] @ expected[k, j]
            expected[i, j] = value @ np.linalg.inv(expected[j, j]) if j < i else value
    a = np.zeros((size * n, size * n))
    for (i, j), value in blocks.items():
        a[size * i : size * (i + 1), size * j : size * (j + 1)] = value
    A = sp.bsr_copy(from_scipy(a, device), block_shape=(size, size))
    pre = BlockILU0(A, factor_sweeps=2 * n + 2, solve_sweeps=3)
    offsets, columns = pre.A.offsets.numpy(), pre.A.columns.numpy()
    factors = pre.factors.numpy()
    assert len(factors) == len(blocks)
    for i in range(n):
        for e in range(offsets[i], offsets[i + 1]):
            np.testing.assert_allclose(factors[e], expected[i, columns[e]], rtol=1e-12, atol=1e-12)

    def apply(v):
        x = wp.array(v, dtype=wp.vec2d, device=device)
        y = wp.empty_like(x)
        pre.matvec(x, y, y, 1.0, 0.0)
        return y.numpy()

    x, y = rng.normal(size=(2, n, size))
    np.testing.assert_allclose(
        apply(0.3 * x + 0.7 * y), 0.3 * apply(x) + 0.7 * apply(y), rtol=1e-12, atol=1e-12
    )
    xw = wp.full(n, wp.vec2d(float("nan")), dtype=wp.vec2d, device=device)
    yw = wp.array(y, dtype=wp.vec2d, device=device)
    pre.matvec(xw, yw, xw, 0.0, 0.7)
    np.testing.assert_allclose(xw.numpy(), 0.7 * y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [2, 3])
def test_mixed_equilibration_preserves_equations(device, order):
    n = 5
    k = np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    mass = np.linspace(0.6, 1.3, n)
    with wp.ScopedDevice(device):
        K = from_scipy(k, device)
        mw = wp.array(mass, dtype=wp.float64)
        original = MixedHarmonicSystem(K, mw, order=order, data_weight=1e-4)
        balanced = MixedHarmonicSystem(K, mw, order=order, data_weight=1e-4, equilibrate=True)
        a = dense(sp.bsr_copy(original.matrix, block_shape=(1, 1)))
        s = np.tile(np.asarray(balanced.scale), n)
        np.testing.assert_allclose(
            dense(sp.bsr_copy(balanced.matrix, block_shape=(1, 1))),
            s[:, None] * a * s[None, :],
            rtol=1e-12,
            atol=1e-12,
        )
        b = wp.array(np.linspace(-1, 1, n), dtype=wp.float64)
        rhs = balanced.rhs(b)
        np.testing.assert_allclose(rhs.numpy().ravel(), original.rhs(b).numpy().ravel() * s)
        x = wp.zeros_like(rhs)
        linear.gmres(
            balanced.matrix,
            rhs,
            x,
            M=ShiftedBlock(balanced, max_row_size=n, kap_tolerance=0),
            tol=1e-12,
            restart=15,
            maxiter=60,
        )
        result = balanced.transform(x).numpy().ravel()
        np.testing.assert_allclose(a @ result, original.rhs(b).numpy().ravel(), atol=1e-9)
        np.testing.assert_array_equal(balanced.solution(x).numpy(), result.reshape(n, order)[:, 0])

import numpy as np
import pytest
import scipy.sparse as ss
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sp

from warp_preconditioners import FSAI

DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")
    ),
]


def from_scipy(a, device, dtype=wp.float64):
    c = ss.coo_matrix(a)
    return sp.bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(c.row, dtype=int, device=device),
        wp.array(c.col, dtype=int, device=device),
        wp.array(c.data, dtype=dtype, device=device),
    )


def dense(a):
    return ss.csr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()),
        shape=a.shape,
    ).toarray()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("step_size", [1, 3])
def test_exact_full_pattern(device, dtype, step_size):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 8))
    a = a @ a.T + np.eye(8)
    m = FSAI(from_scipy(a, device, dtype), max_row_size=8, kap_tolerance=0, max_step_size=step_size)
    g = dense(m.G)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=1e-5, atol=1e-6)
    assert np.allclose(g, np.tril(g))
    x = wp.array(rng.normal(size=8), dtype=dtype, device=device)
    y = wp.array(rng.normal(size=8), dtype=dtype, device=device)
    expected = 1.7 * g.T @ g @ x.numpy() - 0.3 * y.numpy()
    m.matvec(x, y, y, 1.7, -0.3)
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_jacobi_blocks_and_cg(device):
    a = np.diag(np.full(12, 3.0)) + np.diag(np.full(11, -1.0), 1) + np.diag(np.full(11, -1.0), -1)
    A = sp.bsr_copy(from_scipy(a, device), block_shape=(3, 3))
    m = FSAI(A, max_row_size=1)
    np.testing.assert_allclose(dense(m.G), np.eye(12) / np.sqrt(3))
    m = FSAI(A, max_row_size=8)
    b = wp.array(np.ones((4, 3)), dtype=wp.vec3d, device=device)
    x = wp.zeros_like(b)
    with wp.ScopedDevice(device):
        linear.cg(A, b, x, M=m, tol=1e-12, maxiter=100)
    np.testing.assert_allclose(a @ x.numpy().ravel(), 1.0, atol=1e-10)


def test_invalid():
    with pytest.raises(ValueError, match="positive diagonals"):
        FSAI(from_scipy(np.diag([1.0, 0.0]), "cpu"))


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_reference_and_aliases(device):
    # Independent solve on every chosen support, including severe equilibration.
    rng = np.random.default_rng(18)
    a = np.diag(np.full(20, 4.0)) + np.diag(np.full(19, -1.0), 1) + np.diag(np.full(19, -1.0), -1)
    scale = np.geomspace(1e-4, 1e4, 20)
    a = scale[:, None] * a * scale[None, :]
    m = FSAI(from_scipy(a, device), max_row_size=5, kap_tolerance=0)
    g = dense(m.G)
    assert np.linalg.eigvalsh(g.T @ g).min() > 0
    assert np.count_nonzero(g, axis=1).max() <= 5
    for i in range(20):
        support = np.flatnonzero(g[i])
        e = np.zeros(len(support))
        e[-1] = 1
        ref = np.linalg.solve(a[np.ix_(support, support)], e)
        ref /= np.sqrt(ref[-1])
        np.testing.assert_allclose(g[i, support], ref, rtol=1e-10, atol=1e-12)
    for alias in ["none", "xz", "yz", "xy", "xyz"]:
        x = wp.array(rng.normal(size=20), dtype=wp.float64, device=device)
        y = (
            x
            if alias in ["xy", "xyz"]
            else wp.array(rng.normal(size=20), dtype=wp.float64, device=device)
        )
        z = x if alias in ["xz", "xyz"] else y if alias == "yz" else wp.empty_like(x)
        expected = 0.7 * g.T @ g @ x.numpy() - 1.2 * y.numpy()
        m.matvec(x, y, z, 0.7, -1.2)
        np.testing.assert_allclose(z.numpy(), expected, rtol=1e-12, atol=1e-10)
    # NaN affine operand must be ignored for beta=0.
    x = wp.array(rng.normal(size=20), dtype=wp.float64, device=device)
    y = wp.empty_like(x)
    z = wp.empty_like(x)
    y.fill_(float("nan"))
    m.matvec(x, y, y, 1.0, 0.0)
    assert np.isfinite(y.numpy()).all()
    if device == "cuda:0":
        with wp.ScopedCapture(device=device) as capture:
            m.matvec(x, y, z, 1.0, 0.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(z.numpy(), g.T @ g @ x.numpy(), rtol=1e-12, atol=1e-10)


def test_small_pivot_retains_spd_factor():
    a = np.array([[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]])
    m = FSAI(from_scipy(a, "cpu"), pivot_floor=1e-12)
    assert m.truncated_rows == 1
    g = dense(m.G)
    assert np.isfinite(g).all()
    assert np.linalg.eigvalsh(g.T @ g).min() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("step_size", [1, 3])
def test_adaptive_support_matches_independent_greedy_search(device, dtype, step_size):
    # Overlapping frontiers, stored zero entries inside blocks, and scaling
    # exercise candidate deduplication and the incremental forward solve.
    rng = np.random.default_rng(193)
    a = rng.normal(size=(18, 18))
    a[rng.random(a.shape) < 0.65] = 0
    a = a + a.T
    np.fill_diagonal(a, np.sum(np.abs(a), axis=1) + 2)
    scaling = np.geomspace(0.01, 100, 18)
    a *= scaling[:, None] * scaling[None, :]
    raw = sp.bsr_copy(from_scipy(a, device, dtype), block_shape=(3, 3))
    equilibrated = a / np.sqrt(np.diag(a)[:, None] * np.diag(a)[None, :])
    for tolerance in [0.0, 0.003, 0.1]:
        pre = FSAI(raw, max_row_size=8, kap_tolerance=tolerance, max_step_size=step_size)
        g = dense(pre.G)
        for i in range(len(a)):
            support = [i]
            z = np.ones(1)
            while len(support) < 8:
                frontier = sorted(
                    set(np.flatnonzero(np.any(a[support] != 0, axis=0))) - set(support)
                )
                frontier = [c for c in frontier if c < i]
                if not frontier:
                    break
                scores = [abs(equilibrated[c, support] @ z) for c in frontier]
                selected = sorted(zip(scores, frontier), key=lambda item: (-item[0], item[1]))
                selected = [c for score, c in selected if score > 0][
                    : min(step_size, 8 - len(support))
                ]
                if not selected:
                    break
                support.extend(selected)
                old = z[0]
                rhs = np.zeros(len(support))
                rhs[0] = 1
                z = np.linalg.solve(equilibrated[np.ix_(support, support)], rhs)
                if 1 - old / z[0] <= tolerance:
                    break
            expected = np.zeros(len(a))
            expected[support] = z / np.sqrt(z[0] * np.diag(a)[support])
            np.testing.assert_allclose(g[i], expected, rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_width_cap_and_pivot_guard(device):
    a = np.array([[1.0, 1.0 - 1e-14, 0.0], [1.0 - 1e-14, 1.0, 0.0], [0.0, 0.0, 2.0]])
    pre = FSAI(from_scipy(a, device), max_row_size=2, max_step_size=6)
    assert pre.truncated_rows == 1
    np.testing.assert_allclose(dense(pre.G), np.diag(1 / np.sqrt(np.diag(a))))
    pre = FSAI(from_scipy(a, device), max_row_size=1, max_step_size=6)
    np.testing.assert_allclose(dense(pre.G), np.diag(1 / np.sqrt(np.diag(a))))


def test_invalid_step_size():
    a = from_scipy(np.eye(3), "cpu")
    for value in [0, -1, 65, 1.5]:
        with pytest.raises(ValueError, match="max_step_size"):
            FSAI(a, max_step_size=value)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("step_size", [2, 3])
def test_batch_grows_through_narrow_frontier(device, step_size):
    # A path exposes only one candidate per search. A short batch must not
    # consume the budget for a full batch and silently stop below the cap.
    a = np.diag(np.full(12, 3.0)) + np.diag(-np.ones(11), 1) + np.diag(-np.ones(11), -1)
    pre = FSAI(from_scipy(a, device), max_row_size=8, max_step_size=step_size, kap_tolerance=0.0)
    g = dense(pre.G)
    np.testing.assert_array_equal(np.count_nonzero(g, axis=1), np.minimum(np.arange(1, 13), 8))
    for i in range(12):
        support = np.arange(max(0, i - 7), i + 1)
        rhs = np.zeros(len(support))
        rhs[-1] = 1
        z = np.linalg.solve(a[np.ix_(support, support)], rhs)
        np.testing.assert_allclose(g[i, support], z / np.sqrt(z[-1]), rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_tolerance_disables_roundoff_stopping(device):
    n = 32
    a = np.diag(np.full(n, 3.0)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    pre = FSAI(from_scipy(a, device), max_row_size=n, kap_tolerance=0, max_step_size=3)
    np.testing.assert_array_equal(np.diff(pre.G.offsets.numpy()), np.arange(1, n + 1))
    g = dense(pre.G)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=1e-12, atol=1e-15)

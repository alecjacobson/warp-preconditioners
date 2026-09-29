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
def test_exact_full_pattern(device, dtype):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 8))
    a = a @ a.T + np.eye(8)
    m = FSAI(from_scipy(a, device, dtype), max_row_size=8, kap_tolerance=0)
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

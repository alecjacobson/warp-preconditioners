import numpy as np
import pytest
import scipy.sparse as ss
import warp as wp
import warp.optim.linear as linear
from test_fsai import DEVICES, dense, from_scipy

from warp_preconditioners import FSAI, SquaredLaplacianOperator


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_factored_dirichlet_operator_and_rhs(device, dtype, lanes):
    rng = np.random.default_rng(27)
    n = 67
    laplacian = ss.diags(
        [-np.ones(n - 1), np.r_[1.0, np.full(n - 2, 2.0), 1.0], -np.ones(n - 1)], [-1, 0, 1]
    ).tocsr()
    mass = rng.uniform(0.4, 2, size=n).astype(np.float32 if dtype == wp.float32 else np.float64)
    free = rng.permutation(np.arange(3, n - 4)).astype(np.int32)
    A = SquaredLaplacianOperator(
        from_scipy(laplacian, device, dtype),
        wp.array(mass, dtype=dtype, device=device),
        wp.array(free, dtype=int, device=device),
        row_lanes=lanes,
    )
    q = laplacian @ ss.diags(1 / mass) @ laplacian
    reduced = q[free][:, free]
    rtol, atol = (3e-5, 3e-6) if dtype == wp.float32 else (1e-12, 1e-12)
    for alias in ["none", "xz", "yz", "xyz"]:
        x = wp.array(rng.normal(size=len(free)), dtype=dtype, device=device)
        y = (
            x
            if alias == "xyz"
            else wp.array(rng.normal(size=len(free)), dtype=dtype, device=device)
        )
        z = x if alias in ["xz", "xyz"] else y if alias == "yz" else wp.empty_like(x)
        expected = 0.7 * (reduced @ x.numpy()) - 0.2 * y.numpy()
        A.matvec(x, y, z, 0.7, -0.2)
        np.testing.assert_allclose(z.numpy(), expected, rtol=rtol, atol=atol)
    bc = rng.normal(size=n).astype(mass.dtype)
    prescribed = bc.copy()
    prescribed[free] = 0
    bw = wp.array(bc, dtype=dtype, device=device)
    result = A.rhs(bw)
    np.testing.assert_allclose(result.numpy(), -(q @ prescribed)[free], rtol=rtol, atol=atol)
    x = wp.ones(len(free), dtype=dtype, device=device)
    y = wp.empty_like(x)
    y.fill_(float("nan"))
    z = wp.empty_like(x)
    A.matvec(x, y, z, 1.0, 0.0)
    np.testing.assert_allclose(z.numpy(), reduced @ np.ones(len(free)), rtol=rtol, atol=atol)
    if device == "cuda:0":
        with wp.ScopedCapture(device=device) as cap:
            A.matvec(x, y, z, 1.0, 0.0)
            A.rhs(bw, out=result)
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(z.numpy(), reduced @ np.ones(len(free)), rtol=rtol, atol=atol)
        np.testing.assert_allclose(result.numpy(), -(q @ prescribed)[free], rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lanes", [1, 4])
def test_mixed_storage_spd_and_solve(device, lanes):
    rng = np.random.default_rng(72)
    n = 25
    a = (
        np.diag(np.full(n, 3.0))
        + np.diag(np.full(n - 1, -1.0), 1)
        + np.diag(np.full(n - 1, -1.0), -1)
    )
    A = from_scipy(a, device)
    pre = FSAI(A, max_row_size=8, factor_dtype=wp.float32, apply_lanes=lanes)
    g = dense(pre.G).astype(np.float64)
    assert np.all(np.diag(g) > 0)
    np.testing.assert_array_equal(dense(pre.GT), dense(pre.G).T)
    x = wp.array(rng.normal(size=n), dtype=wp.float64, device=device)
    y = wp.array(rng.normal(size=n), dtype=wp.float64, device=device)
    expected = 0.7 * g.T @ g @ x.numpy() - 0.3 * y.numpy()
    pre.matvec(x, y, x, 0.7, -0.3)
    np.testing.assert_allclose(x.numpy(), expected, rtol=1e-12, atol=1e-12)
    b = wp.ones(n, dtype=wp.float64, device=device)
    x.zero_()
    with wp.ScopedDevice(device):
        linear.cg(A, b, x, M=pre, tol=1e-12, maxiter=200)
    np.testing.assert_allclose(a @ x.numpy(), 1.0, atol=1e-11)


def test_squared_validation_and_factor_conversion():
    L = from_scipy(ss.eye(5), "cpu")
    mass = wp.ones(5, dtype=wp.float64, device="cpu")
    for free in [[0, 0], [0, 5], [-1, 2]]:
        with pytest.raises(ValueError, match="free_indices"):
            SquaredLaplacianOperator(L, mass, wp.array(free, dtype=int, device="cpu"))
    mass.zero_()
    with pytest.raises(ValueError, match="mass entries"):
        SquaredLaplacianOperator(L, mass, wp.array([1, 2], dtype=int, device="cpu"))
    with pytest.raises(ValueError, match="positive diagonal"):
        FSAI(from_scipy(ss.eye(3) * 1e100, "cpu"), factor_dtype=wp.float32)

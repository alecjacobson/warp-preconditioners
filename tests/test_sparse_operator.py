import numpy as np
import pytest
import scipy.sparse as ss
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sp
from test_fsai import DEVICES, from_scipy

from warp_preconditioners import FSAI, SparseOperator


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_sparse_operator_rows_aliases_and_graph(device, dtype, lanes):
    # Irregular rows and a partial final thread block; the first row is empty.
    rng = np.random.default_rng(91)
    a = ss.random(67, 67, density=0.12, random_state=rng, format="csr")
    a.data = rng.normal(size=a.nnz)
    a = a.tolil()
    a[0, :] = 0
    a = a.tocsr()
    A = SparseOperator(from_scipy(a, device, dtype), row_lanes=lanes)
    rtol, atol = (2e-5, 2e-6) if dtype == wp.float32 else (1e-12, 1e-12)
    for alias in ["none", "xz", "yz", "xyz"]:
        x = wp.array(rng.normal(size=67), dtype=dtype, device=device)
        y = x if alias == "xyz" else wp.array(rng.normal(size=67), dtype=dtype, device=device)
        z = x if alias in ["xz", "xyz"] else y if alias == "yz" else wp.empty_like(x)
        ref = 0.7 * (a @ x.numpy()) - 0.3 * y.numpy()
        A.matvec(x, y, z, 0.7, -0.3)
        np.testing.assert_allclose(z.numpy(), ref, rtol=rtol, atol=atol)
    x = wp.array(rng.normal(size=67), dtype=dtype, device=device)
    y = wp.empty_like(x)
    y.fill_(float("nan"))
    z = wp.empty_like(x)
    A.matvec(x, y, z, 1.0, 0.0)
    np.testing.assert_allclose(z.numpy(), a @ x.numpy(), rtol=rtol, atol=atol)
    x.fill_(float("nan"))
    y.fill_(2.0)
    A.matvec(x, y, z, 0.0, 0.5)
    np.testing.assert_array_equal(z.numpy(), 1.0)
    if device == "cuda:0":
        x.fill_(1.0)
        with wp.ScopedCapture(device=device) as cap:
            A.matvec(x, y, z, 1.0, 0.0)
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(z.numpy(), a @ np.ones(67), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lanes", [4, 8])
def test_cooperative_fsai_blocks_and_spd_solve(device, lanes):
    n = 69
    a = ss.diags([-np.ones(n - 1), np.full(n, 2.1), -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    raw = sp.bsr_copy(from_scipy(a, device), block_shape=(3, 3))
    operator = SparseOperator(raw, row_lanes=8)
    pre = FSAI(raw, max_row_size=8, apply_lanes=lanes)
    x = wp.zeros(23, dtype=wp.vec3d, device=device)
    b = wp.ones_like(x)
    with wp.ScopedDevice(device):
        linear.cg(operator, b, x, M=pre, tol=1e-12, maxiter=1000)
    np.testing.assert_allclose(a @ x.numpy().ravel(), 1.0, atol=1e-10)
    # Explicit dense preconditioner check includes all alpha/beta alias paths.
    reference = FSAI(raw, max_row_size=8)
    rng = np.random.default_rng(7)
    for alias in ["none", "xz", "yz", "xyz"]:
        u = wp.array(rng.normal(size=(23, 3)), dtype=wp.vec3d, device=device)
        v = (
            u
            if alias == "xyz"
            else wp.array(rng.normal(size=(23, 3)), dtype=wp.vec3d, device=device)
        )
        out = u if alias in ["xz", "xyz"] else v if alias == "yz" else wp.empty_like(u)
        expected = wp.empty_like(u)
        reference.matvec(u, v, expected, 0.7, -0.2)
        pre.matvec(u, v, out, 0.7, -0.2)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)


def test_invalid_row_lanes():
    raw = from_scipy(ss.eye(4), "cpu")
    with pytest.raises(ValueError, match="row_lanes"):
        SparseOperator(raw, row_lanes=3)
    with pytest.raises(ValueError, match="apply_lanes"):
        FSAI(raw, apply_lanes=3)

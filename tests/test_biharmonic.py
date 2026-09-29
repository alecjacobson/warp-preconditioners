import numpy as np
import pytest
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sp
from test_fsai import DEVICES, dense, from_scipy

from warp_preconditioners import BiharmonicSystem


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tau", [0.02, 2.0])
def test_lifted_equivalence(device, tau):
    rng = np.random.default_rng(13)
    k = rng.normal(size=(10, 10))
    k = k @ k.T
    mass = rng.uniform(0.4, 2.0, size=10)
    b_np = rng.normal(size=10)
    system = BiharmonicSystem(
        from_scipy(k, device), wp.array(mass, dtype=wp.float64, device=device), tau
    )
    Q = np.diag(mass) + tau * (k / mass) @ k
    b = system.rhs(wp.array(b_np, dtype=wp.float64, device=device))
    x = wp.zeros_like(b)
    pre = system.preconditioner(max_row_size=10, kap_tolerance=0)
    h_inv = np.linalg.inv(np.diag(mass) + np.sqrt(tau) * k)
    v = wp.array(rng.normal(size=(10, 2)), dtype=wp.vec2d, device=device)
    out = wp.empty_like(v)
    pre.matvec(v, out, out, 1.0, 0.0)
    np.testing.assert_allclose(out.numpy(), h_inv @ v.numpy(), atol=1e-11)
    with wp.ScopedDevice(device):
        linear.gmres(system.matrix, b, x, M=pre, tol=1e-12, maxiter=100, restart=20)
    u = system.solution(x).numpy()
    np.testing.assert_allclose(u, np.linalg.solve(Q, b_np), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(Q @ u, b_np, rtol=1e-9, atol=1e-9)
    # Direct block entries catch sign, ordering and tau scaling mistakes.
    actual = dense(sp.bsr_copy(system.matrix, block_shape=(1, 1)))
    expect = np.zeros((20, 20))
    expect[::2, ::2] = np.diag(mass)
    expect[1::2, 1::2] = np.diag(mass)
    expect[::2, 1::2] = -np.sqrt(tau) * k
    expect[1::2, ::2] = np.sqrt(tau) * k
    np.testing.assert_allclose(actual, expect)

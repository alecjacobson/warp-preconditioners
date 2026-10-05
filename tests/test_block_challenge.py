"""Independent controls for the stronger block benchmarks."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import warp.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from block_challenge import coupled
from block_fsai_candidate import BlockFSAI, BsrBlockFSAI
from elasticity_problem import assemble, cpu_matrix, independent_force
from spd_block_jacobi_candidate import SpdBlockJacobi
from test_fsai import DEVICES, dense, from_scipy


@pytest.mark.parametrize("device", DEVICES)
def test_fiber_energy_force_and_rigid_modes(device):
    v = np.array([[0.1, 0.2, 0.3], [1.2, 0.1, 0.4], [0.2, 1.3, 0.5], [0.3, 0.4, 1.4]])
    t = np.array([[0, 1, 2, 3]], dtype=np.int32)
    q = np.array([[1.0, 2.0, 3.0]]) / np.sqrt(14)
    young, nu, tau = 17.0, 0.31, 1900.0
    fixed = np.zeros(4, dtype=bool)
    a, _, _, _ = assemble(v, t, fixed, young, nu, device=device, fiber=q, reinforcement=tau)
    iso, _, _, _ = assemble(v, t, fixed, young, nu, device=device)
    zero, _, _, _ = assemble(v, t, fixed, young, nu, device=device, fiber=q, reinforcement=0.0)
    k, ki = cpu_matrix(a).toarray(), cpu_matrix(iso).toarray()
    np.testing.assert_array_equal(cpu_matrix(zero).toarray(), ki)
    np.testing.assert_allclose(k, k.T, atol=1e-12)
    eig = np.linalg.eigvalsh(k)
    np.testing.assert_allclose(eig[:6], 0.0, atol=1e-12)
    assert eig[6] > 0
    h = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.1], [0.2, 0.1, 0.4]])
    u = v @ h.T
    volume = abs(np.linalg.det((v[1:] - v[0]).T)) / 6
    expected = volume * tau * (q @ h @ q.T).item() ** 2 / 2
    np.testing.assert_allclose(0.5 * u.ravel() @ (k - ki) @ u.ravel(), expected, rtol=2e-13)
    np.testing.assert_allclose(
        (k @ u.ravel()).reshape(-1, 3), independent_force(v, t, u, young, nu, q, tau), rtol=2e-13
    )
    rigid = np.cross([0.3, -0.1, 0.8], v) + [1.0, 2.0, 3.0]
    np.testing.assert_allclose(k @ rigid.ravel(), 0.0, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_block_jacobi_rotation_control(device):
    # Explicitly test the predicted preconditioned spectrum, independently of b.
    with wp.ScopedDevice(device):
        for direction in ["aligned", "rotated", "rotated2"]:
            a, _, _, _, _ = coupled(2, 1e6, direction)
            k = cpu_matrix(a).toarray()
            pre = SpdBlockJacobi(a)
            p = np.zeros_like(k)
            for i, inv in enumerate(pre.inverse_diagonal.numpy()):
                p[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = inv
            eigen = np.sort(np.linalg.eigvals(p @ k).real)
            expected = np.repeat([0.5, 5 / 6, 5 / 6, 5 / 6, 7 / 6, 7 / 6, 7 / 6, 1.5], 3)
            np.testing.assert_allclose(eigen, expected, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("factory", [BlockFSAI, BsrBlockFSAI])
def test_block_refit_selected_principal_inverse(device, factory):
    rng = np.random.default_rng(992)
    r = rng.normal(size=(12, 12))
    old = r @ r.T + np.eye(12) * 3
    r = rng.normal(size=(12, 12))
    new = r @ r.T + np.eye(12) * 5
    a = sp.bsr_copy(from_scipy(old, device), block_shape=(3, 3))
    b = sp.bsr_copy(from_scipy(new, device), block_shape=(3, 3))
    pre = factory(a, max_blocks=2, factor_dtype=wp.float64, reuse_pattern=True)
    columns = pre.G.columns.numpy().copy()
    pre.update(b)
    np.testing.assert_array_equal(columns, pre.G.columns.numpy())
    g = dense(pre.G)
    for i in range(4):
        row = g[3 * i : 3 * i + 3]
        support = np.flatnonzero(np.any(row != 0, axis=0))
        rhs = np.zeros((len(support), 3))
        rhs[-3:] = np.eye(3)
        z = np.linalg.solve(new[np.ix_(support, support)], rhs)
        expected = np.linalg.solve(np.linalg.cholesky(z[-3:]), z.T)
        np.testing.assert_allclose(row[:, support], expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(row @ new @ row.T, np.eye(3), atol=2e-12)
    x = wp.array(rng.normal(size=(4, 3)), dtype=wp.vec3d, device=device)
    expected = g.T @ g @ x.numpy().ravel()
    pre.matvec(x, x, x, 1.0, 0.0)
    np.testing.assert_allclose(x.numpy().ravel(), expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_spd_block_jacobi_update_failure_and_capture(device):
    rng = np.random.default_rng(46)
    r = rng.normal(size=(6, 6))
    a = r @ r.T + np.eye(6)
    matrix = sp.bsr_copy(from_scipy(a, device), block_shape=(3, 3))
    pre = SpdBlockJacobi(matrix)
    # Exercise aliases, alpha/beta, failed-update rollback and graph replay.
    x = wp.array(rng.normal(size=(2, 3)), dtype=wp.vec3d, device=device)
    initial = x.numpy().copy()
    expected = np.stack(
        [np.linalg.solve(a[3 * i : 3 * i + 3, 3 * i : 3 * i + 3], initial[i]) for i in range(2)]
    )
    pre.matvec(x, x, x, 0.7, -0.2)
    np.testing.assert_allclose(x.numpy(), 0.7 * expected - 0.2 * initial, rtol=1e-12, atol=1e-13)
    saved = pre.inverse_diagonal.numpy().copy()
    bad = sp.bsr_copy(from_scipy(-a, device), block_shape=(3, 3))
    with pytest.raises(ValueError, match="nonpositive"):
        pre.update(bad)
    np.testing.assert_array_equal(saved, pre.inverse_diagonal.numpy())
    pre.update(matrix)
    if device != "cpu":
        x.assign(initial)
        z = wp.empty_like(x)
        with wp.ScopedCapture(device=device) as capture:
            pre.matvec(x, z, z, 1.0, 0.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(z.numpy(), expected, rtol=1e-12, atol=1e-13)

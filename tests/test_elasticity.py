"""Check elasticity assembly against continuum strain energy and rigid motions."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from elasticity_problem import assemble, cpu_matrix, independent_force
from elasticity_stress import recover
from warp_pr1890 import _make_block_jacobi_preconditioner

from warp_preconditioners import BlockJacobi

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_tetrahedron_affine_energy_and_rigid_modes(device):
    v = np.array([[0.1, 0.2, 0.3], [1.2, 0.1, 0.4], [0.2, 1.3, 0.5], [0.3, 0.4, 1.4]])
    t = np.array([[0, 1, 2, 3]], dtype=np.int32)
    young, poisson, density = 17.0, 0.31, 2.0
    a, b, volume, free = assemble(v, t, np.zeros(4, dtype=bool), young, poisson, density, device)
    K = cpu_matrix(a).toarray()
    np.testing.assert_allclose(K, K.T, atol=1e-14)
    eigen = np.linalg.eigvalsh(K)
    np.testing.assert_allclose(eigen[:6], 0, atol=1e-13)
    assert np.all(eigen[6:] > 0)
    H = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.1], [0.2, 0.1, 0.4]])
    u = v @ H.T
    eps = (H + H.T) / 2
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    vol = abs(np.linalg.det((v[1:] - v[0]).T)) / 6
    exact = vol * (mu * np.sum(eps**2) + 0.5 * lame * np.trace(eps) ** 2)
    np.testing.assert_allclose(0.5 * u.ravel() @ K @ u.ravel(), exact, rtol=1e-13)
    rigid = np.cross(np.array([0.3, -0.1, 0.8]), v) + np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(K @ rigid.ravel(), 0, atol=1e-13)
    np.testing.assert_allclose(volume.numpy(), vol / 4, rtol=1e-14)
    np.testing.assert_allclose(b.numpy().sum(axis=0), [0, 0, -9.81 * density * vol], rtol=1e-14)
    np.testing.assert_allclose(
        (K @ u.ravel()).reshape(-1, 3),
        independent_force(v, t, u, young, poisson),
        rtol=1e-13,
        atol=1e-14,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_elimination_and_pr1890_equivalence(device):
    v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    t = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    fixed = np.array([True, True, True, False, False])
    full, _, _, _ = assemble(v, t, np.zeros(5, dtype=bool), device=device)
    a, b, _, free = assemble(v, t, fixed, device=device)
    dofs = (3 * free[:, None] + np.arange(3)).ravel()
    np.testing.assert_allclose(
        cpu_matrix(a).toarray(), cpu_matrix(full).toarray()[np.ix_(dofs, dofs)]
    )
    assert np.linalg.eigvalsh(cpu_matrix(a).toarray()).min() > 0
    x = wp.array([[0.1, -0.3, 0.7], [0.2, 0.4, -0.8]], dtype=wp.vec3d, device=device)
    z = wp.zeros_like(x)
    ours = BlockJacobi(a)
    ours.matvec(x, z, z, 1.0, 0.0)
    expected = z.numpy()
    for strategy in ["auto", "direct", "sequential", "tile"]:
        pre = _make_block_jacobi_preconditioner(a, strategy)
        pre.matvec(x, z, z, 1.0, 0.0)
        np.testing.assert_allclose(z.numpy(), expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("device", DEVICES)
def test_von_mises_affine_and_hydrostatic(device):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([[0, 1, 2, 3]], dtype=np.int32)
    young, poisson = 17.0, 0.31
    H = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.1], [0.2, 0.1, 0.4]])
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    stress = mu * (H + H.T) + lame * np.trace(H) * np.eye(3)
    dev = stress - np.trace(stress) / 3 * np.eye(3)
    expected = np.sqrt(1.5 * np.sum(dev**2))
    nodal, element = recover(v, t, v @ H.T, young, poisson, device)
    np.testing.assert_allclose(nodal, expected, rtol=1e-13)
    np.testing.assert_allclose(element, expected, rtol=1e-13)
    nodal, element = recover(v, t, 0.1 * v, young, poisson, device)
    np.testing.assert_allclose(nodal, 0.0, atol=1e-13)
    np.testing.assert_allclose(element, 0.0, atol=1e-13)

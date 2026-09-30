"""Check imported load mechanics and changing element moduli independently."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from elasticity_problem import assemble, cpu_matrix
from simjeb_problem import distribute, number
from test_fsai import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_pin_resultant_moment_and_work(device):
    points = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    center = points.mean(0)
    ref = np.array([0.1, 0.2, 0.3])
    force = np.array([3.0, -2.0, 8.0])
    out = wp.empty(len(points), dtype=wp.vec3d, device=device)
    wp.launch(
        distribute,
        1,
        [
            wp.array(points, dtype=wp.vec3d, device=device),
            wp.vec3d(center),
            wp.vec3d(ref),
            wp.vec3d(force),
            out,
        ],
        device=device,
    )
    loads = out.numpy()
    np.testing.assert_allclose(loads.sum(0), force, atol=1e-13)
    np.testing.assert_allclose(np.cross(points - ref, loads).sum(0), 0, atol=1e-13)
    translation = np.array([0.4, 0.5, 0.6])
    rotation = np.array([0.3, -0.2, 0.7])
    motion = translation + np.cross(rotation, points - ref)
    np.testing.assert_allclose(np.sum(loads * motion), force @ translation, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_element_moduli_superposition(device):
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
    t = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32)
    fixed = np.array([True, True, True, False, False])
    A = assemble(v, t, fixed, device=device, material_scale=[2.0, 7.0])[0]
    one = assemble(v, t[:1], fixed, young=2e7, device=device)[0]
    two = assemble(v, t[1:], fixed, young=7e7, device=device)[0]
    np.testing.assert_allclose(
        cpu_matrix(A).toarray(),
        (cpu_matrix(one) + cpu_matrix(two)).toarray(),
        rtol=1e-14,
        atol=1e-8,
    )


def test_nastran_exponents():
    assert number("5.222-13") == 5.222e-13
    assert number("-3.14+7") == -3.14e7
    assert number(".342") == 0.342

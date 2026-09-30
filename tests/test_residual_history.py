"""The diagnostic sampler must leave native CG/CR trajectories unchanged."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from test_fsai import DEVICES
from warp.optim import linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from residual_history import sample_iterations


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solver", [linear.cg, linear.cr])
def test_sampling_preserves_recurrence(solver, device):
    with wp.ScopedDevice(device):
        a = wp.array(np.diag(np.geomspace(1.0, 1e4, 30)), dtype=wp.float64)
        b = wp.array(np.random.default_rng(3).normal(size=30), dtype=wp.float64)
        native, sampled = wp.zeros_like(b), wp.zeros_like(b)
        solver(a, b, native, tol=0.0, atol=0.0, maxiter=37, check_every=0, use_cuda_graph=True)
        observations = []
        with sample_iterations([1, 2, 7, 19, 37], lambda i, r: observations.append((i, r))):
            nit, _, _ = solver(
                a, b, sampled, tol=0.0, atol=0.0, maxiter=37, check_every=0, use_cuda_graph=True
            )
        assert nit == 37
        assert [i for i, _ in observations] == [0, 1, 2, 7, 19, 37]
        np.testing.assert_array_equal(native.numpy(), sampled.numpy())

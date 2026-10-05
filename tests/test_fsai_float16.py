"""Native half-precision FSAI; no matrix/vector promotion wrapper."""

import numpy as np
import pytest
import warp as wp
import warp.sparse as sp
from test_fsai import DEVICES, from_scipy

from warp_preconditioners import FSAI


def dense_half(matrix):
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    out = np.zeros(matrix.shape, dtype=np.float64)
    for i in range(matrix.nrow):
        out[i, columns[offsets[i] : offsets[i + 1]]] = values[offsets[i] : offsets[i + 1]]
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lanes", [1, 4])
def test_half_factor_application_and_update(device, lanes):
    rng = np.random.default_rng(19)
    r = rng.normal(size=(9, 9))
    a = (r @ r.T + 4 * np.eye(9)).astype(np.float16).astype(np.float64)
    raw = sp.bsr_copy(from_scipy(a, device, wp.float16), block_shape=(3, 3))
    pre = FSAI(raw, max_row_size=9, kap_tolerance=0, reuse_pattern=True, apply_lanes=lanes)
    assert pre.G.scalar_type == wp.float16
    assert pre.GT.scalar_type == wp.float16
    g = dense_half(pre.G)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=1e-2, atol=1e-3)
    x = wp.array(rng.normal(size=(3, 3)), dtype=wp.vec3h, device=device)
    expected = 0.7 * g.T @ g @ x.numpy().ravel() - 0.2 * x.numpy().ravel()
    pre.matvec(x, x, x, 0.7, -0.2)
    np.testing.assert_allclose(x.numpy().ravel(), expected, rtol=5e-3, atol=1e-3)
    pre.update(sp.bsr_copy(from_scipy(2 * a, device, wp.float16), block_shape=(3, 3)))
    g = dense_half(pre.G)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(2 * a), rtol=1e-2, atol=1e-3)
    invalid = sp.bsr_copy(raw)
    invalid.values.zero_()
    with pytest.raises(ValueError):
        pre.update(invalid)
    np.testing.assert_array_equal(dense_half(pre.G), g)
    np.testing.assert_array_equal(dense_half(pre.GT), g.T)


@pytest.mark.parametrize("device", DEVICES)
def test_half_scaled_and_rank_deficient_blocks(device):
    raw = sp.bsr_copy(from_scipy(np.diag([1, 0.01]), device, wp.float16), block_shape=(2, 2))
    pre = FSAI(raw)
    x = wp.ones(2, dtype=wp.float16, device=device)
    y = wp.zeros_like(x)
    pre.matvec(x, y, y, 1, 0)
    np.testing.assert_allclose(y.numpy(), [1, 100], rtol=0, atol=0.01)
    for a in [np.array([[1, 2], [2, 4]]), np.array([[9, 21], [21, 49]])]:
        pre = FSAI(from_scipy(a, device, wp.float16), max_row_size=2)
        assert pre.truncated_rows == 1
        g = dense_half(pre.G)
        np.testing.assert_allclose(g, np.diag(1 / np.sqrt(np.diag(a))), rtol=2e-3)

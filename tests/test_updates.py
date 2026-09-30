import numpy as np
import pytest
import warp as wp
import warp.sparse as sp
from test_fsai import DEVICES, dense, from_scipy

from warp_preconditioners import FSAI, BlockJacobi


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_refit_local_solves_and_rollback(device, block, dtype):
    rng = np.random.default_rng(48)
    r = rng.normal(size=(12, 12))
    a = r @ r.T + np.eye(12) * 3
    A = sp.bsr_copy(from_scipy(a, device, dtype), block_shape=(block, block))
    m = FSAI(A, max_row_size=5, reuse_pattern=True, factor_dtype=wp.float32)
    gptr, tptr = m.G.values.ptr, m.GT.values.ptr
    b = a + np.diag(np.arange(12))
    B = sp.bsr_copy(from_scipy(b, device, dtype), block_shape=(block, block))
    m.update(B)
    assert (m.G.values.ptr, m.GT.values.ptr) == (gptr, tptr)
    g = dense(m.G)
    np.testing.assert_array_equal(dense(m.GT), g.T)
    off, col = m.G.offsets.numpy(), m.G.columns.numpy()
    for i in range(12):
        support = col[off[i] : off[i + 1]]
        e = np.zeros(len(support))
        e[-1] = 1
        z = np.linalg.solve(b[np.ix_(support, support)], e)
        np.testing.assert_allclose(g[i, support], z / np.sqrt(z[-1]), rtol=2e-4, atol=2e-7)
    invalid = sp.bsr_copy(B)
    invalid.values.zero_()
    with pytest.raises(ValueError, match="positive diagonals"):
        m.update(invalid)
    np.testing.assert_array_equal(dense(m.G), g)
    changed = sp.bsr_copy(B)
    columns = changed.columns.numpy()
    columns[0] += 1
    changed.columns.assign(columns)
    with pytest.raises(ValueError, match="sparsity"):
        m.update(changed)
    np.testing.assert_array_equal(dense(m.G), g)
    # Same object, changed numerical values, then no-argument update.
    B.values.assign(A.values.numpy())
    m.update()
    fresh = FSAI(A, max_row_size=12, kap_tolerance=0, reuse_pattern=True)
    fresh.update(B)
    np.testing.assert_allclose(
        dense(fresh.G).T @ dense(fresh.G), np.linalg.inv(a), rtol=2e-4, atol=1e-7
    )
    assert fresh.quality(probes=2) < (1e-5 if dtype == wp.float32 else 1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_block_update_and_capture(device):
    a = np.array([[3.0, 0.4, 0], [0.4, 2, 0], [0, 0, 4.0]])
    A = sp.bsr_copy(from_scipy(a, device), block_shape=(3, 3))
    B = sp.bsr_copy(from_scipy(2 * a, device), block_shape=(3, 3))
    for m in [BlockJacobi(A), FSAI(A, max_row_size=3, kap_tolerance=0, reuse_pattern=True)]:
        x = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3d, device=device)
        y = wp.zeros_like(x)
        m.matvec(x, y, y, 1.0, 0.0)
        if device != "cpu":
            with wp.ScopedCapture(device=device) as cap:
                m.matvec(x, y, y, 1.0, 0.0)
        m.update(B)
        if device != "cpu":
            wp.capture_launch(cap.graph)
        else:
            m.matvec(x, y, y, 1.0, 0.0)
        np.testing.assert_allclose(
            y.numpy().ravel(), np.linalg.solve(2 * a, x.numpy().ravel()), rtol=1e-12
        )
        B.values.zero_()
        with pytest.raises(ValueError):
            m.update(B)
        m.matvec(x, y, y, 1.0, 0.0)
        np.testing.assert_allclose(
            y.numpy().ravel(), np.linalg.solve(2 * a, x.numpy().ravel()), rtol=1e-12
        )
        B.values.assign(2 * A.values.numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_numeric_zero_becomes_nonzero_inside_block(device):
    # BSR stores whole blocks: zero scalar entries can become nonzero without
    # changing block topology. The cached gather map must retain those entries.
    a = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    b = a + 0.2 * np.ones((3, 3))
    A = sp.bsr_copy(from_scipy(a, device), block_shape=(3, 3))
    B = sp.bsr_copy(from_scipy(b, device), block_shape=(3, 3))
    m = FSAI(A, max_row_size=3, kap_tolerance=0, reuse_pattern=True)
    m.update(A)
    np.testing.assert_allclose(dense(m.G).T @ dense(m.G), np.linalg.inv(a), atol=1e-13)
    m.update(B)
    np.testing.assert_allclose(dense(m.G).T @ dense(m.G), np.linalg.inv(b), atol=1e-13)


def test_refit_requires_explicit_plan():
    a = from_scipy(np.eye(3), "cpu")
    with pytest.raises(ValueError, match="reuse_pattern=True"):
        FSAI(a).update(a)

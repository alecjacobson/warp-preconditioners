"""Independent numerical checks for experimental block FSAI."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import warp.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from block_fsai_candidate import BlockFSAI, BsrBlockFSAI
from test_fsai import DEVICES, dense, from_scipy


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bs", [2, 3])
@pytest.mark.parametrize("blocks", [1, 2, 4])
@pytest.mark.parametrize("factory", [BlockFSAI, BsrBlockFSAI])
def test_block_principal_inverse(device, bs, blocks, factory):
    rng = np.random.default_rng(712)
    n = 4 * bs
    r = rng.normal(size=(n, n))
    a = r @ r.T + np.eye(n)
    s = np.geomspace(0.01, 100, n)
    a *= s[:, None] * s[None, :]
    A = sp.bsr_copy(from_scipy(a, device), block_shape=(bs, bs))
    pre = factory(A, max_blocks=blocks, factor_dtype=wp.float64)
    g = dense(pre.G)
    assert np.linalg.eigvalsh(g.T @ g).min() > 0
    for i in range(4):
        rows = slice(bs * i, bs * (i + 1))
        row = g[rows]
        support = np.flatnonzero(np.any(row != 0, axis=0))
        # Independent dense solve for the selected support and identity RHS.
        rhs = np.zeros((len(support), bs))
        rhs[-bs:, :] = np.eye(bs)
        z = np.linalg.solve(a[np.ix_(support, support)], rhs)
        expected = np.linalg.solve(np.linalg.cholesky(z[-bs:, :]), z.T)
        np.testing.assert_allclose(row[:, support], expected, rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(row @ a @ row.T, np.eye(bs), atol=2e-12)
    if blocks == 4:
        np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=2e-11, atol=1e-10)
    if blocks == 1:
        for i in range(4):
            rows = slice(bs * i, bs * (i + 1))
            np.testing.assert_allclose(
                (g.T @ g)[rows, rows], np.linalg.inv(a[rows, rows]), rtol=2e-11
            )
    x = wp.array(rng.normal(size=(4, bs)), dtype=wp.types.vector(bs, wp.float64), device=device)
    expected = 0.7 * g.T @ g @ x.numpy().ravel() - 0.2 * x.numpy().ravel()
    pre.matvec(x, x, x, 0.7, -0.2)
    np.testing.assert_allclose(x.numpy().ravel(), expected, rtol=2e-11, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("storage", [wp.float32, wp.float64])
def test_fused_product_aliases_and_capture(device, storage):
    from fused_fsai_candidate import FusedFSAI

    rng = np.random.default_rng(19)
    r = rng.normal(size=(15, 15))
    a = r @ r.T + np.eye(15)
    pre = FusedFSAI(from_scipy(a, device), max_row_size=4, factor_dtype=storage)
    g = dense(pre.G).astype(np.float64)
    for alias in ["none", "xz", "yz", "xyz"]:
        x = wp.array(rng.normal(size=15), dtype=wp.float64, device=device)
        y = x if alias == "xyz" else wp.array(rng.normal(size=15), dtype=wp.float64, device=device)
        z = x if alias in ["xz", "xyz"] else y if alias == "yz" else wp.empty_like(x)
        expected = 0.7 * g.T @ g @ x.numpy() - 0.2 * y.numpy()
        pre.matvec(x, y, z, 0.7, -0.2)
        np.testing.assert_allclose(z.numpy(), expected, rtol=1e-12, atol=1e-12)
    if device != "cpu":
        x = wp.array(rng.normal(size=15), dtype=wp.float64, device=device)
        y = wp.zeros_like(x)
        with wp.ScopedCapture(device=device) as cap:
            pre.matvec(x, y, y, 1.0, 0.0)
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(y.numpy(), g.T @ g @ x.numpy(), rtol=1e-12, atol=1e-12)

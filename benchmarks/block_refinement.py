"""Nested P1 cantilever refinement check for the anisotropic material experiment.

This is a discretization diagnostic, not a solver performance comparison. Constant
body load, fixed Poisson ratio and smooth constant fiber direction on every mesh.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp
from block_challenge import native, rotation
from elasticity_problem import assemble, independent_force
from warp.optim import linear

from warp_preconditioners import FSAI, BlockJacobi


def beam_mesh(m):
    shape = (4 * m + 1, m + 1, m + 1)
    v = np.indices(shape).reshape(3, -1).T.astype(float) / m
    base = np.indices((4 * m, m, m)).reshape(3, -1).T
    cells = []
    for order in itertools.permutations(range(3)):
        corners = [np.zeros(3, dtype=int)]
        for axis in order:
            corner = corners[-1].copy()
            corner[axis] += 1
            corners.append(corner)
        idx = base[:, None, :] + np.array(corners)[None, :, :]
        cells.append(np.ravel_multi_index(idx.transpose(2, 0, 1), shape))
    return v, np.concatenate(cells).astype(np.int32), v[:, 0] == 0


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--contrasts", type=float, nargs="+", default=[1, 1000])
    p.add_argument("--output", type=Path, default=Path("results/block-refinement.json"))
    args = p.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    output = dict(
        description="Nested Freudenthal tetrahedra, 4x1x1 cantilever, x=0 fixed, constant gravity, nu=.342; no timings claimed",
        cases=[],
    )
    young, nu = 1e7, 0.342
    mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
    for contrast in args.contrasts:
        for level in args.levels:
            v, t, fixed = beam_mesh(level)
            q = np.tile(rotation("rotated")[:, 2], (len(t), 1))
            tau = (contrast - 1) * (lam + 2 * mu)
            a, b, volume, free = assemble(v, t, fixed, young, nu, 1.0, fiber=q, reinforcement=tau)
            pre = FSAI(a, max_row_size=12, kap_tolerance=0.001, factor_dtype=wp.float64)
            x, nit, _ = native(a, b, pre, limit=100000, tol=1e-12)
            residual = wp.empty_like(b)
            linear.aslinearoperator(a).matvec(x, b, residual, -1.0, 1.0)
            delta, cnit, _ = native(a, residual, BlockJacobi(a), limit=100000, tol=1e-10)
            u = x.numpy() + delta.numpy()
            full = np.zeros_like(v)
            full[free] = u
            force = independent_force(v, t, full, young, nu, q, tau)[free]
            rel = np.linalg.norm(force - b.numpy()) / np.linalg.norm(b.numpy())
            correction = np.linalg.norm(delta.numpy()) / np.linalg.norm(u)
            assert rel < 1e-7 and correction < 1e-7
            row = dict(
                contrast=contrast,
                level=level,
                tets=len(t),
                dofs=a.shape[0],
                compliance=float(np.sum(u * b.numpy())),
                iterations=nit,
                correction_iterations=cnit,
                element_relative_residual=float(rel),
                correction_relative_displacement=float(correction),
                maximum_displacement=float(np.linalg.norm(u, axis=1).max()),
            )
            output["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()

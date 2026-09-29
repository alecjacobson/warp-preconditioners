"""Grid biharmonic ablation; SciPy constructs inputs and checks residuals only."""

import argparse
import json
import time

import numpy as np
import scipy.sparse as ss
import warp as wp
import warp.optim.linear as linear
from dragon import berr, norm, upload

from warp_preconditioners import FSAI


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--output")
    args = parser.parse_args()
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    n = args.size
    d = ss.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    k = ss.kronsum(d, d).tocsr()
    mass = 1 / (n + 1) ** 2
    q = mass * ss.eye(n * n) + k @ k / mass
    exact = np.random.default_rng(13).normal(size=n * n)
    b_np = q @ exact
    a = upload(q, "cuda:0")
    b = wp.array(b_np, dtype=wp.float64, device="cuda:0")
    records = []
    for width in args.widths:

        def make(width=width):
            return linear.preconditioner(a, "diag") if width == 1 else FSAI(a, max_row_size=width)

        pre = make()
        x = wp.zeros_like(b)
        linear.cg(a, b, x, M=pre, tol=1e-11, maxiter=10, check_every=10)
        wp.synchronize()
        start = time.perf_counter()
        pre = make()
        wp.synchronize()
        setup = time.perf_counter() - start
        x.zero_()
        wp.synchronize()
        start = time.perf_counter()
        nit, res, atol = linear.cg(a, b, x, M=pre, tol=1e-11, maxiter=20000, check_every=25)
        wp.synchronize()
        solve = time.perf_counter() - start
        solution = x.numpy()
        record = dict(
            size=n,
            width=width,
            setup_s=setup,
            solve_s=solve,
            iterations=int(nit),
            berr=berr(q, abs(q), b_np, solution),
            relative_residual=float(norm(b_np - q @ solution) / norm(b_np)),
            forward_error=float(norm(solution - exact) / norm(exact)),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

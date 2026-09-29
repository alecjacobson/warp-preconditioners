"""Separate the changed equation, stopping rule, and RHS in the timing jump."""

import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import warp as wp
from dragon import berr, norm, upload
from sksparse.cholmod import cholesky
from tune_dirichlet import load_problem
from warp.optim import linear

from warp_preconditioners import FSAI


def main():
    a0, b0, m0, _, _, _ = load_problem()
    data = np.load("data/dirichlet/fields.npz")
    original = sio.mmread("/tmp/dump/k2_Q.mtx").tocsr()
    old_rhs = np.asarray(sio.mmread("/tmp/dump/k2_rhs.mtx"))[:, 2]
    cases = [("original_height_alpha1", original, old_rhs, data["mass"])]
    cases += [
        (f"fixed_patches_alpha{alpha:g}", a0 + ss.diags(alpha * m0), b0, m0)
        for alpha in [1.0, 1e-4, 0.0]
    ]
    results = []
    for name, a, b_np, mass in cases:
        # Verification only, outside GPU timings. No shift inside Cholesky.
        reference = cholesky(a.tocsc(), ordering_method="amd")(b_np)
        denom = float(np.sum(mass * reference**2))
        A = upload(a, "cuda:0")
        b = wp.array(b_np, dtype=wp.float64, device="cuda:0")
        x = wp.zeros_like(b)
        pre = FSAI(A)
        linear.cg(A, b, x, M=pre, tol=0.0, atol=0.0, maxiter=10, check_every=0)
        wp.synchronize()
        for budget, tol in [(500, 0.0), (200000, 1e-12)]:
            x.zero_()
            wp.synchronize()
            start = time.perf_counter()
            nit, res, _ = linear.cg(
                A, b, x, M=pre, tol=tol, atol=0.0, maxiter=budget, check_every=0
            )
            wp.synchronize()
            elapsed = time.perf_counter() - start
            u = x.numpy()
            result = dict(
                case=name,
                budget=budget,
                tol=tol,
                iterations=int(nit.numpy()[0]),
                solve_s=elapsed,
                recursive_relative_residual=float(np.sqrt(res.numpy()[0])) / norm(b_np),
                relative_residual=norm(b_np - a @ u) / norm(b_np),
                backward_error=berr(a, abs(a), b_np, u),
                mass_relative_error=float(np.sqrt(np.sum(mass * (u - reference) ** 2) / denom)),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            Path("data/tuning/regularization.json").write_text(
                json.dumps(dict(results=results), indent=2) + "\n"
            )


if __name__ == "__main__":
    main()

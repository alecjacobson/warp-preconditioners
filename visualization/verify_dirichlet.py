"""Independently check the Dirichlet fields using Cholesky with AMD ordering.

Requires scikit-sparse / SuiteSparse, only for this optional verification.
Run after solve_dirichlet.py has finished.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sksparse.cholmod import cholesky


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dir", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--data", type=Path, default=Path("data/dirichlet"))
    parser.add_argument("--refined-reference", type=Path)
    args = parser.parse_args()
    data = np.load(args.data / "fields.npz")
    meta = json.loads((args.data / "solve.json").read_text())
    if meta.get("tuned") and args.refined_reference is None:
        parser.error(
            "The tuned operator requires --refined-reference; run benchmarks/refine_dirichlet.py first"
        )
    n = len(data["vertices"])
    free, fixed, initial = data["free"], data["fixed"], data["constraints"]
    mixed = sio.mmread(args.dir / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n]
    L = -mixed[:n, n:]
    del mixed
    Q = L @ ss.diags(1 / mass) @ L
    A = Q[free][:, free].tocsc()
    rhs = -(Q @ initial)[free]
    start = time.perf_counter()
    factor = cholesky(A, ordering_method="amd")
    reference = initial.copy()
    reference[free] = factor(rhs)
    elapsed = time.perf_counter() - start
    assert factor.D().min() > 0
    refined = None
    if args.refined_reference:
        refined = np.load(args.refined_reference)
        assert refined.shape == initial.shape and np.isfinite(refined).all()
        np.testing.assert_array_equal(refined[fixed], initial[fixed])
    checks = {}
    for name in meta["display_order"]:
        u = data[name]
        np.testing.assert_array_equal(u[fixed], initial[fixed])
        reduced_residual = A @ u[free] - rhs
        # This also detects accidentally squaring the restricted Laplacian.
        # Dot products on nearly cancelling rows need an absolute roundoff
        # bound based on |Q| |u|, not a relative tolerance on the small residual.
        row_terms = np.diff(Q.indptr)[free]
        eps_terms = row_terms * np.finfo(float).eps
        gamma = eps_terms / (1 - eps_terms)
        roundoff_bound = 4 * gamma * (abs(Q) @ abs(u))[free] + np.finfo(float).tiny
        elimination_error = np.abs((Q @ u)[free] - reduced_residual)
        assert np.all(elimination_error <= roundoff_bound)

        energy = 0.5 * np.sum((L @ u) ** 2 / mass)
        np.testing.assert_allclose(energy, meta["results"][name]["bending_energy"], rtol=1e-12)
        checks[name] = dict(
            elimination_error_over_roundoff_bound=float(np.max(elimination_error / roundoff_bound)),
            mass_relative_error_to_cholesky=float(
                np.sqrt(np.sum(mass * (u - reference) ** 2) / np.sum(mass * reference**2))
            ),
            max_abs_error_to_cholesky=float(np.max(np.abs(u - reference))),
            energy_relative_difference=float(
                energy / (0.5 * np.sum((L @ reference) ** 2 / mass)) - 1
            ),
        )
        if refined is not None:
            checks[name]["mass_relative_error_to_refined"] = float(
                np.sqrt(np.sum(mass * (u - refined) ** 2) / np.sum(mass * refined**2))
            )
        if meta.get("tuned"):
            factored_rhs = -(L.T @ ((L @ initial) / mass))[free]
            stationarity = (L.T @ ((L @ u) / mass))[free]
            checks[name]["factored_relative_residual"] = float(
                np.linalg.norm(stationarity) / np.linalg.norm(factored_rhs)
            )
            np.testing.assert_allclose(
                checks[name]["factored_relative_residual"],
                meta["results"][name]["relative_residual"],
                rtol=1e-10,
            )
    if meta["display_order"]:
        fsai = meta["display_order"][0]
        if refined is not None:
            assert checks[fsai]["mass_relative_error_to_refined"] < 1e-6, checks[fsai]
        else:
            assert checks[fsai]["mass_relative_error_to_cholesky"] < 1e-4, checks[fsai]
    audit = dict(
        method="CHOLMOD Cholesky with AMD ordering; no diagonal shift",
        factor_and_solve_s=elapsed,
        minimum_D=float(factor.D().min()),
        range=[float(reference.min()), float(reference.max())],
        energy=float(0.5 * np.sum((L @ reference) ** 2 / mass)),
        relative_residual=float(np.linalg.norm(rhs - A @ reference[free]) / np.linalg.norm(rhs)),
        comparisons=checks,
        refined_reference=(
            dict(
                path=str(args.refined_reference),
                sha256=hashlib.sha256(refined.tobytes()).hexdigest(),
                method="long-double factored energy gradient with AMD Cholesky correction solves",
            )
            if refined is not None
            else None
        ),
    )
    np.save(args.data / "cholesky.npy", reference)
    (args.data / "cholesky.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Separate audit file: never overwrite solver fields or solver metadata.
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

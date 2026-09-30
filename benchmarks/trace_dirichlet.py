"""Collect actual residual histories for the tuned dragon Dirichlet comparison."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from dirichlet_inputs import load_laplacian
from dragon import norm, upload
from residual_history import sample_iterations
from warp.optim import linear

from warp_preconditioners import FSAI, SquaredLaplacianOperator


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dirichlet-current"))
    parser.add_argument("--output", type=Path, default=Path("results/dirichlet-residuals.json"))
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument(
        "--methods", nargs="+", choices=["fsai_cg", "fsai_cr", "jacobi_cg", "jacobi_cr"]
    )
    parser.add_argument(
        "--resume", action="store_true", help="Keep completed matching trajectories"
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("samples must be at least two")
    data = np.load(args.data / "fields.npz")
    reference_meta = json.loads((args.data / "solve.json").read_text())
    k = int(reference_meta["convergence"]["k"])
    methods = args.methods or [
        reference_meta.get("selected_fsai", "fsai_cg"),
        "jacobi_cg",
        "jacobi_cr",
    ]
    L = load_laplacian(args.data)
    mass, free, prescribed = data["mass"], data["free"], data["constraints"]
    Q = L @ sp.diags(1 / mass) @ L
    A = Q[free][:, free].tocsr()
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    matrix, laplacian = upload(A, "cuda:0"), upload(L, "cuda:0")
    operator = SquaredLaplacianOperator(
        laplacian,
        wp.array(mass, dtype=wp.float64),
        wp.array(free.astype(np.int32), dtype=wp.int32),
        row_lanes=4,
    )
    b = operator.rhs(wp.array(prescribed, dtype=wp.float64))
    bnorm = norm(b.numpy())
    checkpoints = np.unique(
        np.r_[np.rint(np.geomspace(1, 100 * k, args.samples)).astype(int), k, 10 * k, 100 * k]
    )
    records = dict(
        problem="Dirichlet biharmonic interpolation, head=+1 and tail=-1; no data term",
        warp=wp.__version__,
        gpu=wp.get_device().name,
        k=k,
        methods=methods,
        field_target=reference_meta["convergence"].get("field_target"),
        b_norm=bnorm,
        sampling="logarithmic checkpoints plus k, 10k, 100k; same live Krylov state, no restarts",
        residual="norm((L.T @ ((L @ full_u) / mass))[free]) / norm(b), independently recomputed CPU float64",
        recursive_residual="Warp internal recursively updated Euclidean residual norm / norm(b)",
        initial_guess="zero at free vertices, exact prescribed values at fixed vertices",
        operator="SquaredLaplacianOperator, float64, row_lanes=4; identical for all methods",
        input_sha256=hashlib.sha256(b.numpy().tobytes() + free.tobytes()).hexdigest(),
        results={},
    )
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        previous["operator"] = previous["operator"].replace("both methods", "all methods")
        for key in (
            "input_sha256",
            "k",
            "warp",
            "gpu",
            "operator",
            "initial_guess",
            "field_target",
        ):
            if previous[key] != records[key]:
                raise ValueError(f"Cannot resume: {key} differs")
        records["results"] = {
            name: row for name, row in previous["results"].items() if name in methods
        }
        for name, row in records["results"].items():
            expected = k if name.startswith("fsai_") else 100 * k
            if row["actual_iterations"] != expected or row["tol"] != 0.0:
                raise ValueError(f"Cannot resume: {name} stopping rule differs")
    refined_path = args.data / "refined_reference.npy"
    refined = np.load(refined_path) if refined_path.exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in methods:
        solver = getattr(linear, name.rsplit("_", 1)[1])
        budget = k if name.startswith("fsai_") else 100 * k
        tol = 0.0
        if name in records["results"]:
            print(name, "already recorded", flush=True)
            continue
        pre = (
            FSAI(
                matrix, max_row_size=48, kap_tolerance=0.003, apply_lanes=4, factor_dtype=wp.float32
            )
            if name.startswith("fsai_")
            else linear.preconditioner(matrix, "diag")
        )
        x = wp.zeros_like(b)
        solver(operator, b, x, M=pre, tol=0.0, atol=0.0, maxiter=10, check_every=0)
        x.zero_()
        wp.synchronize()
        samples = []
        start = time.perf_counter()
        last_report = start
        full = prescribed.copy()

        def observe(iteration, recursive):
            nonlocal last_report
            full[free] = x.numpy()
            grad = (L.T @ ((L @ full) / mass))[free]
            residual = norm(grad) / bnorm
            assert np.isfinite(residual) and residual >= 0
            samples.append(
                dict(
                    iteration=iteration,
                    relative_residual=residual,
                    recursive_relative_residual=recursive / bnorm,
                )
            )
            if refined is not None:
                samples[-1]["mass_relative_error_to_refined"] = float(
                    np.sqrt(np.sum(mass * (full - refined) ** 2) / np.sum(mass * refined**2))
                )
            now = time.perf_counter()
            if now - last_report > 25 or iteration in (0, k, 10 * k, 100 * k):
                print(
                    name,
                    iteration,
                    "recomputed",
                    residual,
                    "recursive",
                    recursive / bnorm,
                    flush=True,
                )
                last_report = now

        with sample_iterations(checkpoints, observe):
            nit, _, _ = solver(
                operator, b, x, M=pre, tol=tol, atol=0.0, maxiter=budget, check_every=0
            )
        field = x.numpy()
        reference_keys = [
            key
            for key in reference_meta["display_order"]
            if key.startswith(name + "_")
            and reference_meta["results"][key]["actual_iterations"] == nit
        ]
        discrepancy = None
        if reference_keys:
            reference = data[reference_keys[0]][free]
            discrepancy = norm(field - reference) / max(norm(reference), np.finfo(float).tiny)
            assert discrepancy < 1e-7, (name, discrepancy)
        records["results"][name] = dict(
            actual_iterations=nit,
            tol=tol,
            samples=samples,
            preconditioner=(
                "FSAI width=48, kap=0.003, float32 storage/float64 arithmetic, apply_lanes=4"
                if name.startswith("fsai_")
                else "Jacobi"
            ),
            relative_l2_difference_from_figure_field=discrepancy,
            instrumented_wall_seconds=time.perf_counter() - start,
            timing_note="Includes transfers, graph captures and CPU verification; not benchmark solve time",
        )
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(name, "DONE", nit, "field difference", discrepancy, flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

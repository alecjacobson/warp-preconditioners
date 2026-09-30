"""Select one FSAI solver at matched field accuracy, then refresh dragon fields."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from dirichlet_inputs import load_laplacian
from dragon import upload
from residual_history import sample_iterations
from warp.optim import linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualization"))
from solve_dirichlet import run_tuned

from warp_preconditioners import FSAI, SquaredLaplacianOperator


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, default=Path("data/dirichlet-tuned"))
    p.add_argument("--output", type=Path, default=Path("data/dirichlet-current"))
    p.add_argument("--report", type=Path, default=Path("results/dirichlet-refresh.json"))
    p.add_argument("--stage", choices=["select", "fields"], required=True)
    p.add_argument("--target", type=float, default=1e-8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = np.load(args.source / "fields.npz")
    old = json.loads((args.source / "solve.json").read_text())
    L = load_laplacian(args.source)
    mass, free, initial = d["mass"], d["free"], d["constraints"]
    Q = L @ sp.diags(1 / mass) @ L
    A = Q[free][:, free].tocsr()
    reference = np.load(args.source / "refined_reference.npy")
    reference_audit = json.loads((args.source / "cholesky.json").read_text())
    assert (
        hashlib.sha256(reference.tobytes()).hexdigest()
        == reference_audit["refined_reference"]["sha256"]
    )
    denominator = float(np.sum(mass * reference**2))

    def error(u):
        return float(np.sqrt(np.sum(mass * (u - reference) ** 2) / denominator))

    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    if args.stage == "select":
        state = dict(
            code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            utc=datetime.now(timezone.utc).isoformat(),
            gpu=wp.get_device().name,
            warp=wp.__version__,
            target=args.target,
            target_definition="Relative lumped-mass field error against the independently refined reference <= target",
            reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            selection="Three interleaved warm repeats; smallest median operator/preconditioner setup + solve at verified matched field accuracy",
            stopping="Sample field error every 500 iterations, refine first qualifying interval every 25; validate uninstrumented endpoint",
            candidates={},
        )
        matrix, laplacian = upload(A, "cuda:0"), upload(L, "cuda:0")
        operator = SquaredLaplacianOperator(
            laplacian,
            wp.array(mass, dtype=wp.float64),
            wp.array(free.astype(np.int32), dtype=wp.int32),
            row_lanes=4,
        )
        b = operator.rhs(wp.array(initial, dtype=wp.float64))
        pre = FSAI(
            matrix, max_row_size=48, kap_tolerance=0.003, apply_lanes=4, factor_dtype=wp.float32
        )
        for method in ["fsai_cg", "fsai_cr"]:
            solver = getattr(linear, method.rsplit("_", 1)[1])
            x = wp.zeros_like(b)
            samples = []

            def observe(iteration, recursive):
                full = initial.copy()
                full[free] = x.numpy()
                samples.append(dict(iteration=iteration, field_error=error(full)))

            with sample_iterations(np.arange(500, 100001, 500), observe):
                solver(operator, b, x, M=pre, tol=0.0, atol=0.0, maxiter=100000, check_every=0)
            hit = next(s for s in samples if s["field_error"] <= args.target)
            start = max(1, hit["iteration"] - 500)
            finer = []
            x.zero_()

            def observe_fine(iteration, recursive):
                full = initial.copy()
                full[free] = x.numpy()
                finer.append(dict(iteration=iteration, field_error=error(full)))

            with sample_iterations(np.arange(start, hit["iteration"] + 1, 25), observe_fine):
                solver(
                    operator,
                    b,
                    x,
                    M=pre,
                    tol=0.0,
                    atol=0.0,
                    maxiter=hit["iteration"],
                    check_every=0,
                )
            stop = next(s["iteration"] for s in finer if s["field_error"] <= args.target)
            sol, row = run_tuned(A, L, mass, free, initial, method, stop)
            full = initial.copy()
            full[free] = sol
            assert error(full) <= args.target
            state["candidates"][method] = dict(
                iterations=stop, coarse_samples=samples, fine_samples=finer, repeats=[]
            )
            print("selected stop", method, stop, error(full), flush=True)
        for repeat in range(3):
            for method in ["fsai_cg", "fsai_cr"] if repeat % 2 == 0 else ["fsai_cr", "fsai_cg"]:
                row = state["candidates"][method]
                sol, result = run_tuned(A, L, mass, free, initial, method, row["iterations"])
                full = initial.copy()
                full[free] = sol
                result["mass_relative_error_to_refined"] = error(full)
                assert result["mass_relative_error_to_refined"] <= args.target
                result["total_s"] = result["setup_s"] + result["solve_s"]
                result["repeat"] = repeat
                row["repeats"].append(result)
                np.save(args.output / f"{method}-candidate.npy", full)
                print(
                    "repeat",
                    method,
                    repeat,
                    result["total_s"],
                    result["mass_relative_error_to_refined"],
                    flush=True,
                )
        for row in state["candidates"].values():
            row["median_total_s"] = float(np.median([r["total_s"] for r in row["repeats"]]))
            row["median_setup_s"] = float(np.median([r["setup_s"] for r in row["repeats"]]))
            row["median_solve_s"] = float(np.median([r["solve_s"] for r in row["repeats"]]))
        state["winner"] = min(
            state["candidates"], key=lambda key: state["candidates"][key]["median_total_s"]
        )
        args.report.write_text(json.dumps(state, indent=2) + "\n")
        print("WINNER", state["winner"], flush=True)
        return
    state = json.loads(args.report.read_text())
    assert state["target"] == args.target
    chosen = state["winner"]
    candidate = state["candidates"][chosen]
    k = candidate["iterations"]
    basekeys = ["vertices", "faces", "constraints", "fixed", "free", "tail", "head", "mass"]
    fields = {key: d[key] for key in basekeys}
    meta = {
        key: value
        for key, value in old.items()
        if key
        not in [
            "results",
            "display_order",
            "convergence",
            "scalar_range",
            "cholesky_audit",
            "refinement_audit",
        ]
    }
    meta.update(
        results={},
        display_order=[],
        convergence=dict(k=k, criterion=state["target_definition"], field_target=args.target),
        selected_fsai=chosen,
        jacobi_method="jacobi_cg",
        refresh=state,
        timing="FSAI: median of three warm repeats. Jacobi: one warm uninterrupted run per iteration budget. Setup includes factored operator and preconditioner; uploads and CPU diagnostics excluded.",
    )

    def save():
        np.savez_compressed(args.output / "fields.npz", **fields)
        (args.output / "solve.json").write_text(json.dumps(meta, indent=2) + "\n")

    def record(full, row, method, multiplier):
        name = f"{method}_{row['actual_iterations']}"
        row.update(
            iteration_multiplier=multiplier,
            field_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
            range=[float(full.min()), float(full.max())],
            bending_energy=float(0.5 * np.sum((L @ full) ** 2 / mass)),
            constraint_max_error=float(np.max(abs(full[d["fixed"]] - initial[d["fixed"]]))),
            mass_relative_error_to_refined=error(full),
            mass_relative_error_to_fsai=float(
                np.sqrt(np.sum(mass * (full - fsai_field) ** 2) / np.sum(mass * fsai_field**2))
            ),
        )
        assert row["constraint_max_error"] == 0
        meta["results"][name] = row
        meta["display_order"].append(name)
        fields[name] = full
        save()
        print("field", name, row["solve_s"], row["mass_relative_error_to_refined"], flush=True)

    fsai_field = np.load(args.output / f"{chosen}-candidate.npy")
    result = dict(candidate["repeats"][-1])
    median_trial = sorted(candidate["repeats"], key=lambda row: row["total_s"])[1]
    result.update(
        setup_s=median_trial["setup_s"],
        solve_s=median_trial["solve_s"],
        total_s=candidate["median_total_s"],
        timing_statistic="Median-total trial among three warm repeats; setup and solve are paired components of that trial",
        field_source="last validated repeat; repeated fields meet the same accuracy target",
    )
    record(fsai_field, result, chosen, 1)
    for multiplier in [1, 10, 100]:
        sol, result = run_tuned(A, L, mass, free, initial, "jacobi_cg", multiplier * k)
        assert result["actual_iterations"] == multiplier * k
        full = initial.copy()
        full[free] = sol
        result["total_s"] = result["setup_s"] + result["solve_s"]
        result["timing_statistic"] = "single warm run"
        record(full, result, "jacobi_cg", multiplier)
    meta["scalar_range"] = [
        min(float(fields[key].min()) for key in meta["display_order"]),
        max(float(fields[key].max()) for key in meta["display_order"]),
    ]
    np.save(args.output / "refined_reference.npy", reference)
    save()


if __name__ == "__main__":
    main()

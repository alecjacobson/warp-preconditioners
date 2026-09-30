"""GPU elasticity comparison: scalar Jacobi, PR #1890 block Jacobi, and FSAI.

NumPy/SciPy are used for mesh I/O and independent diagnostics only. Assembly,
preconditioner construction, the reference, and every solve run in Warp.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from elasticity_problem import assemble, cpu_matrix, independent_force
from residual_history import sample_iterations
from warp.optim import linear
from warp_pr1890 import _make_block_jacobi_preconditioner

from warp_preconditioners import FSAI, BlockJacobi, SparseOperator

PR_SHA = "0b58bcf2320b77e548bbbac0292f7f4415455eec"


def norm(x):
    return float(np.sqrt(np.sum(x * x)))


def make_pre(a, config):
    kind = config["kind"]
    if kind == "jacobi":
        return linear.preconditioner(a, "diag")
    if kind == "block":
        return (
            BlockJacobi(a)
            if config["strategy"] == "ours"
            else _make_block_jacobi_preconditioner(a, config["strategy"])
        )
    return FSAI(
        a,
        max_row_size=config["width"],
        kap_tolerance=config.get("kap", 0.003),
        apply_lanes=config.get("lanes", 4),
        factor_dtype=wp.float32 if config.get("storage", 32) == 32 else wp.float64,
    )


def timed_solve(operator, b, pre, solver, iterations, tol=0.0):
    x = wp.zeros_like(b)
    wp.synchronize()
    start = time.perf_counter()
    nit, recursive, _ = solver(
        operator, b, x, M=pre, maxiter=iterations, tol=tol, atol=0.0, check_every=0
    )
    wp.synchronize()
    elapsed = time.perf_counter() - start
    nit = int(nit.numpy()[0]) if isinstance(nit, wp.array) else int(nit)
    return x, nit, elapsed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path("data/simjeb"))
    p.add_argument("--output", type=Path, default=Path("results/elasticity.json"))
    p.add_argument("--stage", choices=["reference", "sweep", "final"], required=True)
    p.add_argument("--configs", type=Path)
    p.add_argument("--target", type=float, default=1e-4)
    args = p.parse_args()
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    mesh = np.load(args.data / "mesh.npz")
    vertices, tets, fixed = mesh["vertices"], mesh["tets"], mesh["fixed"]
    mesh_meta = json.loads((args.data / "mesh.json").read_text())
    young, poisson, density = (
        mesh_meta.get(k, v)
        for k, v in [("young_pa", 1e7), ("poisson", 0.35), ("density_kg_m3", 1000.0)]
    )
    A, b, volume, free = assemble(vertices, tets, fixed, young, poisson, density)
    if "forces" in mesh:
        b = wp.array(mesh["forces"][free], dtype=wp.vec3d, device=A.device)
    cpu = cpu_matrix(A)
    mass = volume.numpy()[free]
    b_np = b.numpy().ravel()
    bnorm = norm(b_np)
    operator = A
    state = dict(
        mesh=mesh_meta,
        warp=wp.__version__,
        gpu=wp.get_device().name,
        young_pa=young,
        poisson=poisson,
        density_kg_m3=density,
        gravity_m_s2=None if "forces" in mesh else [0, 0, -9.81],
        pr1890_commit=PR_SHA,
        target=args.target,
        target_definition="Both relative lumped-mass displacement error and relative energy error <= target",
        timing="Warm wall clock including preconditioner setup and solver graph capture; assembly, uploads, JIT, and independent diagnostics excluded",
        reference_backend="Pure Warp CG, independently checked with CPU element strain/stress evaluation",
        results=[],
    )
    if args.output.exists():
        state = json.loads(args.output.read_text())
        assert state["mesh"]["mesh_sha256"] == mesh_meta["mesh_sha256"]
        assert state["target"] == args.target

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(state, indent=2) + "\n")

    if args.stage == "reference":
        probe = np.random.default_rng(17).normal(size=(len(vertices), 3))
        probe[fixed] = 0
        independent = independent_force(vertices, tets, probe, young, poisson)[free].ravel()
        assembly_error = norm(cpu @ probe[free].ravel() - independent) / norm(independent)
        symmetry_error = norm((cpu - cpu.T).data) / norm(cpu.data)
        assert assembly_error < 1e-12 and symmetry_error < 1e-12
        assert np.all(mass > 0)
        # Compare exact local block inverses and GPU application costs.
        pre_checks = {}
        x = wp.array(probe[free], dtype=wp.vec3d)
        y = wp.zeros_like(x)
        ours = BlockJacobi(A)
        ours.matvec(x, y, y, 1.0, 0.0)
        expected = y.numpy()
        for strategy in ["ours", "auto", "direct", "sequential", "tile"]:
            config = dict(kind="block", strategy=strategy)
            pre = make_pre(A, config)
            pre.matvec(x, y, y, 1.0, 0.0)
            wp.synchronize()
            start = time.perf_counter()
            pre = make_pre(A, config)
            wp.synchronize()
            setup_s = time.perf_counter() - start
            pre.matvec(x, y, y, 1.0, 0.0)
            error = norm(y.numpy() - expected) / norm(expected)
            assert error < 1e-12, (strategy, error)
            with wp.ScopedCapture() as capture:
                for _ in range(100):
                    pre.matvec(x, y, y, 1.0, 0.0)
            wp.capture_launch(capture.graph)
            wp.synchronize()
            times = []
            for _ in range(5):
                start = time.perf_counter()
                wp.capture_launch(capture.graph)
                wp.synchronize()
                times.append((time.perf_counter() - start) / 100)
            pre_checks[strategy] = dict(
                relative_difference=error, setup_s=setup_s, apply_s=float(np.median(times))
            )
            print("block check", strategy, pre_checks[strategy], flush=True)
        # Pick a common operator independently of the preconditioner.
        operators = {}
        for lanes in [0, 4, 8, 16]:
            op = linear.aslinearoperator(A) if lanes == 0 else SparseOperator(A, row_lanes=lanes)
            op.matvec(x, y, y, 1.0, 0.0)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(50):
                    op.matvec(x, y, y, 1.0, 0.0)
            wp.capture_launch(capture.graph)
            wp.synchronize()
            start = time.perf_counter()
            wp.capture_launch(capture.graph)
            wp.synchronize()
            operators[lanes] = (time.perf_counter() - start) / 50
        lanes = min(operators, key=operators.get)
        state["operator_lanes"] = lanes
        state["operator_apply_s"] = operators
        operator = A if lanes == 0 else SparseOperator(A, row_lanes=lanes)
        print("operator", operators, "selected", lanes, flush=True)
        pre = FSAI(A, max_row_size=48, kap_tolerance=0.001, apply_lanes=4)
        x, nit, elapsed = timed_solve(operator, b, pre, linear.cg, 50000, 1e-12)
        ref = x.numpy().ravel()
        residual = b_np - cpu @ ref
        print("reference", nit, elapsed, norm(residual) / bnorm, flush=True)
        # Correct the first reference with a different preconditioner; this also
        # measures the change to its displacement and guards against drift.
        correction, cnit, _ = timed_solve(
            operator,
            wp.array(residual.reshape(-1, 3), dtype=wp.vec3d),
            BlockJacobi(A),
            linear.cg,
            50000,
            1e-10,
        )
        delta = correction.numpy().ravel()
        refined = ref + delta
        correction_mass = np.sqrt(
            np.sum(mass[:, None] * delta.reshape(-1, 3) ** 2)
            / np.sum(mass[:, None] * ref.reshape(-1, 3) ** 2)
        )
        full = np.zeros_like(vertices)
        full[free] = refined.reshape(-1, 3)
        independent = independent_force(vertices, tets, full, young, poisson)[free].ravel()
        residual_error = norm(independent - b_np) / bnorm
        assert correction_mass < 1e-7 and residual_error < 1e-8
        np.save(args.data / "reference.npy", full)
        state["verification"] = dict(
            assembly_relative_error=assembly_error,
            symmetry_relative_error=symmetry_error,
            reference_iterations=nit,
            correction_iterations=cnit,
            reference_correction_mass_error=float(correction_mass),
            reference_element_relative_residual=residual_error,
            maximum_displacement_m=float(np.linalg.norm(full, axis=1).max()),
            block_jacobi=pre_checks,
        )
        save()
        return

    lanes = state["operator_lanes"]
    operator = A if lanes == 0 else SparseOperator(A, row_lanes=lanes)
    reference = np.load(args.data / "reference.npy")[free].ravel()
    ref_mass = float(np.sum(mass[:, None] * reference.reshape(-1, 3) ** 2))
    ref_energy = float(reference @ (cpu @ reference))

    def errors(x):
        field = x.numpy().ravel() if isinstance(x, wp.array) else x.ravel()
        e = field - reference
        return dict(
            mass_error=float(np.sqrt(np.sum(mass[:, None] * e.reshape(-1, 3) ** 2) / ref_mass)),
            energy_error=float(np.sqrt(max(0.0, e @ (cpu @ e)) / ref_energy)),
            relative_residual=norm(b_np - cpu @ field) / bnorm,
        )

    if args.stage == "sweep":
        if args.configs:
            configs = json.loads(args.configs.read_text())
        else:
            configs = []
            for solver in ["cg", "cr"]:
                configs.append(dict(kind="jacobi", solver=solver))
                for strategy in ["ours", "direct", "sequential", "tile"]:
                    configs.append(dict(kind="block", solver=solver, strategy=strategy))
                for width in [4, 8, 16, 32, 48]:
                    configs.append(
                        dict(
                            kind="fsai", solver=solver, width=width, kap=0.003, lanes=4, storage=32
                        )
                    )
        for config in configs:
            if any(r["config"] == config for r in state["results"]):
                continue
            pre = make_pre(A, config)
            solver = getattr(linear, config["solver"])
            timed_solve(operator, b, pre, solver, 10)
            wp.synchronize()
            start = time.perf_counter()
            pre = make_pre(A, config)
            wp.synchronize()
            setup_s = time.perf_counter() - start
            x, nit, elapsed = timed_solve(operator, b, pre, solver, 30000, 1e-8)
            tight_iterations = nit
            # Compare settings at the same independently verified field accuracy,
            # rather than granting a win for a small recursive residual alone.
            probes = []
            x.zero_()
            checkpoints = np.unique(np.rint(np.geomspace(1, nit, 70)).astype(int))

            def observe_sweep(iteration, recursive):
                probes.append(dict(iteration=iteration, **errors(x)))

            with sample_iterations(checkpoints, observe_sweep):
                solver(operator, b, x, M=pre, maxiter=nit, tol=0.0, atol=0.0, check_every=0)
            hits = [s for s in probes if max(s["mass_error"], s["energy_error"]) <= args.target]
            if hits:
                nit = hits[0]["iteration"]
            x, nit, elapsed = timed_solve(operator, b, pre, solver, nit)
            metrics = errors(x)
            row = dict(
                config=config,
                setup_s=setup_s,
                solve_s=elapsed,
                total_s=setup_s + elapsed,
                iterations=nit,
                tight_iterations=tight_iterations,
                accuracy_probes=probes,
                **metrics,
            )
            row["passed"] = max(metrics["mass_error"], metrics["energy_error"]) <= args.target
            state["results"].append(row)
            save()
            print({k: v for k, v in row.items() if k != "accuracy_probes"}, flush=True)
        return

    # Finalists are chosen within each family from the recorded sweep.
    finalists = {}
    state["selection_repeats"] = {}
    for family in ["jacobi", "block", "fsai"]:
        candidates = [r for r in state["results"] if r["config"]["kind"] == family and r["passed"]]
        candidates.sort(key=lambda r: r["total_s"])
        shortlist = [r for r in candidates if r["total_s"] <= 1.12 * candidates[0]["total_s"]][:6]
        repeated = []
        for candidate in shortlist:
            config = candidate["config"]
            pre = make_pre(A, config)
            solver = getattr(linear, config["solver"])
            timed_solve(operator, b, pre, solver, 10)
            probes = candidate["accuracy_probes"]
            hit = next(s for s in probes if max(s["mass_error"], s["energy_error"]) <= args.target)
            prev = max(s["iteration"] for s in probes if s["iteration"] < hit["iteration"])
            stop = hit["iteration"]
            for it in range(prev + 1, stop + 1, max(1, (stop - prev) // 16)):
                trial, _, _ = timed_solve(operator, b, pre, solver, it)
                metric = errors(trial)
                if max(metric["mass_error"], metric["energy_error"]) <= args.target:
                    stop = it
                    break
            totals = []
            for _ in range(3):
                wp.synchronize()
                start = time.perf_counter()
                pre = make_pre(A, config)
                wp.synchronize()
                setup_s = time.perf_counter() - start
                trial, _, elapsed = timed_solve(operator, b, pre, solver, stop)
                totals.append(setup_s + elapsed)
            row = dict(
                config=config,
                iterations=stop,
                total_repeats_s=totals,
                total_s=float(np.median(totals)),
                **errors(trial),
            )
            repeated.append(row)
            print("selection", row, flush=True)
        state["selection_repeats"][family] = repeated
        finalists[family] = min(repeated, key=lambda r: r["total_s"])["config"]
        save()
    state["finalists"] = {}
    for family, config in finalists.items():
        pre = make_pre(A, config)
        solver = getattr(linear, config["solver"])
        timed_solve(operator, b, pre, solver, 10)
        x = wp.zeros_like(b)
        sample_rows = []
        baseline = next(r for r in state["results"] if r["config"] == config)
        stops = np.unique(
            np.r_[
                np.rint(np.geomspace(1, baseline["iterations"], 65)).astype(int),
                baseline["iterations"],
            ]
        )

        def observe(iteration, recursive):
            sample_rows.append(
                dict(
                    iteration=iteration, recursive_relative_residual=recursive / bnorm, **errors(x)
                )
            )

        with sample_iterations(stops, observe):
            solver(operator, b, x, M=pre, tol=0.0, atol=0.0, maxiter=int(stops[-1]), check_every=0)
        crossing = next(
            s for s in sample_rows if max(s["mass_error"], s["energy_error"]) <= args.target
        )
        # Refine the crossing to a small interval without assuming monotonicity
        # beyond the sampled bracket. Every solve starts from zero.
        prev = max(s["iteration"] for s in sample_rows if s["iteration"] < crossing["iteration"])
        stop = crossing["iteration"]
        for it in range(prev + 1, stop + 1, max(1, (stop - prev) // 12)):
            trial, _, _ = timed_solve(operator, b, pre, solver, it)
            metric = errors(trial)
            if max(metric["mass_error"], metric["energy_error"]) <= args.target:
                stop = it
                break
        setup_times, solve_times = [], []
        for _ in range(3):
            wp.synchronize()
            start = time.perf_counter()
            pre = make_pre(A, config)
            wp.synchronize()
            setup_times.append(time.perf_counter() - start)
            final, _, elapsed = timed_solve(operator, b, pre, solver, stop)
            solve_times.append(elapsed)
        setup_s = float(np.median(setup_times))
        # Time prefixes independently: CPU error computation is never charged
        # to a solver, and the plotted times include real graph-capture costs.
        time_stops = {int(i) for i in np.rint(np.geomspace(1, int(stops[-1]), 22))} | {stop}
        timed_rows = []
        for it in sorted(time_stops):
            trial, _, elapsed = timed_solve(operator, b, pre, solver, it)
            timed_rows.append(
                dict(iteration=it, solve_s=elapsed, total_s=setup_s + elapsed, **errors(trial))
            )
        row = dict(
            config=config,
            iterations=stop,
            setup_s=setup_s,
            solve_s=float(np.median(solve_times)),
            total_s=float(np.median(np.array(setup_times) + solve_times)),
            setup_repeats_s=setup_times,
            solve_repeats_s=solve_times,
            samples=sample_rows,
            timed_prefixes=timed_rows,
            **errors(final),
        )
        state["finalists"][family] = row
        save()
        print(
            "finalist",
            family,
            {k: v for k, v in row.items() if k not in ["samples", "timed_prefixes"]},
            flush=True,
        )
    winner = min(state["finalists"], key=lambda k: state["finalists"][k]["total_s"])
    budget = state["finalists"][winner]["total_s"]
    state["winner"], state["snapshot_budget_s"] = winner, budget
    fields = dict(
        vertices=vertices,
        faces=mesh["faces"],
        fixed=fixed,
        reference=np.load(args.data / "reference.npy"),
    )
    for family, row in state["finalists"].items():
        pre = make_pre(A, row["config"])
        solver = getattr(linear, row["config"]["solver"])
        available = budget - row["setup_s"]
        if available <= 0:
            x = wp.zeros_like(b)
            it, elapsed = 0, 0.0
        else:
            # Use measured prefix timings to estimate a fixed iteration budget;
            # repeat after adjustment until actual elapsed time is within 5%.
            times = sorted(row["timed_prefixes"], key=lambda r: r["solve_s"])
            it = max(
                1,
                int(
                    np.interp(
                        available, [r["solve_s"] for r in times], [r["iteration"] for r in times]
                    )
                ),
            )
            if family == winner:
                it = row["iterations"]
            for _ in range(6):
                x, _, elapsed = timed_solve(operator, b, pre, solver, it)
                if abs(row["setup_s"] + elapsed - budget) / budget < 0.05:
                    break
                if family != winner:
                    it = max(1, int(it * available / elapsed))
        full = np.zeros_like(vertices)
        full[free] = x.numpy()
        fields[family] = full
        row["snapshot"] = dict(
            iterations=it,
            solve_s=elapsed,
            total_s=row["setup_s"] + elapsed,
            field_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
            **errors(x),
        )
        print("snapshot", family, row["snapshot"], flush=True)
    np.savez_compressed(args.data / "comparison.npz", **fields)
    save()


if __name__ == "__main__":
    main()

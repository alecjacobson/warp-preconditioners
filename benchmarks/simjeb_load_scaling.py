"""Measure force scaling with a fixed linear-elasticity matrix on the L40.

All references, preconditioners and solves use Warp. CPU sparse products and
independent element forces are diagnostics outside timed regions.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from elasticity import make_pre, norm, timed_solve
from elasticity_problem import assemble, cpu_matrix, independent_force
from residual_history import sample_iterations
from warp.optim import linear

from warp_preconditioners import FSAI, BlockJacobi, SparseOperator


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/simjeb-ftetwild"))
    parser.add_argument("--baseline", type=Path, default=Path("results/simjeb-ftetwild.json"))
    parser.add_argument("--output", type=Path, default=Path("results/simjeb-load-scaling.json"))
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    assert args.repeats >= 3 and min(args.scales) > 0
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    mesh = np.load(args.data / "mesh.npz")
    base = json.loads(args.baseline.read_text())
    assert (
        hashlib.sha256((args.data / "mesh.npz").read_bytes()).hexdigest()
        == base["mesh"]["mesh_sha256"]
    )
    v, t, fixed = mesh["vertices"], mesh["tets"], mesh["fixed"]
    young, poisson = base["young_pa"], base["poisson"]
    A, _, volumes, free = assemble(v, t, fixed, young, poisson, base["density_kg_m3"])
    matrix_hash = digest(A.values.numpy())
    cpu = cpu_matrix(A)
    mass = volumes.numpy()[free]
    lanes = base["operator_lanes"]
    operator = A if lanes == 0 else SparseOperator(A, row_lanes=lanes)
    target = base["target"]
    baseline_ref = np.load(args.data / "reference.npy")
    configs = {k: r["config"] for k, r in base["finalists"].items()}
    outdir = args.data / "load-scaling"
    outdir.mkdir(exist_ok=True)
    state = dict(
        mesh=base["mesh"],
        gpu=wp.get_device().name,
        warp=wp.__version__,
        baseline=str(args.baseline),
        target=target,
        configs=configs,
        scales=args.scales,
        model="Fixed small-strain linear elasticity; only the RHS force magnitude changes",
        matrix_values_sha256=matrix_hash,
        repeats=args.repeats,
        timing="Warm wall clock, fresh setup plus graph-captured solve; allocation of x, assembly, JIT, uploads, diagnostics and meshing excluded",
        stopping="First sampled crossing of BOTH relative lumped-mass displacement and energy errors, then validate uninstrumented solve; 4-iteration spacing near baseline crossing",
        tuning="Hold the best tested baseline configurations fixed to isolate load scaling; no new parameter sweep",
        cases=[],
    )

    def save():
        args.output.write_text(json.dumps(state, indent=2) + "\n")

    refpre = FSAI(A, max_row_size=48, kap_tolerance=0.001, apply_lanes=4)
    correction_pre = BlockJacobi(A)
    jobs = []
    case_data = []
    for scale in args.scales:
        force = mesh["forces"] * scale
        b_np = force[free].ravel()
        bnorm = norm(b_np)
        b = wp.array(force[free], dtype=wp.vec3d)
        x, nit, _ = timed_solve(operator, b, refpre, linear.cg, 50000, 1e-12)
        residual = b_np - cpu @ x.numpy().ravel()
        correction, _, _ = timed_solve(
            operator,
            wp.array(residual.reshape(-1, 3), dtype=wp.vec3d),
            correction_pre,
            linear.cg,
            50000,
            1e-10,
        )
        reference = x.numpy().ravel() + correction.numpy().ravel()
        full = np.zeros_like(v)
        full[free] = reference.reshape(-1, 3)
        independent = independent_force(v, t, full, young, poisson)[free].ravel()
        residual_error = norm(independent - b_np) / bnorm
        scaling_error = norm(full / scale - baseline_ref) / norm(baseline_ref)
        assert residual_error < 1e-8 and scaling_error < 1e-7
        ref_mass = np.sum(mass[:, None] * reference.reshape(-1, 3) ** 2)
        ref_energy = reference @ (cpu @ reference)

        def errors(
            field,
            reference=reference,
            b_np=b_np,
            bnorm=bnorm,
            ref_mass=ref_mass,
            ref_energy=ref_energy,
        ):
            u = field.numpy().ravel() if isinstance(field, wp.array) else field.ravel()
            e = u - reference
            return dict(
                mass_error=float(np.sqrt(np.sum(mass[:, None] * e.reshape(-1, 3) ** 2) / ref_mass)),
                energy_error=float(np.sqrt(max(0, e @ (cpu @ e)) / ref_energy)),
                relative_residual=norm(b_np - cpu @ u) / bnorm,
            )

        case = dict(
            scale=scale,
            resultant_n=force.sum(0).tolist(),
            reference_iterations=nit,
            reference_element_relative_residual=residual_error,
            reference_scaling_relative_error=scaling_error,
            reference_max_displacement_mm=float(np.linalg.norm(full, axis=1).max() * 1000),
            methods={},
        )
        fields = dict(vertices=v, faces=mesh["faces"], fixed=fixed, reference=full)
        state["cases"].append(case)
        case_data.append((case, b, errors, fields))
        for family, config in configs.items():
            pre = make_pre(A, config)
            solver = getattr(linear, config["solver"])
            timed_solve(operator, b, pre, solver, 10)
            center = base["finalists"][family]["iterations"]
            end = int(center * 1.15) + 40
            stops = np.unique(
                np.r_[
                    np.rint(np.geomspace(1, end, 65)).astype(int),
                    np.arange(max(1, center - 120), min(end, center + 120) + 1, 4),
                ]
            )
            trial = wp.zeros_like(b)
            samples = []

            def observe(iteration, recursive):
                samples.append(
                    dict(
                        iteration=iteration,
                        recursive_relative_residual=recursive / bnorm,
                        **errors(trial),
                    )
                )

            with sample_iterations(stops, observe):
                solver(operator, b, trial, M=pre, maxiter=end, tol=0.0, atol=0.0, check_every=0)
            hit = next(s for s in samples if max(s["mass_error"], s["energy_error"]) <= target)
            stop = hit["iteration"]
            for _ in range(10):
                trial, _, _ = timed_solve(operator, b, pre, solver, stop)
                metrics = errors(trial)
                if max(metrics["mass_error"], metrics["energy_error"]) <= target:
                    break
                stop += 4
            assert max(metrics["mass_error"], metrics["energy_error"]) <= target
            row = dict(config=config, iterations=stop, samples=samples, repeats=[], **metrics)
            case["methods"][family] = row
            jobs.append((case, b, errors, family, row))
            print("accuracy", scale, family, stop, metrics, flush=True)
        save()
    # Randomized interleaving avoids attributing warmup/drift to larger loads.
    schedule = [(job, repeat) for repeat in range(args.repeats) for job in range(len(jobs))]
    np.random.default_rng(41).shuffle(schedule)
    for job_index, repeat in schedule:
        case, b, errors, family, row = jobs[job_index]
        wp.synchronize()
        start = time.perf_counter()
        pre = make_pre(A, row["config"])
        wp.synchronize()
        setup = time.perf_counter() - start
        trial, _, elapsed = timed_solve(
            operator, b, pre, getattr(linear, row["config"]["solver"]), row["iterations"]
        )
        metric = errors(trial)
        assert max(metric["mass_error"], metric["energy_error"]) <= target
        row["repeats"].append(
            dict(repeat=repeat, setup_s=setup, solve_s=elapsed, total_s=setup + elapsed, **metric)
        )
    for case, b, errors, fields in case_data:
        for family, row in case["methods"].items():
            for key in ["setup_s", "solve_s", "total_s"]:
                values = [r[key] for r in row["repeats"]]
                row[key] = float(np.median(values))
                row[key + "_min"] = float(min(values))
                row[key + "_max"] = float(max(values))
        winner = min(case["methods"], key=lambda k: case["methods"][k]["total_s"])
        budget = case["methods"][winner]["total_s"]
        case.update(winner=winner, snapshot_budget_s=budget)
        for family, row in case["methods"].items():
            pre = make_pre(A, row["config"])
            solver = getattr(linear, row["config"]["solver"])
            available = max(0, budget - row["setup_s"])
            iterations = max(1, round(row["iterations"] * available / row["solve_s"]))
            if family == winner:
                iterations = row["iterations"]
            for _ in range(8):
                trial, _, elapsed = timed_solve(operator, b, pre, solver, iterations)
                if abs(row["setup_s"] + elapsed - budget) / budget < 0.05:
                    break
                if family != winner:
                    iterations = max(1, round(iterations * available / elapsed))
            field = np.zeros_like(v)
            field[free] = trial.numpy()
            fields[family] = field
            row["snapshot"] = dict(
                iterations=iterations,
                solve_s=elapsed,
                total_s=row["setup_s"] + elapsed,
                field_sha256=digest(field),
                **errors(trial),
            )
        case["reference_sha256"] = digest(fields["reference"])
        case["fields_file"] = str(outdir / f"scale-{case['scale']:g}.npz")
        np.savez_compressed(case["fields_file"], **fields)
        print(
            "timing",
            case["scale"],
            {k: (r["iterations"], r["total_s"]) for k, r in case["methods"].items()},
            flush=True,
        )
    assert matrix_hash == digest(A.values.numpy())
    state["matrix_unchanged_verified"] = True
    save()


if __name__ == "__main__":
    main()

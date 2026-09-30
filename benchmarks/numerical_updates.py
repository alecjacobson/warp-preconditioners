"""Warm setup and fixed-sparsity value updates on the SimJEB bracket.

Every matrix assembly, preconditioner, diagnostic and CG solve runs in Warp.
NumPy supplies spatial material fields and verifies downloaded solutions only.
Assembly is excluded from setup/solve timing and is identical for all methods.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from elasticity import timed_solve
from elasticity_problem import assemble
from warp.optim import linear

from warp_preconditioners import FSAI, BlockJacobi


def timed(fn):
    wp.synchronize()
    start = time.perf_counter()
    result = fn()
    wp.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--width", type=int, default=8)
    options = parser.parse_args()
    output = Path(f"results/numerical-updates-w{options.width}.json")
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    mesh = np.load("data/simjeb/mesh.npz")
    meta = json.loads(Path("data/simjeb/mesh.json").read_text())
    v, t, fixed = mesh["vertices"], mesh["tets"], mesh["fixed"]

    def matrix(scale=None):
        return assemble(
            v,
            t,
            fixed,
            meta["young_pa"],
            meta["poisson"],
            meta["density_kg_m3"],
            material_scale=scale,
        )[0]

    A = matrix()
    A.nnz_sync()
    b = wp.array(mesh["forces"][~fixed], dtype=wp.vec3d)
    bnorm = np.linalg.norm(b.numpy())
    args = dict(max_row_size=options.width, kap_tolerance=0.003, apply_lanes=4)
    state = dict(
        mesh=meta,
        warp=wp.__version__,
        gpu=wp.get_device().name,
        config=args,
        baseline_commit="164d848",
        protocol="Warm medians of five setup calls; median of three zero-start CG solves per method per matrix, tolerance 1e-8. Assembly/JIT excluded, source nnz counts settled before timing; diagnostic separately timed.",
        setup={},
        sequence=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        baseline = {}
        for name, cls in [("fsai", "FSAI"), ("block_ilu", "BlockJacobi")]:
            path = Path(tmp) / f"{name}.py"
            path.write_bytes(
                subprocess.check_output(["git", "show", f"164d848:warp_preconditioners/{name}.py"])
            )
            module_name = f"warp_preconditioners._baseline_{name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            baseline[cls] = getattr(module, cls)
        scalar = sp.bsr_copy(A, block_shape=(1, 1))
        scalar.nnz_sync()
        factories = {
            "block_old": lambda: baseline["BlockJacobi"](A),
            "block_new": lambda: BlockJacobi(A),
            "fsai_bsr_old": lambda: baseline["FSAI"](A, **args),
            "fsai_bsr_new": lambda: FSAI(A, **args),
            "fsai_csr_old": lambda: baseline["FSAI"](scalar, **args),
            "fsai_csr_new": lambda: FSAI(scalar, **args),
            "fsai_reuse_plan": lambda: FSAI(A, **args, reuse_pattern=True),
        }
        for fn in factories.values():
            fn()
        samples = {name: [] for name in factories}
        names = list(factories)
        for repeat in range(5):
            for name in names[repeat:] + names[:repeat]:
                samples[name].append(timed(factories[name])[1])
        for name, times in samples.items():
            state["setup"][name] = dict(median_s=float(np.median(times)), samples_s=times)
            print(name, state["setup"][name], flush=True)
        old = baseline["FSAI"](A, **args)
        new = FSAI(A, **args)
        np.testing.assert_array_equal(old.G.offsets.numpy(), new.G.offsets.numpy())
        nnz = old.G.nnz_sync()
        np.testing.assert_array_equal(old.G.columns.numpy()[:nnz], new.G.columns.numpy()[:nnz])
        np.testing.assert_allclose(
            old.G.values.numpy()[:nnz], new.G.values.numpy()[:nnz], rtol=1e-13, atol=1e-20
        )
        state["identical_initial_factor"] = True
    reuse = FSAI(A, **args, reuse_pattern=True)
    stale = FSAI(A, **args)
    block = BlockJacobi(A)
    # Warm the numeric-refit kernels, quality products and all solver variants.
    reuse.update(A)
    block.update(A)
    reuse.quality(A)
    for pre in [reuse, stale, block]:
        timed_solve(A, b, pre, linear.cg, 10)
    centroid = v[t].mean(1)
    pos = (centroid - v.min(0)) / np.ptp(v, axis=0)
    field = np.sin(2 * np.pi * pos[:, 0]) * np.cos(np.pi * pos[:, 1]) * np.sin(np.pi * pos[:, 2])
    amplitudes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0]
    for step, amp in enumerate(amplitudes):
        scale = np.exp(amp * field)
        changed = matrix(scale)
        np.testing.assert_array_equal(changed.offsets.numpy(), A.offsets.numpy())
        np.testing.assert_array_equal(
            changed.columns.numpy()[: changed.nnz_sync()], A.columns.numpy()[: A.nnz_sync()]
        )
        fresh, setup = timed(lambda: FSAI(changed, **args))
        _, update = timed(lambda: reuse.update(changed))
        _, blockupdate = timed(lambda: block.update(changed))
        record = dict(
            step=step,
            amplitude=amp,
            modulus_min_multiplier=float(scale.min()),
            modulus_max_multiplier=float(scale.max()),
            methods={},
        )
        solutions = {}
        # Rotate measurement order to reduce a fixed ordering bias.
        methods = [
            ("fresh", fresh, setup),
            ("refit", reuse, update),
            ("stale", stale, 0.0),
            ("block_update", block, blockupdate),
        ]
        methods = methods[step % 4 :] + methods[: step % 4]
        for name, pre, setup_s in methods:
            solve_times = []
            for _ in range(3):
                x, nit, elapsed = timed_solve(changed, b, pre, linear.cg, 50000, 1e-8)
                solve_times.append(elapsed)
            elapsed = float(np.median(solve_times))
            check = wp.clone(b)
            sp.bsr_mv(changed, x, check, alpha=-1.0, beta=1.0)
            residual = float(np.linalg.norm(check.numpy()) / bnorm)
            assert residual < 1e-7, (step, name, residual)
            solutions[name] = x.numpy()
            row = dict(
                setup_s=setup_s,
                solve_s=elapsed,
                solve_repeats_s=solve_times,
                total_s=setup_s + elapsed,
                iterations=nit,
                true_relative_residual=residual,
            )
            if name != "block_update":
                defect, cost = timed(lambda: pre.quality(changed, probes=4, seed=17))
                row.update(frobenius_defect=defect, quality_s=cost)
            record["methods"][name] = row
        for name, solution in solutions.items():
            error = float(
                np.linalg.norm(solution - solutions["fresh"]) / np.linalg.norm(solutions["fresh"])
            )
            record["methods"][name]["relative_difference_from_fresh_solution"] = error
            assert error < 1e-5, (step, name, error)
        state["sequence"].append(record)
        print("step", step, record, flush=True)
        output.write_text(json.dumps(state, indent=2) + "\n")
    state["totals"] = {}
    for name in ["fresh", "refit", "stale", "block_update"]:
        initial = (
            0.0
            if name == "fresh"
            else state["setup"][
                {"refit": "fsai_reuse_plan", "stale": "fsai_bsr_new", "block_update": "block_new"}[
                    name
                ]
            ]["median_s"]
        )
        rows = [step["methods"][name] for step in state["sequence"]]
        state["totals"][name] = dict(
            initial_setup_s=initial,
            setup_s=initial + sum(r["setup_s"] for r in rows),
            setup_and_solve_s=initial + sum(r["total_s"] for r in rows),
            with_quality_every_step_s=initial
            + sum(r["total_s"] + r.get("quality_s", 0.0) for r in rows),
        )
    output.write_text(json.dumps(state, indent=2) + "\n")
    print(state["totals"], flush=True)


if __name__ == "__main__":
    main()

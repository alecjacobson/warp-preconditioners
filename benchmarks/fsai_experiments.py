"""Controlled FSAI experiments; diagnostics are outside GPU timing regions."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import warp as wp
from elasticity_problem import assemble, cpu_matrix
from warp.optim import linear

from warp_preconditioners import FSAI, SquaredLaplacianOperator
from warp_preconditioners.sparse_operator import _matvec

BASELINE = "8f25bf7"


def baseline_class(variant="baseline"):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:warp_preconditioners/fsai.py"], text=True
    )
    if variant in ["forward", "dedup_forward"]:
        source = source.replace(
            "        z = vector(dtype(0))\n",
            "        z = vector(dtype(0))\n        v = vector(dtype(0))\n",
            1,
        )
        source = source.replace(
            "        z[0] = dtype(1)\n", "        z[0] = dtype(1)\n        v[0] = dtype(1)\n", 1
        )
        source = source.replace(
            "            size += 1\n            old_z0",
            "            rhs = dtype(0)\n            for p in range(size):\n                rhs -= w[p] * v[p]\n            v[size] = rhs / chol[size, size]\n            size += 1\n            old_z0",
            1,
        )
        start = source.index("            v = vector(dtype(0))", source.index("# A[S,S] z"))
        end = source.index("            for rev in range(size):", start)
        source = source[:start] + source[end:]
    if variant == "dedup_forward":
        source = source.replace(
            "                    if not selected:\n                        residual",
            "                    if not selected:\n                        for prior in range(p):\n                            if entry(offsets, columns, values, pattern[prior], c) != dtype(0):\n                                selected = True\n                                break\n                    if not selected:\n                        residual",
            1,
        )
    path = Path(f"data/fsai-experiments/fsai_{variant}_8f25bf7.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"warp_preconditioners._{variant}_8f25bf7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FSAI


class SplitFSAI(FSAI):
    def __init__(self, a, lanes, **kw):
        self.lanes = lanes
        super().__init__(a, **kw)

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(self.scalar_type).flatten() for v in (x, y, z))
        _matvec(self.G, x, self._tmp, self._tmp, 1.0, 0.0, self.lanes[0])
        _matvec(self.GT, self._tmp, y, z, alpha, beta, self.lanes[1])


def load_problem(name):
    if name == "dragon":
        import scipy.sparse as ss
        from dragon import upload

        path = Path("data/dirichlet-current")
        d = np.load(path / "fields.npz")
        stamp = json.loads((path / "tuning_inputs.json").read_text())
        digest = hashlib.sha256()
        for key in ["vertices", "mass", "free", "constraints"]:
            digest.update(d[key].tobytes())
        assert digest.hexdigest() == stamp["data_sha256"]
        lap = ss.load_npz(path / "tuning_laplacian.npz")
        free, mass = d["free"], d["mass"]
        q = lap @ ss.diags(1 / mass) @ lap
        a = upload(q[free][:, free].tocsr(), "cuda:0")
        op = SquaredLaplacianOperator(
            upload(lap, "cuda:0"),
            wp.array(mass, dtype=wp.float64),
            wp.array(free, dtype=int),
            row_lanes=4,
        )
        b = op.rhs(wp.array(d["constraints"], dtype=wp.float64))
        ref = np.load(path / "refined_reference.npy")
        denom = np.sum(mass * ref**2)

        def errors(x):
            return {
                "field_error": float(
                    np.sqrt(np.sum(mass[free] * (x.numpy() - ref[free]) ** 2) / denom)
                )
            }

        return a, op, b, errors, 48, 4
    path = Path("data") / name
    d = np.load(path / "mesh.npz")
    meta = json.loads((path / "mesh.json").read_text())
    a, b, volume, free = assemble(
        d["vertices"],
        d["tets"],
        d["fixed"],
        meta["young_pa"],
        meta["poisson"],
        meta["density_kg_m3"],
    )
    b = wp.array(d["forces"][free], dtype=wp.vec3d)
    ref = np.load(path / "reference.npy")[free].ravel()
    cpu = cpu_matrix(a)
    mass = volume.numpy()[free, None]
    denom = np.sum(mass * ref.reshape(-1, 3) ** 2)
    energy = ref @ (cpu @ ref)

    def errors(x):
        e = x.numpy().ravel() - ref
        return {
            "field_error": float(np.sqrt(np.sum(mass * e.reshape(-1, 3) ** 2) / denom)),
            "energy_error": float(np.sqrt(max(0.0, e @ (cpu @ e)) / energy)),
        }

    return a, a, b, errors, 4, 1


def solve(op, b, pre, iterations, tol=0.0):
    x = wp.zeros_like(b)
    wp.synchronize()
    start = time.perf_counter()
    nit, _, _ = linear.cg(op, b, x, M=pre, maxiter=iterations, tol=tol, atol=0.0, check_every=0)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return x, int(nit.numpy()[0]), elapsed


def require_exclusive_gpu():
    """Reject shared-GPU timings rather than silently reporting interference."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader", "--id=0"],
        text=True,
    )
    others = [line for line in output.splitlines() if int(line.split(",", 1)[0]) != os.getpid()]
    if others:
        raise RuntimeError("Other GPU processes are active; retry when idle: " + "; ".join(others))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--problem", default="simjeb-ftetwild", choices=["dragon", "simjeb", "simjeb-ftetwild"]
    )
    p.add_argument(
        "--stage",
        choices=[
            "lanes",
            "setup",
            "setup_ablation",
            "block",
            "block_apply",
            "batch",
            "fused",
            "final",
        ],
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--width", type=int)
    p.add_argument("--block-widths", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    args = p.parse_args()
    require_exclusive_gpu()
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    a, op, b, errors, width, lanes = load_problem(args.problem)
    width = args.width or width
    a.nnz_sync()
    signature = hashlib.sha256()
    for array in [
        a.offsets.numpy(),
        a.columns.numpy()[: a.nnz],
        a.values.numpy()[: a.nnz],
        b.numpy(),
    ]:
        signature.update(array.tobytes())
    provenance = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        matrix_rhs_sha256=signature.hexdigest(),
        library_sha256=hashlib.sha256(
            Path("warp_preconditioners/fsai.py").read_bytes()
        ).hexdigest(),
        clocks="unlocked; untimed native solve immediately before every measured setup/solve",
        gpu_isolation="Reject other compute processes at start and before/after every timed trial",
        trials="alternating forward/reverse order after warming all candidates",
    )
    old = baseline_class()
    kw = dict(max_row_size=width, kap_tolerance=0.003, apply_lanes=lanes, factor_dtype=wp.float32)
    makers = {"baseline": lambda: old(a, **kw)}
    if args.stage == "lanes":
        for g in [1, 2, 4, 8, 16]:
            for gt in [1, 2, 4, 8, 16]:
                makers[f"{g},{gt}"] = lambda g=g, gt=gt: SplitFSAI(a, (g, gt), **kw)
    elif args.stage == "fused":
        from fused_fsai_candidate import FusedFSAI

        makers["fused"] = lambda: FusedFSAI(a, **kw)
    elif args.stage == "setup_ablation":
        for variant in ["forward", "dedup_forward"]:
            factory = baseline_class(variant)
            makers[variant] = lambda factory=factory: factory(a, **kw)
    elif args.stage == "setup":
        makers["current"] = lambda: FSAI(a, **kw)
    elif args.stage == "batch":
        for step in args.steps:
            makers[f"batch{step}"] = lambda step=step: FSAI(a, max_step_size=step, **kw)
    elif args.stage == "block_apply":
        from block_fsai_candidate import BlockFSAI, BsrBlockFSAI

        for w in args.block_widths:
            makers[f"block{w}_lanes4"] = lambda w=w: BlockFSAI(a, max_blocks=w, apply_lanes=4)
            makers[f"block{w}_bsr"] = lambda w=w: BsrBlockFSAI(a, max_blocks=w)
    elif args.stage == "block":
        from block_fsai_candidate import BlockFSAI

        for w in [1, 2, 3, 4, 6, 8]:
            makers[f"block{w}"] = lambda w=w: BlockFSAI(a, max_blocks=w)
    elif args.stage == "final":
        makers["current"] = lambda: FSAI(a, **kw)
    results = {k: {"trials": []} for k in makers}
    pres = {}
    for name, make in makers.items():
        pre = make()
        solve(op, b, pre, 10)
        pres[name] = pre
        pre.G.nnz_sync()
        results[name]["factor_nnz"] = pre.G.nnz
        results[name]["mean_factor_row_size"] = pre.G.nnz / pre.G.nrow
        if args.stage in ["block", "block_apply", "batch"]:
            x, nit, elapsed = solve(
                op,
                b,
                pre,
                100000 if args.problem == "dragon" else 10000,
                1e-12 if args.problem == "dragon" else 1e-8,
            )
            results[name]["tight_iterations"] = nit
            results[name]["tight_errors"] = errors(x)
        if args.stage in ["block", "block_apply", "batch", "fused", "final"]:
            from residual_history import sample_iterations

            limit = results[name].get(
                "tight_iterations", 50000 if args.problem == "dragon" else 4000
            )
            target = 1e-8 if args.problem == "dragon" else 1e-4
            step = 25 if args.problem == "dragon" else 10
            x = wp.zeros_like(b)
            samples = []

            def observe(it, recursive):
                samples.append(dict(iteration=it, **errors(x)))

            with sample_iterations(np.arange(step, limit + step, step), observe):
                linear.cg(op, b, x, M=pre, maxiter=limit, tol=0.0, atol=0.0, check_every=0)
            hit = next(
                (s for s in samples if max(v for k, v in s.items() if k != "iteration") <= target),
                None,
            )
            results[name]["samples"] = samples
            results[name]["iterations"] = hit["iteration"] if hit else limit
        print("warm", name, results[name].get("iterations"), flush=True)
    for name, pre in pres.items():
        if args.stage in ["setup", "setup_ablation"] or name in ["current", "batch1", "fused"]:
            for attr in ["offsets", "columns", "values"]:
                np.testing.assert_array_equal(
                    getattr(pres["baseline"].G, attr).numpy(), getattr(pre.G, attr).numpy()
                )
            results[name]["identical_baseline_factor"] = True
            print("identical factors", name, flush=True)
    # Warm every specialization before interleaved timing; alternate ordering.
    for r in range(args.repeats):
        order = list(makers)
        if r % 2:
            order.reverse()
        for name in order:
            require_exclusive_gpu()
            budget = results[name].get("iterations", 2000)
            # Keep hardware warm as well as JIT-warm; CPU diagnostics between
            # trials otherwise leave short solves sensitive to clock ramp-up.
            solve(op, b, pres[name], min(2000, budget))
            wp.synchronize()
            start = time.perf_counter()
            pre = makers[name]() if args.stage != "lanes" else pres[name]
            wp.synchronize()
            setup = time.perf_counter() - start
            budget = results[name].get("iterations", 2000)
            x, nit, elapsed = solve(op, b, pre, budget)
            require_exclusive_gpu()
            row = dict(
                setup_s=setup, solve_s=elapsed, total_s=setup + elapsed, iterations=nit, **errors(x)
            )
            residual = wp.empty_like(b)
            linear.aslinearoperator(op).matvec(x, b, residual, -1.0, 1.0)
            row["true_relative_residual"] = float(
                np.linalg.norm(residual.numpy().ravel()) / np.linalg.norm(b.numpy().ravel())
            )
            if args.stage in ["batch", "block", "block_apply", "fused", "final"]:
                target = 1e-8 if args.problem == "dragon" else 1e-4
                assert row["field_error"] <= target
                assert row.get("energy_error", 0.0) <= target
            results[name]["trials"].append(row)
            print(r, name, row, flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    baseline=BASELINE,
                    provenance=provenance,
                    problem=args.problem,
                    stage=args.stage,
                    gpu=wp.get_device().name,
                    warp=wp.__version__,
                    width=width,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )
    for k, v in results.items():
        print(k, np.median([t["total_s"] for t in v["trials"]]))


if __name__ == "__main__":
    main()

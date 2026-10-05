"""Paired FSAI dtype regression timing against a commit in the Warp checkout.

Run with the checkout's Python environment. Construction, numerical refits,
application and solves use Warp; NumPy supplies deterministic test inputs and
checks outputs. No benchmark or timing assertion belongs in Warp's unit tests.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from elasticity_problem import assemble
from warp._src.optim import fsai as candidate
from warp.optim.linear import cg


def exclusive():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    others = [
        int(line) for line in output.splitlines() if line.strip() and int(line) != os.getpid()
    ]
    if others:
        raise RuntimeError(
            f"Other GPU processes are active: {others}; discard this configuration and retry."
        )


def timed(call, device):
    wp.synchronize_device(device)
    begin = time.perf_counter()
    result = call()
    wp.synchronize_device(device)
    return result, time.perf_counter() - begin


def signature(a):
    count = a.nnz_sync()
    digest = hashlib.sha256()
    for x in (a.offsets.numpy()[: a.nrow + 1], a.columns.numpy()[:count], a.values.numpy()[:count]):
        digest.update(x.tobytes())
    return digest.hexdigest()


def grid(side, dtype, device):
    n = side * side
    ids = np.arange(n).reshape(side, side)
    horizontal = np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel()))
    vertical = np.column_stack((ids[:-1].ravel(), ids[1:].ravel()))
    edges = np.concatenate((horizontal, vertical))
    rows = np.concatenate((np.arange(n), edges[:, 0], edges[:, 1]))
    cols = np.concatenate((np.arange(n), edges[:, 1], edges[:, 0]))
    values = np.concatenate((np.full(n, 4.0), np.full(2 * len(edges), -1.0)))
    lap = sp.bsr_from_triplets(
        n,
        n,
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(values, dtype=dtype, device=device),
    )
    matrix = sp.bsr_mm(lap, lap)
    sp.bsr_axpy(sp.bsr_identity(n, block_type=dtype, device=device), matrix, alpha=0.05)
    return matrix


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--warp-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/simjeb-ftetwild"))
    parser.add_argument("--baseline", default="8b0e91e42e5b97c9693179de3dfa2419842d9fbe")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Keep completed isolated configurations."
    )
    args = parser.parse_args()
    exclusive()
    wp.init()
    gc.disable()  # Collect allocation cycles explicitly outside paired timing regions.
    wp.config.log_level = wp.LOG_WARNING
    source = subprocess.check_output(
        ["git", "-C", str(args.warp_repo), "show", f"{args.baseline}:warp/_src/optim/fsai.py"],
        text=True,
    )
    current_source = (args.warp_repo / "warp/_src/optim/fsai.py").read_text()

    # Stronger than timings alone: assert every computational helper, including
    # the build/refit/application kernels, is AST-identical to the baseline.
    def numerical_ast(text):
        nodes = ast.parse(text).body
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != "FSAI"
        }

    if numerical_ast(source) != numerical_ast(current_source):
        raise AssertionError(
            "Numerical implementation changed; review the difference before benchmarking."
        )
    state = dict(
        baseline=args.baseline,
        candidate_revision=subprocess.check_output(
            ["git", "-C", str(args.warp_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        candidate_source_sha256=hashlib.sha256(current_source.encode()).hexdigest(),
        warp=wp.__version__,
        gpu=wp.get_device("cuda:0").name,
        numerical_helpers_ast_identical=True,
        repeats=args.repeats,
        timing="Synchronized warm wall seconds; alternating baseline/candidate; compilation, input assembly, cyclic garbage collection and checks excluded. Apply amortizes 100 captured matvecs; solve executes 100 CG iterations. Setup includes reuse-plan construction.",
        isolation="Check for other GPU compute processes before/after every paired trial; discard the entire current configuration on overlap. Only completed isolated configurations are checkpointed.",
        results=[],
    )
    if args.resume:
        previous = json.loads(args.output.read_text())
        for key in ("baseline", "candidate_source_sha256", "warp", "gpu", "repeats", "timing"):
            if previous[key] != state[key]:
                raise ValueError(f"Resume metadata differs: {key}")
        state["results"] = previous["results"]
    completed = {(r["device"], r["problem"], r["dtype"], r["width"]) for r in state["results"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsai_baseline_") as tmp:
        path = Path(tmp) / "fsai_baseline.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("fsai_dtype_baseline", path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        classes = {"baseline": baseline.FSAI, "candidate": candidate.FSAI}
        mesh = np.load(args.data / "mesh.npz")
        metadata = json.loads((args.data / "mesh.json").read_text())
        elastic, _, _, _ = assemble(
            mesh["vertices"],
            mesh["tets"],
            mesh["fixed"],
            metadata["young_pa"],
            metadata["poisson"],
            metadata["density_kg_m3"],
        )
        for device, problem in [
            ("cuda:0", "grid_biharmonic"),
            ("cuda:0", "simjeb_ftetwild"),
            ("cpu", "grid_biharmonic"),
        ]:
            for dtype in (wp.float32, wp.float64):
                a = (
                    sp.bsr_copy(elastic, scalar_type=dtype)
                    if problem == "simjeb_ftetwild"
                    else grid(128 if device == "cuda:0" else 16, dtype, device)
                )
                for width, lanes, step in (
                    [(8, 1, 1), (48, 4, 2)] if device == "cuda:0" else [(8, 1, 1)]
                ):
                    if (device, problem, dtype.__name__, width) in completed:
                        continue
                    settings = dict(
                        max_row_size=width,
                        apply_lanes=lanes,
                        max_step_size=step,
                        factor_dtype=wp.float32 if lanes == 4 else dtype,
                        kap_tolerance=0.003,
                        reuse_pattern=True,
                    )
                    b = wp.array(
                        np.random.default_rng(41).normal(size=a.shape[0]),
                        dtype=dtype,
                        device=device,
                    )
                    x = wp.zeros_like(b)
                    updated = sp.bsr_copy(1.01 * a)
                    pres, graphs = {}, {}
                    for label, cls in classes.items():
                        pre = cls(a, **settings)
                        pre.matvec(b, x, x, 1, 0)
                        if device == "cuda:0":
                            with wp.ScopedCapture(device=device) as capture:
                                for _ in range(100):
                                    pre.matvec(b, x, x, 1, 0)
                            graphs[label] = capture.graph
                            wp.capture_launch(capture.graph)
                            wp.synchronize_device(device)
                        pre.update(updated)
                        with wp.ScopedDevice(device):
                            cg(a, b, x, M=pre, maxiter=100, tol=0.0, atol=0.0, check_every=0)
                        pres[label] = pre
                    kernels_equal = []
                    for old, new in zip(
                        baseline._build_kernels(dtype, width, step),
                        candidate._build_kernels(dtype, width, step),
                        strict=True,
                    ):
                        kernels_equal.append(
                            old.module.get_module_hash() == new.module.get_module_hash()
                        )
                    if not all(kernels_equal):
                        raise AssertionError("Construction kernel hashes differ")
                    samples = {
                        label: {metric: [] for metric in ("setup", "refit", "apply", "solve")}
                        for label in classes
                    }
                    for trial in range(args.repeats):
                        gc.collect()
                        exclusive()
                        factors, solutions, applications, initial_factors = {}, {}, {}, {}
                        for label in list(classes) if trial % 2 == 0 else list(reversed(classes)):
                            pre, elapsed = timed(lambda: classes[label](a, **settings), device)
                            samples[label]["setup"].append(elapsed)
                            initial_factors[label] = signature(pre.G)
                            # Use fixed captured buffers for both update and apply timings.
                            pre = pres[label]
                            _, elapsed = timed(lambda: pre.update(updated), device)
                            samples[label]["refit"].append(elapsed)
                            apply = (
                                (lambda: wp.capture_launch(graphs[label]))
                                if device == "cuda:0"
                                else (lambda: pre.matvec(b, x, x, 1, 0))
                            )
                            _, elapsed = timed(apply, device)
                            samples[label]["apply"].append(
                                elapsed / (100 if device == "cuda:0" else 1)
                            )
                            applications[label] = x.numpy().copy()
                            x.zero_()
                            with wp.ScopedDevice(device):
                                _, elapsed = timed(
                                    lambda: cg(
                                        a,
                                        b,
                                        x,
                                        M=pre,
                                        maxiter=100,
                                        tol=0.0,
                                        atol=0.0,
                                        check_every=0,
                                    ),
                                    device,
                                )
                            samples[label]["solve"].append(elapsed)
                            solutions[label] = x.numpy().copy()
                            factors[label] = signature(pre.G)
                        exclusive()
                        if initial_factors["baseline"] != initial_factors["candidate"]:
                            raise AssertionError("Initial factor mismatch")
                        if factors["baseline"] != factors["candidate"]:
                            raise AssertionError("Factor mismatch")
                        np.testing.assert_array_equal(
                            applications["baseline"], applications["candidate"]
                        )
                        np.testing.assert_array_equal(solutions["baseline"], solutions["candidate"])
                        if not np.isfinite(solutions["candidate"]).all():
                            raise AssertionError("Nonfinite solver output")
                    medians = {
                        label: {k: float(np.median(v)) for k, v in metrics.items()}
                        for label, metrics in samples.items()
                    }
                    ratio = {
                        k: medians["candidate"][k] / medians["baseline"][k]
                        for k in medians["baseline"]
                    }
                    record = dict(
                        problem=problem,
                        device=device,
                        dtype=dtype.__name__,
                        shape=a.shape,
                        nnz=a.nnz_sync(),
                        matrix_sha256=signature(a),
                        width=width,
                        lanes=lanes,
                        step=step,
                        storage=settings["factor_dtype"].__name__,
                        construction_kernel_hashes_identical=True,
                        factors_apply_solutions_bitwise_equal=True,
                        samples_s=samples,
                        medians_s=medians,
                        candidate_over_baseline=ratio,
                    )
                    state["results"].append(record)
                    args.output.write_text(json.dumps(state, indent=2) + "\n")
                    print(problem, device, dtype.__name__, width, ratio, flush=True)

    gc.enable()


if __name__ == "__main__":
    main()

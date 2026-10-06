"""Capture original Warp example solves and compare preconditioners on their inputs.

Run with the FSAI Warp checkout's Python. Captures use CLI-default spatial and
physical parameters, the first step, and the first solve at each call site.
APIC advances to the first pressure RHS with norm > 1e-5, up to 60 frames.
Timing replays preserve A, b, x0, solver and tolerance, excluding assembly/JIT.
"""

import argparse
import ast
import gc
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
import warp.sparse as sp
from warp._src.types import type_length
from warp.examples.fem import utils
from warp.optim import linear

EXAMPLES = [
    "diffusion",
    "diffusion_3d",
    "deformed_geometry",
    "convection_diffusion",
    "convection_diffusion_dg",
    "distortion_energy",
    "mixed_elasticity",
    "nonconforming_contact",
    "stokes_transfer",
    "magnetostatics",
    "darcy_ls_optimization",
    "elastic_shape_optimization",
    "cantilever_topology_optimization",
    "stokes",
    "navier_stokes",
    "taylor_green",
    "apic_fluid",
    "apic_fluid_multi_env",
]


def cli_defaults(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        name = ast.literal_eval(node.args[0]).removeprefix("--").replace("-", "_")
        kw = {k.arg: k.value for k in node.keywords}
        if "default" in kw:
            try:
                result[name] = ast.literal_eval(kw["default"])
            except (ValueError, TypeError):
                pass
        elif "action" in kw:
            result[name] = ast.literal_eval(kw["action"]) == "store_false"
    return result


def save_matrix(data, prefix, matrix):
    matrix = sp.bsr_copy(matrix)
    nnz = matrix.nnz_sync()
    data[prefix + "offsets"] = matrix.offsets.numpy()[: matrix.nrow + 1]
    data[prefix + "columns"] = matrix.columns.numpy()[:nnz]
    data[prefix + "values"] = matrix.values.numpy()[:nnz]
    return dict(
        nrow=matrix.nrow,
        ncol=matrix.ncol,
        block_shape=matrix.block_shape,
        dtype=matrix.scalar_type.__name__,
        nnz=nnz,
    )


def load_matrix(data, prefix, meta):
    dtype = getattr(wp, meta["dtype"])
    shape = tuple(meta["block_shape"])
    block = dtype if shape == (1, 1) else wp.types.matrix(shape, dtype)
    rows = np.repeat(np.arange(meta["nrow"], dtype=np.int32), np.diff(data[prefix + "offsets"]))
    return sp.bsr_from_triplets(
        meta["nrow"],
        meta["ncol"],
        wp.array(rows, dtype=int),
        wp.array(data[prefix + "columns"], dtype=int),
        wp.array(
            data[prefix + "values"].reshape((-1,) + (() if shape == (1, 1) else shape)), dtype=block
        ),
    )


def capture(args):
    module = importlib.import_module("warp.examples.fem.example_" + args.example)
    path = Path(module.__file__)
    defaults = cli_defaults(path)
    params = inspect.signature(module.Example).parameters
    settings = {k: defaults[k] for k in params if k in defaults}
    if "quiet" in params:
        settings["quiet"] = True
    if "stage_path" in params:
        settings["stage_path"] = None
    if "domain_radius" in params:
        settings["domain_radius"] = defaults["radius"]
    if "args" in params:
        settings = {"args": SimpleNamespace(**(defaults | {"device": "cuda:0", "headless": True}))}
    # Snapshot outside outer frame capture; solver graph use is retained in replay.
    if "use_cuda_graph" in params:
        settings["use_cuda_graph"] = False
    original = utils.bsr_cg
    seen = set()
    records = []
    frame = 0
    folder = args.data / args.example
    folder.mkdir(parents=True, exist_ok=True)

    def snapshot(A, x, b, options):
        caller = next((f for f in inspect.stack() if Path(f.filename) == path), None)
        site = caller.lineno if caller else 0
        if site in seen:
            return
        if args.example.startswith("apic_fluid") and np.linalg.norm(b.numpy()) <= 1e-5:
            return  # Uniform initial free fall requires no pressure correction.
        seen.add(site)
        data = {"b": b.numpy(), "x0": x.numpy()}
        meta = dict(
            example=args.example,
            frame=frame,
            site=site,
            settings={
                k: vars(v) if isinstance(v, SimpleNamespace) else v for k, v in settings.items()
            },
            method=options.get("method", "cg"),
            tol=options.get("tol", 1e-4),
            maxiter=options.get("max_iters", 0),
            check_every=0,
            rhs_vector_length=type_length(b.dtype),
            matrices={},
        )
        if isinstance(A, sp.BsrMatrix):
            meta["kind"] = "bsr"
            meta["matrices"]["A"] = save_matrix(data, "A", A)
        elif isinstance(A, utils.SaddleSystem):
            meta["kind"] = "saddle"
            for key, value in (("A", A._A), ("B", A._B), ("Bt", A._Bt)):
                meta["matrices"][key] = save_matrix(data, key, value)
        else:
            # Multi-environment APIC exposes its sparse factors through the matvec closure.
            closure = inspect.getclosurevars(A.matvec).nonlocals
            if "divergence_mat" not in closure:
                raise TypeError(f"Unrecognized matrix-free operator: {closure.keys()}")
            meta["kind"] = "matrix_free"
            for key, name in (("B", "divergence_mat"), ("Bt", "transposed_divergence_mat")):
                meta["matrices"][key] = save_matrix(data, key, closure[name])
            data["inverse_diagonal"] = options["M"].numpy()
            data["inv_volume"] = caller.frame.f_locals["inv_volume"].numpy()
            if A.batch_offsets is not None:
                data["batch_offsets"] = A.batch_offsets.numpy()
        stem = str(site)
        np.savez_compressed(folder / (stem + ".npz"), **data)
        (folder / (stem + ".json")).write_text(json.dumps(meta, indent=2) + "\n")
        records.append(meta)
        print(
            "CAPTURED", args.example, site, meta["kind"], b.size * type_length(b.dtype), flush=True
        )

    def wrapped(*a, **kw):
        bound = inspect.signature(original).bind(*a, **kw)
        options = dict(bound.arguments)
        A, x, b = (options.pop(k) for k in ("A", "x", "b"))
        snapshot(A, x, b, options)
        kw["quiet"] = True
        return original(*a, **kw)

    utils.bsr_cg = wrapped
    if args.example == "taylor_green":
        # Constructor's outer capture is bypassed solely to extract the actual system.
        supported = wp.is_conditional_graph_supported
        wp.is_conditional_graph_supported = lambda *a, **kw: False
        original_cg = module.cg_solve

        def wrapped_cg(A, b, x, **kw):
            snapshot(A, x, b, dict(method="cg", tol=kw["tol"], max_iters=kw.get("maxiter", 0)))
            return original_cg(A, b, x, **kw)

        module.cg_solve = wrapped_cg
        try:
            example = module.Example(**settings)
        finally:
            wp.is_conditional_graph_supported = supported
        example.use_cuda_graph = False
    else:
        example = module.Example(**settings)
    if args.example == "cantilever_topology_optimization":
        example.evaluate(example.initial_x)
    else:
        for frame in range(1, 61 if args.example.startswith("apic_fluid") else 2):
            example.step()
            if records:
                break
    if not records:
        raise RuntimeError("No nontrivial pressure solve encountered within 60 frames")
    (folder / "capture.json").write_text(
        json.dumps(
            dict(settings=records[0]["settings"] if records else {}, solves=len(records)), indent=2
        )
        + "\n"
    )


def exclusive():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    others = [int(x) for x in out.splitlines() if x.strip() and int(x) != os.getpid()]
    if others:
        raise RuntimeError(f"GPU occupied by {others}; discard current record and rerun")


def timed(fn):
    wp.synchronize()
    start = time.perf_counter()
    result = fn()
    wp.synchronize()
    return result, time.perf_counter() - start


def measure(args):
    exclusive()
    results = []
    revision = subprocess.check_output(
        ["git", "-C", str(Path(wp.__file__).resolve().parent.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    for path in sorted((args.data / args.example).glob("[0-9]*.json")):
        meta = json.loads(path.read_text())
        meta["residual_units"] = "norm"
        meta["warp_revision"] = revision
        meta["snapshot_sha256"] = hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
        meta["repeats"] = args.repeats
        meta["device"] = wp.get_device().name
        data = np.load(path.with_suffix(".npz"))
        matrices = {k: load_matrix(data, k, m) for k, m in meta["matrices"].items()}
        dtype = getattr(wp, next(iter(meta["matrices"].values()))["dtype"])
        rhs_dtype = (
            dtype
            if meta["rhs_vector_length"] == 1
            else wp.types.vector(meta["rhs_vector_length"], dtype)
        )
        b = wp.array(data["b"], dtype=rhs_dtype)
        x0 = wp.array(data["x0"], dtype=rhs_dtype)
        x = wp.clone(x0)
        kind = meta["kind"]
        if kind == "bsr":
            A = matrices["A"]
        elif kind == "saddle":
            A = utils.SaddleSystem(**matrices, use_diag_precond=False)
        else:
            B, Bt = matrices["B"], matrices["Bt"]
            tmp = wp.empty(Bt.nrow, dtype=wp.types.vector(Bt.block_shape[0], dtype))

            def mv(v, y, z, alpha, beta):
                sp.bsr_mv(Bt, v, tmp)
                if y.ptr != z.ptr and beta != 0:
                    wp.copy(z, y)
                sp.bsr_mv(B, tmp, z, alpha=alpha, beta=beta)

            batches = (
                wp.array(data["batch_offsets"], dtype=int) if "batch_offsets" in data else None
            )
            A = linear.LinearOperator((b.size, b.size), dtype, b.device, mv, batch_offsets=batches)
        # Recompute the true residual in float64 on the GPU, outside all timings.
        if kind == "bsr":
            residual_operator = linear.aslinearoperator(sp.bsr_copy(A, scalar_type=wp.float64))
            delta = sp.bsr_copy(A, scalar_type=wp.float64)
            sp.bsr_axpy(sp.bsr_transposed(delta), delta, alpha=-1.0, beta=1.0)
            meta["relative_asymmetry"] = float(
                np.linalg.norm(delta.values.numpy()[: delta.nnz_sync()])
                / max(np.linalg.norm(A.values.numpy()[: A.nnz_sync()].astype(np.float64)), 1e-300)
            )
        elif kind == "saddle":
            residual_operator = utils.SaddleSystem(
                **{k: sp.bsr_copy(v, scalar_type=wp.float64) for k, v in matrices.items()},
                use_diag_precond=False,
            )
        else:
            residual_operator = linear.aslinearoperator(
                sp.bsr_mm(
                    sp.bsr_copy(B, scalar_type=wp.float64), sp.bsr_copy(Bt, scalar_type=wp.float64)
                )
            )
        double_dtype = (
            wp.float64
            if meta["rhs_vector_length"] == 1
            else wp.types.vector(meta["rhs_vector_length"], wp.float64)
        )
        b64 = wp.array(data["b"], dtype=double_dtype)
        rhs_norm = np.linalg.norm(data["b"].ravel().astype(np.float64))
        residual = wp.zeros_like(b64)
        if kind == "matrix_free":
            inverse_volume = wp.array(data["inv_volume"], dtype=dtype)
        solver = getattr(linear, meta["method"])
        options = dict(tol=meta["tol"], maxiter=meta["maxiter"], check_every=0, use_cuda_graph=True)

        def construct(label):
            factory = (
                linear.preconditioner
                if label == "jacobi"
                else lambda a, _p: linear.FSAI(a, max_row_size=int(label.split("_")[1]))
            )
            if kind == "bsr":
                return factory(A, "diag")
            if kind == "saddle":
                old = utils.preconditioner
                utils.preconditioner = factory
                try:
                    return A._diag_preconditioner()
                finally:
                    utils.preconditioner = old
            if label == "jacobi":
                module = importlib.import_module("warp.examples.fem.example_apic_fluid_multi_env")
                inv_diag = wp.empty_like(b)
                wp.launch(
                    module.schur_inverse_diagonal_kernel,
                    dim=B.nrow,
                    inputs=[B.offsets, B.columns, B.values, inverse_volume, inv_diag],
                )
                return linear.aslinearoperator(inv_diag)
            return factory(sp.bsr_mm(B, Bt), "diag")

        variants = {}
        for label in ["jacobi"] + [f"fsai_{w}" for w in (4, 8, 16, 32)]:
            if label != "jacobi" and meta.get("relative_asymmetry", 0) > 1e-6:
                variants[label] = {
                    "status": "not_applicable",
                    "reason": "Nonsymmetric operator; FSAI requires SPD input.",
                }
                continue
            try:
                pre = construct(label)
                wp.copy(x, x0)
                solver(A, b, x, M=pre, **options)
                variants[label] = {"status": "ok", "samples": []}
            except (ValueError, TypeError) as error:
                if label == "jacobi":
                    raise
                variants[label] = {"status": "unsupported", "reason": str(error)}
        gc.disable()
        try:
            for trial in range(args.repeats):
                gc.collect()
                exclusive()
                labels = [k for k, v in variants.items() if v["status"] == "ok"]
                if trial % 2:
                    labels.reverse()
                for label in labels:
                    pre, setup = timed(lambda: construct(label))
                    wp.copy(x, x0)
                    info, solve = timed(lambda: solver(A, b, x, M=pre, **options))
                    iterations, error, tolerance = [
                        v.numpy().tolist() if isinstance(v, wp.array) else v for v in info
                    ]
                    if isinstance(info[0], wp.array):
                        # check_every=0 returns squared residual/tolerance arrays.
                        error = np.sqrt(error).tolist()
                        tolerance = np.sqrt(tolerance).tolist()
                    residual_operator.matvec(
                        wp.array(x.numpy(), dtype=double_dtype), b64, residual, -1.0, 1.0
                    )
                    r = residual.numpy().ravel().astype(np.float64)
                    offsets = data["batch_offsets"] if "batch_offsets" in data else [0, r.size]
                    residual_norms = [
                        float(np.linalg.norm(r[start:stop]))
                        for start, stop in zip(offsets[:-1], offsets[1:])
                    ]
                    targets = np.asarray(tolerance).ravel()
                    variants[label]["samples"].append(
                        dict(
                            setup_s=setup,
                            solve_s=solve,
                            total_s=setup + solve,
                            iterations=iterations,
                            reported_error=error,
                            tolerance=tolerance,
                            reported_converged=bool(
                                np.all(np.asarray(error) <= np.asarray(tolerance))
                            ),
                            true_converged=bool(np.all(np.asarray(residual_norms) <= targets)),
                            true_residual_norms=residual_norms,
                            true_relative_residual=float(np.linalg.norm(r) / max(rhs_norm, 1e-300)),
                            finite=bool(np.isfinite(x.numpy()).all()),
                        )
                    )
                exclusive()
        finally:
            gc.enable()
        for label, record in variants.items():
            if record["status"] == "ok":
                record["median_s"] = {
                    k: float(np.median([s[k] for s in record["samples"]]))
                    for k in ("setup_s", "solve_s", "total_s")
                }
        results.append(dict(**meta, variants=variants))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            "MEASURED",
            args.example,
            meta["site"],
            {k: v.get("median_s", v["status"]) for k, v in variants.items()},
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "measure"))
    parser.add_argument("example", choices=[*EXAMPLES, "all"])
    parser.add_argument("--data", type=Path, default=Path("data/warp-examples"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.example == "all":
        for name in EXAMPLES:
            if args.phase == "measure":
                while subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip():
                    print(f"Waiting for idle GPU before {name}", flush=True)
                    time.sleep(15)
            command = [
                sys.executable,
                __file__,
                args.phase,
                name,
                "--data",
                str(args.data),
                "--repeats",
                str(args.repeats),
            ]
            if args.output:
                command += ["--output", str(args.output / (name + ".json"))]
            subprocess.run(command, check=True)
        return
    if args.output is None:
        args.output = Path("results/warp-examples") / (args.example + ".json")
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    with wp.ScopedDevice("cuda:0"):
        if args.phase == "capture":
            capture(args)
        else:
            measure(args)


if __name__ == "__main__":
    main()

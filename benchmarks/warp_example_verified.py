"""Compare preconditioners at a verified residual target using iterative refinement.

The original matrix and inner solver dtype are retained. Residuals and the
accumulated solution use float64 to avoid accepting drifted recursive residuals
or imposing an unattainable float32 solution-storage floor. Both preconditioners
use the same correction policy. All numerical work runs in Warp on CUDA.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from warp.fem.linalg import array_axpy
from warp.optim import linear
from warp_example_fsai import exclusive, load_matrix, timed


class VerifiedSolve:
    def __init__(self, A, b, x0, method, preconditioner, target, inner_max, outer_max):
        self.A = A
        self.A64 = sp.bsr_copy(A, scalar_type=wp.float64)
        self.b64 = wp.empty(
            b.shape,
            dtype=wp.float64
            if b.ndim == 1 and b.dtype == A.scalar_type
            else wp.types.vector(A.block_shape[0], wp.float64),
            device=b.device,
        )
        self.x0 = wp.empty_like(self.b64)
        wp.utils.array_cast(b, self.b64)
        wp.utils.array_cast(x0, self.x0)
        self.x = wp.empty_like(self.b64)
        self.residual = wp.empty_like(self.b64)
        self.correction64 = wp.empty_like(self.b64)
        self.rhs = wp.empty_like(b)
        self.correction = wp.empty_like(b)
        self.norm_squared = wp.empty(1, dtype=wp.float64)
        self.M = (
            linear.preconditioner(A, "diag")
            if preconditioner == "jacobi"
            else linear.FSAI(A, max_row_size=int(preconditioner.split("_")[1]))
        )
        self.solver = getattr(linear, method)
        self.target = target
        self.inner_max = inner_max
        self.outer_max = outer_max

    def norm(self):
        wp.copy(self.residual, self.b64)
        sp.bsr_mv(self.A64, self.x, self.residual, alpha=-1.0, beta=1.0)
        wp.utils.array_inner(self.residual, self.residual, out=self.norm_squared)
        return math.sqrt(float(self.norm_squared.numpy()[0]))

    def solve(self):
        wp.copy(self.x, self.x0)
        residual_norm = self.norm()
        history = [residual_norm]
        iterations = []
        for _ in range(self.outer_max):
            if not math.isfinite(residual_norm) or residual_norm <= self.target:
                break
            wp.utils.array_cast(self.residual, self.rhs)
            self.correction.zero_()
            info = self.solver(
                self.A,
                self.rhs,
                self.correction,
                M=self.M,
                tol=0.05,
                atol=0.0,
                maxiter=self.inner_max,
                check_every=0,
                use_cuda_graph=True,
            )
            iterations.append(int(info[0].numpy()[0]))
            wp.utils.array_cast(self.correction, self.correction64)
            array_axpy(self.correction64, self.x)
            residual_norm = self.norm()
            history.append(residual_norm)
        return dict(
            converged=math.isfinite(residual_norm) and residual_norm <= self.target,
            true_residual_norm=residual_norm,
            target=self.target,
            inner_iterations=iterations,
            total_iterations=sum(iterations),
            corrections=len(iterations),
            residual_history=history,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example")
    parser.add_argument("--data", type=Path, default=Path("data/warp-examples"))
    parser.add_argument("--original", type=Path, default=Path("results/warp-examples"))
    parser.add_argument("--output", type=Path, default=Path("results/warp-examples-verified"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-max", type=int, default=1000)
    parser.add_argument("--outer-max", type=int, default=20)
    args = parser.parse_args()
    exclusive()
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    records = []
    with wp.ScopedDevice("cuda:0"):
        for original in json.loads((args.original / (args.example + ".json")).read_text()):
            assert original["kind"] == "bsr"
            path = args.data / args.example / (str(original["site"]) + ".npz")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == original["snapshot_sha256"]
            data = np.load(path)
            A = load_matrix(data, "A", original["matrices"]["A"])
            rhs_type = (
                A.scalar_type
                if original["rhs_vector_length"] == 1
                else wp.types.vector(original["rhs_vector_length"], A.scalar_type)
            )
            b = wp.array(data["b"], dtype=rhs_type)
            x0 = wp.array(data["x0"], dtype=rhs_type)
            target = original["variants"]["jacobi"]["samples"][0]["tolerance"][0]

            def build(label):
                return VerifiedSolve(
                    A, b, x0, original["method"], label, target, args.inner_max, args.outer_max
                )

            variants = {}
            for label, old in original["variants"].items():
                if old["status"] != "ok":
                    variants[label] = dict(status=old["status"], reason=old["reason"])
                    continue
                instance = build(label)
                info = instance.solve()
                variants[label] = dict(status="measured", samples=[])
                print("WARM", args.example, original["site"], label, info, flush=True)

            gc.disable()
            try:
                for repeat in range(args.repeats):
                    gc.collect()
                    exclusive()
                    labels = [
                        label
                        for label, record in variants.items()
                        if record["status"] == "measured"
                    ]
                    if repeat % 2:
                        labels.reverse()
                    for label in labels:
                        instance, setup = timed(lambda: build(label))
                        info, solve = timed(instance.solve)
                        info.update(
                            setup_s=setup,
                            solve_s=solve,
                            total_s=setup + solve,
                            finite=bool(np.isfinite(instance.x.numpy()).all()),
                        )
                        variants[label]["samples"].append(info)
                    exclusive()
            finally:
                gc.enable()
            for record in variants.values():
                if record["status"] == "measured":
                    record["converged"] = all(s["converged"] for s in record["samples"])
                    record["median_s"] = {
                        key: float(np.median([s[key] for s in record["samples"]]))
                        for key in ("setup_s", "solve_s", "total_s")
                    }
            records.append(
                dict(
                    example=args.example,
                    site=original["site"],
                    warp_revision=original["warp_revision"],
                    snapshot_sha256=original["snapshot_sha256"],
                    unknowns=original["unknowns"],
                    method=original["method"],
                    matrix_dtype=A.scalar_type.__name__,
                    residual_and_solution_dtype="float64",
                    target=target,
                    inner_tol=0.05,
                    inner_atol=0.0,
                    inner_max=args.inner_max,
                    outer_max=args.outer_max,
                    repeats=args.repeats,
                    variants=variants,
                )
            )
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / (args.example + ".json")).write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(
                "DONE",
                args.example,
                original["site"],
                {
                    k: (v["converged"], v["median_s"]["total_s"])
                    for k, v in variants.items()
                    if v["status"] == "measured"
                },
                flush=True,
            )


if __name__ == "__main__":
    main()

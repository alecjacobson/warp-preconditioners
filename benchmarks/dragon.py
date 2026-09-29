"""Run against the unmodified C++ benchmark's MatrixMarket dumps.

SciPy is used ONLY for file I/O and independent residual verification, never
for preconditioner setup or solve. All three RHS columns are measured.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sp

from warp_preconditioners import FSAI, BiharmonicSystem


def upload(a, device):
    a = a.tocoo()
    return sp.bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(a.row, dtype=int, device=device),
        wp.array(a.col, dtype=int, device=device),
        wp.array(a.data, dtype=wp.float64, device=device),
    )


def berr(a, abs_a, b, x):
    return float(
        np.max(np.abs(b - a @ x) / np.maximum(abs_a @ np.abs(x) + np.abs(b), np.finfo(float).tiny))
    )


def norm(x):
    # Avoid starting a BLAS thread team between timed GPU solves.
    return float(np.sqrt(np.sum(np.square(x))))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dir", type=Path, default=Path("/tmp/dump"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--k", type=int, choices=[1, 2, 3], default=2)
    p.add_argument(
        "--methods", nargs="+", default=["jacobi-cg", "jacobi-cr", "fsai-cg", "lifted-gmres"]
    )
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--kap-tolerance", type=float, default=1e-3)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--target", type=float, default=1e-8)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--schedule", choices=["fixed", "upstream"], default="upstream")
    p.add_argument("--rhs-count", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.seconds <= 0 or args.chunk <= 0 or args.target <= 0 or not 1 <= args.rhs_count <= 3:
        p.error("seconds, chunk and target must be positive; rhs-count must be 1..3")
    if args.k != 2 and any(name.startswith("lifted") for name in args.methods):
        p.error("lifted methods require --k 2; select jacobi/fsai methods for other orders")
    wp.init()
    wp.set_device(args.device)
    wp.config.log_level = wp.LOG_WARNING
    Q = sio.mmread(args.dir / f"k{args.k}_Q.mtx").tocsr()
    rhs = np.asarray(sio.mmread(args.dir / f"k{args.k}_rhs.mtx"))
    print("Loaded", Q.shape, "nnz", Q.nnz, flush=True)
    abs_Q = abs(Q)
    A = upload(Q, args.device)
    Kw, mw = None, None
    if args.k == 2:
        H = sio.mmread(args.dir / "k1_Q.mtx").tocsr()
        mixed = sio.mmread(args.dir / "k4_Q.mtx").tocsr()
        mass = mixed.diagonal()[: Q.shape[0]].copy()
        K = -mixed[: Q.shape[0], Q.shape[0] :]
        assert abs(H - (ss.diags(mass) + K)).max() < 1e-10
        expected = ss.diags(mass) + K @ ss.diags(1 / mass) @ K
        assert abs(Q - expected).max() <= 1e-12 * abs(Q).max()
        Kw, mw = (
            upload(K, args.device),
            wp.array(mass, dtype=wp.float64, device=args.device),
        )
    records = []
    for name in args.methods:
        lifted = name.startswith("lifted")
        fn = getattr(linear, name.split("-")[-1])

        def setup(lifted=lifted, name=name):
            if lifted:
                system = BiharmonicSystem(Kw, mw)
                if "jacobi" in name:
                    pre = linear.preconditioner(system.matrix, "diag")
                else:
                    pre = system.preconditioner(
                        max_row_size=args.width, kap_tolerance=args.kap_tolerance
                    )
                return system.matrix, pre, system
            if name.startswith("fsai"):
                return (
                    A,
                    FSAI(A, max_row_size=args.width, kap_tolerance=args.kap_tolerance),
                    None,
                )
            return A, linear.preconditioner(A, "diag"), None

        # Warm all setup/apply/solver specializations, excluding first-use JIT.
        mat, pre, system = setup()
        b0 = wp.array(rhs[:, 0].copy(), dtype=wp.float64, device=args.device)
        b = system.rhs(b0) if lifted else b0
        x = wp.zeros_like(b)
        fn(mat, b, x, M=pre, tol=1e-14, maxiter=10, check_every=0)
        wp.synchronize()
        start = time.perf_counter()
        mat, pre, system = setup()
        wp.synchronize()
        setup_s = time.perf_counter() - start
        print(name, "setup", setup_s, flush=True)
        factor = pre.factor if lifted and hasattr(pre, "factor") else pre
        factor_nnz = factor.G.nnz_sync() if isinstance(factor, FSAI) else None
        cols = []
        for c in range(min(args.rhs_count, rhs.shape[1])):
            b0 = wp.array(rhs[:, c].copy(), dtype=wp.float64, device=args.device)
            b = system.rhs(b0) if lifted else b0
            x = wp.zeros_like(b)
            out = wp.empty_like(b0)
            wp.synchronize()
            start = time.perf_counter()
            iterations = 0
            error = 1.0
            chunk = 10 if args.schedule == "upstream" else args.chunk
            best_error = float("inf")
            stalled = 0
            while time.perf_counter() - start < args.seconds and error > args.target:
                chunk_start = time.perf_counter()
                nit, _res, _atol = fn(mat, b, x, M=pre, tol=1e-14, maxiter=chunk, check_every=0)
                wp.synchronize()
                chunk_dt = time.perf_counter() - chunk_start
                iterations += int(nit.numpy()[0]) if isinstance(nit, wp.array) else int(nit)
                sol = system.solution(x, out).numpy() if lifted else x.numpy()
                error = berr(Q, abs_Q, rhs[:, c], sol)
                print(
                    name,
                    c,
                    iterations,
                    "berr",
                    error,
                    "s",
                    time.perf_counter() - start,
                    flush=True,
                )
                if not np.isfinite(error):
                    break
                if error < best_error * 0.99:
                    best_error, stalled = error, 0
                else:
                    stalled += 1
                if iterations >= 1000000 or stalled >= 5:
                    break
                if args.schedule == "upstream":
                    remaining_s = args.seconds - (time.perf_counter() - start)
                    chunk = max(
                        10,
                        min(
                            1000000 - iterations,
                            int(min(5.0, remaining_s) * chunk / max(chunk_dt, 1e-6)),
                            5000,
                        ),
                    )
            elapsed = time.perf_counter() - start
            # Normwise residual additionally exposes the metric's limitations.
            relres = float(norm(rhs[:, c] - Q @ sol) / norm(rhs[:, c]))
            cols.append(
                dict(
                    solve_s=elapsed,
                    iterations=iterations,
                    berr=error,
                    relative_residual=relres,
                    converged=bool(error <= args.target),
                )
            )
        record = dict(
            method=name,
            width=args.width,
            kap_tolerance=args.kap_tolerance,
            setup_s=setup_s,
            factor_nnz=factor_nnz,
            truncated_rows=getattr(factor, "truncated_rows", None),
            solve_s=sum(x["solve_s"] for x in cols),
            berr=max(x["berr"] for x in cols),
            converged=all(x["converged"] for x in cols),
            columns=cols,
        )
        records.append(record)
        record = clean(record)
        records[-1] = record
        print(json.dumps(record), flush=True)
        if args.output:
            args.output.parent.mkdir(exist_ok=True, parents=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        warp=wp.__version__,
                        device=str(wp.get_device(args.device)),
                        gpu=wp.get_device(args.device).name,
                        k=args.k,
                        n=Q.shape[0],
                        nnz=Q.nnz,
                        target=args.target,
                        chunk=args.chunk,
                        schedule=args.schedule,
                        seconds_per_rhs=args.seconds,
                        results=records,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

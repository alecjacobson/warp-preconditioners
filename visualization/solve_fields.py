"""Recover actual Warp fields for the published dragon comparison.

No analytical replacement, mesh smoothing, or field filtering is used.
The z-coordinate RHS is column 2 of the original C++ benchmark dumps.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import igl
import numpy as np
import scipy.io as sio
import warp as wp
from warp.optim import linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from dragon import berr, norm, upload

from warp_preconditioners import FSAI


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dir", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/visualization"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    vertices, faces = igl.read_triangle_mesh(str(args.mesh))
    a = sio.mmread(args.dir / "k2_Q.mtx").tocsr()
    rhs = np.asarray(sio.mmread(args.dir / "k2_rhs.mtx"))
    n = len(vertices)
    mixed = sio.mmread(args.dir / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n]
    np.testing.assert_allclose(rhs[:, 2], mass * vertices[:, 2], rtol=1e-13, atol=1e-12)
    b_np = rhs[:, 2].copy()
    A, b = upload(a, "cuda:0"), wp.array(b_np, dtype=wp.float64, device="cuda:0")
    abs_a = abs(a)
    fields = {}
    records = {}
    # Compare the strongest non-FSAI baseline and the best total-time FSAI
    # configuration from the existing benchmark. Fixed budgets reproduce the
    # z-column's original run independently of current machine load.
    for name, solver, make, budget in [
        ("jacobi_cr", linear.cr, lambda: linear.preconditioner(A, "diag"), 118500),
        ("fsai_cg", linear.cg, lambda: FSAI(A, max_row_size=8), 500),
    ]:
        pre = make()
        x = wp.zeros_like(b)
        solver(A, b, x, M=pre, tol=1e-14, maxiter=10, check_every=0)
        wp.synchronize()
        start = time.perf_counter()
        pre = make()
        wp.synchronize()
        setup_s = time.perf_counter() - start
        x.zero_()
        wp.synchronize()
        start = time.perf_counter()
        best_error = float("inf")
        chosen_iterations = 0
        chosen_seconds = 0.0
        history = []
        for iterations in range(500, budget + 1, 500):
            solver(A, b, x, M=pre, tol=1e-14, maxiter=500, check_every=0)
            sol = x.numpy()
            error = berr(a, abs_a, b_np, sol)
            elapsed = time.perf_counter() - start
            history.append([iterations, elapsed, error])
            if error < best_error:
                best_error = error
                chosen_iterations = iterations
                chosen_seconds = elapsed
                fields[name] = sol.copy()
            if iterations % 10000 == 0 or iterations == budget:
                print(name, iterations, elapsed, error, flush=True)
        records[name] = dict(
            solver="warp.optim.linear." + solver.__name__,
            preconditioner="Jacobi" if name == "jacobi_cr" else "FSAI, max_row_size=8",
            setup_s=setup_s,
            solve_s=chosen_seconds,
            selected_iterations=chosen_iterations,
            budget_iterations=budget,
            total_run_s=time.perf_counter() - start,
            selection="lowest independently measured backward error among 500-iteration checkpoints",
            backward_error=best_error,
            relative_residual=norm(b_np - a @ fields[name]) / norm(b_np),
            history=history,
        )
    lo = float(min(v.min() for v in fields.values()))
    hi = float(max(v.max() for v in fields.values()))
    np.savez_compressed(args.output / "fields.npz", vertices=vertices, faces=faces, **fields)
    meta = dict(
        mesh=args.mesh.name,
        mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        warp=wp.__version__,
        gpu=wp.get_device().name,
        vertices=n,
        faces=len(faces),
        system="(M + K M^-1 K) u = M z",
        rhs_column=2,
        scalar_range=[lo, hi],
        field_units="original mesh coordinate units",
        display="original mesh, no geometry deformation or postprocessing of either scalar field",
        difference=dict(
            max_abs=float(np.max(np.abs(fields["jacobi_cr"] - fields["fsai_cg"]))),
            relative_l2=norm(fields["jacobi_cr"] - fields["fsai_cg"]) / norm(fields["fsai_cg"]),
        ),
        results=records,
    )
    (args.output / "solve.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()

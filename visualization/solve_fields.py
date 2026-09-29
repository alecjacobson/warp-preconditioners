"""Compute fixed-budget Warp iterates of a screened biharmonic height field.

Every displayed field is an unfiltered solver output from a zero initial guess.
Each budget uses one uninterrupted Krylov solve, without checkpoint restarts.
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
import scipy.sparse as ss
import warp as wp
from warp.optim import linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from dragon import berr, norm, upload

from warp_preconditioners import FSAI


def field_statistics(u, vertices, mass):
    """Area-weighted diagnostics; the fitted affine field is NEVER rendered."""
    z = vertices[:, 2]
    design = np.column_stack([np.ones(len(u)), vertices])
    coeff = np.linalg.solve(design.T @ (mass[:, None] * design), design.T @ (mass * u))
    mean = np.average(u, weights=mass)

    def rms(x):
        return float(np.sqrt(np.average(x * x, weights=mass)))

    variation = rms(u - mean)
    return dict(
        range=[float(u.min()), float(u.max())],
        mass_weighted_mean=float(mean),
        mass_weighted_std=variation,
        height_change_rms=rms(u - z),
        affine_fit_rms=rms(u - design @ coeff),
        nonaffine_fraction=rms(u - design @ coeff) / max(variation, np.finfo(float).tiny),
    )


def run(a, b_np, method, budget):
    A = upload(a, "cuda:0")
    b = wp.array(b_np, dtype=wp.float64, device="cuda:0")
    solver = linear.cg if method == "fsai_cg" else linear.cr

    def make():
        return FSAI(A, max_row_size=8) if method == "fsai_cg" else linear.preconditioner(A, "diag")

    pre = make()
    x = wp.zeros_like(b)
    solver(A, b, x, M=pre, tol=0.0, atol=0.0, maxiter=10, check_every=0)
    wp.synchronize()
    start = time.perf_counter()
    pre = make()
    wp.synchronize()
    setup_s = time.perf_counter() - start
    x.zero_()
    wp.synchronize()
    start = time.perf_counter()
    nit, _, _ = solver(A, b, x, M=pre, tol=0.0, atol=0.0, maxiter=budget, check_every=0)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    actual = int(nit.numpy()[0]) if isinstance(nit, wp.array) else int(nit)
    sol = x.numpy()
    assert np.isfinite(sol).all()
    return sol, dict(
        solver="warp.optim.linear." + solver.__name__,
        preconditioner="FSAI, max_row_size=8" if method == "fsai_cg" else "Jacobi",
        setup_s=setup_s,
        solve_s=elapsed,
        requested_iterations=budget,
        actual_iterations=actual,
        initial_guess="zero",
        restarts=0,
        selection="final iterate of a single solver call; no best-checkpoint selection",
        tol=0.0,
        atol=0.0,
        backward_error=berr(a, abs(a), b_np, sol),
        relative_residual=norm(b_np - a @ sol) / norm(b_np),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dir", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/visualization"))
    parser.add_argument("--data-weight", type=float, default=0.001)
    parser.add_argument("--audit-weights", type=float, nargs="+")
    args = parser.parse_args()
    if args.data_weight <= 0 or (args.audit_weights and min(args.audit_weights) <= 0):
        parser.error("data weights must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    vertices, faces = igl.read_triangle_mesh(str(args.mesh))
    original_q = sio.mmread(args.dir / "k2_Q.mtx").tocsr()
    rhs = np.asarray(sio.mmread(args.dir / "k2_rhs.mtx"))
    n = len(vertices)
    mixed = sio.mmread(args.dir / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n]
    K = -mixed[:n, n:]
    assert mass.min() > 0
    np.testing.assert_allclose(rhs, mass[:, None] * vertices, rtol=1e-13, atol=1e-12)
    # Independently validate the dumped operator and its vertex ordering.
    cot_error = float(abs(K + igl.cotmatrix(vertices, faces)).max())
    # Very thin triangles amplify roundoff differences between libigl builds.
    cot_relative_error = cot_error / float(abs(K).max())
    assert cot_relative_error < 1e-7
    expected = ss.diags(mass) + K @ ss.diags(1 / mass) @ K
    assembly_error = float(abs(original_q - expected).max()) / float(abs(original_q).max())
    assert assembly_error < 1e-12
    del mixed, expected

    def equation(weight):
        return original_q + ss.diags((weight - 1) * mass), weight * rhs[:, 2]

    if args.audit_weights:
        sweep = []
        for weight in args.audit_weights:
            a, b_np = equation(weight)
            for budget in [500, 5000]:
                u, record = run(a, b_np, "fsai_cg", budget)
                record.update(data_weight=weight, **field_statistics(u, vertices, mass))
                sweep.append(record)
                print(json.dumps(record), flush=True)
        (args.output / "weight_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
        return

    a, b_np = equation(args.data_weight)
    fields, records = {}, {}
    order = ["fsai_cg_500", "jacobi_cr_500", "jacobi_cr_5000", "jacobi_cr_50000"]
    for name in order:
        method, budget = name.rsplit("_", 1)
        u, record = run(a, b_np, method, int(budget))
        assert record["actual_iterations"] == int(budget), record
        record.update(field_statistics(u, vertices, mass))
        record["field_sha256"] = hashlib.sha256(u.tobytes()).hexdigest()
        record["data_energy"] = float(
            0.5 * args.data_weight * np.sum(mass * (u - vertices[:, 2]) ** 2)
        )
        record["bending_energy"] = float(0.5 * np.sum((K @ u) ** 2 / mass))
        fields[name], records[name] = u, record
        print(name, json.dumps(record), flush=True)
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
        system="(alpha M + K M^-1 K) u = alpha M z",
        data_weight=args.data_weight,
        smoothing_weight=1.0,
        rhs_column=2,
        display_order=order,
        scalar_range=[lo, hi],
        field_units="original mesh coordinate units",
        display="original geometry colored by u, no scalar filtering or geometry deformation",
        verification=dict(
            rhs="all three original RHS columns equal M times the original vertex coordinates",
            stiffness="dumped K agrees with independently assembled -igl.cotmatrix(V,F)",
            stiffness_max_abs_difference=cot_error,
            stiffness_relative_max_difference=cot_relative_error,
            original_operator_relative_max_difference=assembly_error,
            gpu_field="exact downloaded float64 iterate; hash recorded per field",
        ),
        results=records,
    )
    sweep = args.output / "weight_sweep.json"
    if sweep.exists():
        meta["weight_sweep"] = json.loads(sweep.read_text())
    (args.output / "solve.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

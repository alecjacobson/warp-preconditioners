"""Compare pure-Warp block preconditioners on the actual mixed dragon systems.

NumPy/SciPy perform input I/O and independent verification only. No CPU
factorization, inner SciPy solve, or multigrid is used.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import warp as wp
from dragon import berr, clean, norm, upload
from warp.optim import linear

from warp_preconditioners.block_ilu import BlockILU0, BlockJacobi
from warp_preconditioners.mixed import MatchingSchur, MixedHarmonicSystem, ShiftedBlock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualization"))
from solve_fields import data_function


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dump", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--data", type=Path, default=Path("data/visualization"))
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 1e-4])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["jacobi", "block-jacobi", "ilu4", "ilu8", "schur8", "schur48"],
    )
    parser.add_argument(
        "--rhs", choices=["coordinates", "nonlinear", "manufactured"], default="coordinates"
    )
    parser.add_argument("--rhs-count", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=620)
    parser.add_argument("--restart", type=int, default=31)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("results/indefinite-probe.json"))
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.maxiter < 1
        or args.restart < 1
        or not 1 <= args.rhs_count <= 3
        or args.rtol <= 0
    ):
        parser.error("positive budgets/tolerance/repeats and rhs-count in 1..3 required")
    d = np.load(args.data / "fields.npz")
    n = len(d["vertices"])
    mixed = sio.mmread(args.dump / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n]
    K = -mixed[:n, n:]
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    Kw, mw = upload(K, "cuda:0"), wp.array(mass, dtype=wp.float64)
    records = []
    report = dict(
        warp=wp.__version__,
        gpu=wp.get_device().name,
        vertices=n,
        protocol="Zero initial guesses; native right GMRES with fixed restart; no extra Krylov restarts; same matrix and RHS per case",
        precision="float64; FSAI factors stored in float32",
        preconditioner_parameters=dict(
            fsai_kap=0.003,
            fsai_apply_lanes=4,
            block_ilu_factor_sweeps=16,
            block_ilu_relaxation=0.8,
            shifted_ilu_factor_sweeps=20,
            shifted_ilu_relaxation=1.0,
        ),
        timing="Warmed operator construction + preconditioner setup + native GMRES; uploads and independent CPU verification excluded",
        validation="Original-system relative residual < 1e-8 AND componentwise backward error < 1e-8; manufactured forward error additionally < 1e-6",
        maxiter=args.maxiter,
        restart=args.restart,
        rtol=args.rtol,
        rhs=args.rhs,
        equilibrate=args.equilibrate,
        repeats=args.repeats,
        results=records,
    )
    for order in args.orders:
        original = sio.mmread(args.dump / f"k{order + 2}_Q.mtx").tocsr()
        original_rhs = np.asarray(sio.mmread(args.dump / f"k{order + 2}_rhs.mtx"))
        np.testing.assert_allclose(
            original_rhs[:n], mass[:, None] * d["vertices"], rtol=1e-13, atol=1e-12
        )
        assert np.count_nonzero(original_rhs[n:]) == 0
        expected = (
            ss.bmat([[ss.diags(mass), -K], [-K, -ss.diags(mass)]], format="csr")
            if order == 2
            else ss.bmat(
                [
                    [ss.diags(mass), None, -K],
                    [None, -K, -ss.diags(mass)],
                    [-K, -ss.diags(mass), None],
                ],
                format="csr",
            )
        )
        assert abs(original - expected).max() == 0
        for weight in args.weights:
            diagonal = np.zeros(order * n)
            diagonal[:n] = (weight - 1) * mass
            cpu = original + ss.diags(diagonal)
            system = MixedHarmonicSystem(
                Kw, mw, order=order, data_weight=weight, equilibrate=args.equilibrate
            )
            wp.synchronize()
            start = time.perf_counter()
            system = MixedHarmonicSystem(
                Kw, mw, order=order, data_weight=weight, equilibrate=args.equilibrate
            )
            wp.synchronize()
            operator_setup_s = time.perf_counter() - start
            A = system.matrix
            targets = (
                d["vertices"][:, : args.rhs_count]
                if args.rhs == "coordinates"
                else data_function(d["vertices"])[0][:, None]
            )
            truth = None
            if args.rhs == "manufactured":
                f = targets[:, 0]
                truth = np.column_stack([np.sin(f + (j * 0.3)) for j in range(order)])
                full_rhs = np.asarray(cpu @ truth.T.ravel()).reshape(order, n).T
                rhs = [full_rhs]
            else:
                rhs = [
                    np.column_stack([weight * mass * f] + [np.zeros(n)] * (order - 1))
                    for f in targets.T
                ]
            for trial, method in [
                (trial, method)
                for trial in range(args.repeats)
                for method in (args.methods if trial % 2 == 0 else args.methods[::-1])
            ]:

                def make():
                    if method == "none":
                        return None
                    if method == "jacobi":
                        return linear.preconditioner(A, "diag")
                    if method == "block-jacobi":
                        return BlockJacobi(A)
                    if method.startswith("ilu"):
                        return BlockILU0(
                            A, factor_sweeps=16, solve_sweeps=int(method[3:]), relaxation=0.8
                        )
                    if method.startswith("shift-ilu"):
                        return ShiftedBlock(
                            system,
                            H_inverse=BlockILU0(
                                system.H,
                                factor_sweeps=20,
                                solve_sweeps=int(method[9:]),
                                relaxation=1.0,
                            ),
                        )
                    if method.startswith("shift"):
                        return ShiftedBlock(
                            system,
                            max_row_size=int(method[5:]),
                            kap_tolerance=0.003,
                            apply_lanes=4,
                            factor_dtype=wp.float32,
                        )
                    if method.startswith("schur"):
                        return MatchingSchur(
                            system,
                            max_row_size=int(method[5:]),
                            kap_tolerance=0.003,
                            apply_lanes=4,
                            factor_dtype=wp.float32,
                        )
                    raise ValueError(method)

                print("START", order, weight, method, flush=True)
                try:
                    pre = make()
                    b = system.transform(wp.array(rhs[0], dtype=system.vector_type))
                    x = wp.zeros_like(b)
                    linear.gmres(
                        A,
                        b,
                        x,
                        M=pre,
                        maxiter=args.restart,
                        restart=args.restart,
                        tol=0.0,
                        atol=0.0,
                        check_every=args.restart,
                    )
                    wp.synchronize()
                    start = time.perf_counter()
                    pre = make()
                    wp.synchronize()
                    setup = time.perf_counter() - start
                    columns = []
                    for c, b_np in enumerate(rhs):
                        b = system.transform(wp.array(b_np, dtype=system.vector_type))
                        x = wp.zeros_like(b)
                        wp.synchronize()
                        start = time.perf_counter()
                        nit, res, _ = linear.gmres(
                            A,
                            b,
                            x,
                            M=pre,
                            maxiter=args.maxiter,
                            restart=args.restart,
                            tol=args.rtol,
                            atol=0.0,
                            check_every=args.restart,
                        )
                        wp.synchronize()
                        elapsed = time.perf_counter() - start
                        u = system.transform(x).numpy()
                        flat = u.T.ravel()
                        bflat = b_np.T.ravel()
                        residual = bflat - cpu @ flat
                        col = dict(
                            column=c,
                            iterations=int(nit.numpy()[0])
                            if isinstance(nit, wp.array)
                            else int(nit),
                            solve_s=elapsed,
                            solver_relative_residual=(
                                float(np.sqrt(res.numpy()[0]))
                                if isinstance(res, wp.array)
                                else float(res)
                            )
                            / norm(b.numpy()),
                            relative_residual=norm(residual) / norm(bflat),
                            backward_error=berr(cpu, abs(cpu), bflat, flat),
                            block_relative_residuals=[
                                norm(v) / norm(bflat) for v in residual.reshape(order, n)
                            ],
                        )
                        col["validated"] = bool(
                            np.isfinite(u).all()
                            and col["relative_residual"] < 1e-8
                            and col["backward_error"] < 1e-8
                        )
                        if truth is not None:
                            col["relative_forward_error"] = norm(u - truth) / norm(truth)
                            col["validated"] = (
                                col["validated"] and col["relative_forward_error"] < 1e-6
                            )
                        reference_path = (
                            args.data.parent / "visualization-tuned" / "refined_reference.npy"
                        )
                        if (
                            args.rhs == "nonlinear"
                            and order == 2
                            and weight == 1e-4
                            and reference_path.exists()
                        ):
                            reference = np.load(reference_path)
                            col["u_mass_relative_error_to_smoothing_reference"] = float(
                                np.sqrt(
                                    np.sum(mass * (u[:, 0] - reference) ** 2)
                                    / np.sum(mass * reference**2)
                                )
                            )
                            col["validated"] = (
                                col["validated"]
                                and col["u_mass_relative_error_to_smoothing_reference"] < 1e-6
                            )
                        columns.append(col)
                    record = dict(
                        trial=trial,
                        order=order,
                        data_weight=weight,
                        method=method,
                        setup_s=setup,
                        operator_setup_s=operator_setup_s,
                        columns=columns,
                        total_s=operator_setup_s + setup + sum(c["solve_s"] for c in columns),
                        validated=all(c["validated"] for c in columns),
                    )
                except (ValueError, RuntimeError) as error:
                    record = dict(
                        trial=trial,
                        order=order,
                        data_weight=weight,
                        method=method,
                        error=str(error),
                        validated=False,
                    )
                records.append(clean(record))
                print(json.dumps(records[-1]), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

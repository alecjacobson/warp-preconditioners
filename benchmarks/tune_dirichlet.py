"""Tune on the unchanged Dirichlet problem, validating against saved Cholesky."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as ss
import warp as wp
from compact_candidate import CompactFSAI
from dirichlet_inputs import load_laplacian
from dragon import berr, norm, upload
from ell_candidate import ELLFSAI
from grouped_candidate import GroupedFSAI, GroupedOperator
from scipy.sparse.csgraph import reverse_cuthill_mckee
from squared_candidate import load_squared
from warp.optim import linear

from warp_preconditioners import FSAI, SparseOperator, SquaredLaplacianOperator


def load_problem(path=Path("data/dirichlet")):
    d = np.load(path / "fields.npz")
    L = load_laplacian(path)
    free = d["free"]
    cache = path / "tuning_matrix.npz"
    if not cache.exists():
        Q = L @ ss.diags(1 / d["mass"]) @ L
        a = Q[free][:, free].tocsr()
        ss.save_npz(cache, a)
        np.save(path / "tuning_rhs.npy", -(Q @ d["constraints"])[free])
    a = ss.load_npz(cache)
    b = np.load(path / "tuning_rhs.npy")
    reference = np.load(path / "cholesky.npy")
    return (
        a,
        b,
        d["mass"][free],
        reference[free],
        float(np.sum(d["mass"] * reference**2)),
        d["vertices"][free],
    )


def permutation(a, vertices, kind):
    if kind == "natural":
        return np.arange(a.shape[0])
    if kind == "rcm":
        return reverse_cuthill_mckee(a, symmetric_mode=True)
    if kind == "morton":
        v = ((vertices - vertices.min(0)) / np.ptp(vertices, axis=0) * ((1 << 20) - 1)).astype(
            np.uint64
        )
        key = np.zeros(len(v), dtype=np.uint64)
        for bit in range(20):
            for axis in range(3):
                key |= ((v[:, axis] >> bit) & 1) << (3 * bit + axis)
        return np.argsort(key, kind="stable")
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configs = json.loads(args.configs.read_text())
    a0, b0, m0, r0, denom, v = load_problem()
    records = []
    orders = {}
    for config in configs:
        kind = config.get("ordering", "natural")
        if kind not in orders:
            start = time.perf_counter()
            perm = permutation(a0, v, kind)
            orders[kind] = (perm, time.perf_counter() - start)
        perm, ordering_s = orders[kind]
        a = a0[perm][:, perm].tocsr()
        b_np, mass, reference = b0[perm], m0[perm], r0[perm]
        A = upload(a, "cuda:0")
        operator = GroupedOperator(A, config["matrix_lanes"]) if config.get("matrix_lanes") else A
        if config.get("matrix_format") == "squared":
            operator = load_squared(perm, config.get("squared_lanes", 1))
        b = wp.array(b_np, dtype=wp.float64, device="cuda:0")
        if config.get("rhs_format") == "factored":
            boundary = np.load("data/dirichlet/fields.npz")["constraints"]
            b = operator.rhs(wp.array(boundary, dtype=wp.float64, device="cuda:0"))
            b_np = b.numpy()
        x = wp.zeros_like(b)
        solver = getattr(linear, config.get("solver", "cg"))

        def make():
            if config.get("width", 8) == 0:
                return linear.preconditioner(A, "diag")
            options = {k: config[k] for k in ["kap_tolerance", "pivot_floor"] if k in config}
            factory = ELLFSAI if config.get("apply_format") == "ell" else FSAI
            if config.get("apply_format") == "compact_grouped":
                options["factor_dtype"] = wp.float32
                options["apply_lanes"] = config.get("factor_lanes", 4)
            if config.get("apply_format") == "compact":
                factory = CompactFSAI
            if config.get("apply_format") == "grouped":
                factory = GroupedFSAI
                options["lanes"] = config.get("factor_lanes", 4)
            return factory(A, max_row_size=config.get("width", 8), **options)

        pre = make()
        solver(operator, b, x, M=pre, tol=0.0, atol=0.0, maxiter=10, check_every=0)
        wp.synchronize()
        start = time.perf_counter()
        if isinstance(operator, SquaredLaplacianOperator):
            operator = SquaredLaplacianOperator(
                operator.L, operator.mass, operator.free, row_lanes=operator.row_lanes
            )
        elif isinstance(operator, SparseOperator):
            operator = SparseOperator(A, row_lanes=operator.row_lanes)
        wp.synchronize()
        operator_setup_s = time.perf_counter() - start
        start = time.perf_counter()
        pre = make()
        wp.synchronize()
        setup = time.perf_counter() - start
        x.zero_()
        wp.synchronize()
        start = time.perf_counter()
        nit, res, _ = solver(
            operator,
            b,
            x,
            M=pre,
            tol=config.get("tol", 1e-12),
            atol=0.0,
            maxiter=config.get("maxiter", 200000),
            check_every=0,
        )
        wp.synchronize()
        elapsed = time.perf_counter() - start
        residual = wp.empty_like(b)
        linear.aslinearoperator(operator).matvec(x, b, residual, -1.0, 1.0)
        operator_relative_residual = norm(residual.numpy()) / norm(b_np)
        u = x.numpy()
        if config.get("save_field"):
            np.save(config["save_field"], u[np.argsort(perm)])
        iterations = int(nit.numpy()[0])
        record = dict(
            config=config,
            ordering_s=ordering_s,
            setup_s=setup,
            solve_s=elapsed,
            total_s=operator_setup_s + setup + elapsed,
            operator_setup_s=operator_setup_s,
            iterations=iterations,
            us_per_iteration=elapsed / iterations * 1e6,
            recursive_relative_residual=float(np.sqrt(res.numpy()[0])) / norm(b_np),
            operator_relative_residual=operator_relative_residual,
            relative_residual=norm(b_np - a @ u) / norm(b_np),
            backward_error=berr(a, abs(a), b_np, u),
            mass_relative_error=float(np.sqrt(np.sum(mass * (u - reference) ** 2) / denom)),
            max_abs_error=float(np.max(abs(u - reference))),
            factor_nnz=pre.G.nnz_sync() if isinstance(pre, FSAI) else 0,
            truncated_rows=pre.truncated_rows if isinstance(pre, FSAI) else 0,
        )
        refined_path = Path("data/tuning/refined_reference.npy")
        if refined_path.exists():
            data = np.load("data/dirichlet/fields.npz")
            full_reference = np.load(refined_path)
            physical_reference = full_reference[data["free"]][perm]
            physical_denom = np.sum(data["mass"] * full_reference**2)
            record["physical_mass_relative_error"] = float(
                np.sqrt(np.sum(mass * (u - physical_reference) ** 2) / physical_denom)
            )
            record["physical_accuracy_pass"] = record["physical_mass_relative_error"] < 1e-6
        record["reached_tolerance"] = record["recursive_relative_residual"] <= config.get(
            "tol", 1e-12
        )
        record["accuracy_pass"] = record["mass_relative_error"] < 1e-4
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    problem="unchanged head/tail Dirichlet; zero free initial guess",
                    reference="mass_relative_error: stored-Q CHOLMOD; physical_mass_relative_error: long-double factored-gradient refinement",
                    results=records,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

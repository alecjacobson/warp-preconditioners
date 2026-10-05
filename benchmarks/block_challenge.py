"""Block-coupling and reinforced-bracket experiments, entirely Warp numerical solves.

CPU NumPy handles geometry, exact fields, independent element checks and reporting.
All assembled sparse operators, preconditioners, references and Krylov iterations
run on the selected CUDA device. No external sparse solver or multigrid is used.
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from block_fsai_candidate import BsrBlockFSAI
from elasticity_problem import assemble, independent_force
from fsai_experiments import require_exclusive_gpu
from residual_history import sample_iterations
from spd_block_jacobi_candidate import SpdBlockJacobi
from warp.optim import linear

from warp_preconditioners import FSAI, BlockJacobi


@wp.kernel(enable_backward=False)
def grid_blocks(
    n: int,
    coupling: wp.mat33d,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    values: wp.array(dtype=wp.mat33d),
):
    i = wp.tid()
    x, y, z = i % n, (i // n) % n, i // (n * n)
    for k in range(7):
        j = i
        valid = bool(True)
        if k == 1:
            j, valid = i - 1, x > 0
        elif k == 2:
            j, valid = i + 1, x < n - 1
        elif k == 3:
            j, valid = i - n, y > 0
        elif k == 4:
            j, valid = i + n, y < n - 1
        elif k == 5:
            j, valid = i - n * n, z > 0
        elif k == 6:
            j, valid = i + n * n, z < n - 1
        e = 7 * i + k
        rows[e] = i
        cols[e] = wp.max(0, j)
        value = wp.mat33d(wp.float64(0))
        if k == 0:
            value = wp.float64(6) * coupling
        elif valid:
            value = -coupling
        values[e] = value


def rotation(name):
    if name == "aligned":
        return np.eye(3)
    # Fixed generic orthogonal basis, deterministic and independent of kappa.
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.83 if name == "rotated" else 1.41
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def coupled(n, contrast, direction):
    r = rotation(direction)
    c = r @ np.diag([1.0, 1.0, contrast]) @ r.T
    rows = wp.empty(n**3 * 7, dtype=int)
    cols, values = wp.empty_like(rows), wp.empty(n**3 * 7, dtype=wp.mat33d)
    wp.launch(grid_blocks, n**3, [n, wp.mat33d(c), rows, cols, values])
    a = sp.bsr_from_triplets(n**3, n**3, rows, cols, values)
    # Same physical exact field, rotated with the material axes. Smooth and
    # broadband components avoid a fortuitous one-eigenmode right hand side.
    xyz = np.indices((n, n, n)).reshape(3, -1).T[:, ::-1] / (n + 1) + 1 / (n + 1)
    rng = np.random.default_rng(9241)
    envelope = np.prod(np.sin(np.pi * xyz), axis=1)[:, None]
    exact = envelope * (
        rng.normal(size=(n**3, 3)) * 0.15 + np.sin(xyz * [7.0, 11.0, 17.0]) + [1.0, -0.5, 0.7]
    )
    exact = exact @ r.T
    x = wp.array(exact, dtype=wp.vec3d)
    b = wp.empty_like(x)
    linear.aslinearoperator(a).matvec(x, b, b, 1.0, 0.0)
    # Analytic block Jacobi action is C^-1 / 6 at every node.
    pre, out = SpdBlockJacobi(a), wp.empty_like(x)
    pre.matvec(b, out, out, 1.0, 0.0)
    expected = b.numpy() @ np.linalg.inv(c).T / 6
    control = np.linalg.norm(out.numpy() - expected) / np.linalg.norm(expected)
    assert control < 1e-8
    direct = BlockJacobi(a, pivot_tolerance=1e-15)
    direct.matvec(b, out, out, 1.0, 0.0)
    direct_error = np.linalg.norm(out.numpy() - expected) / np.linalg.norm(expected)
    return (
        a,
        b,
        exact,
        np.ones(n**3),
        dict(
            block_jacobi_action_error=float(control),
            direct_block_jacobi_action_error=float(direct_error),
            direct_pivot_tolerance=1e-15,
            nodes=n**3,
            grid=n,
            coupling=c.tolist(),
        ),
    )


def fiber_field(vertices, tets, direction):
    centers = vertices[tets].mean(axis=1)
    if direction == "varying":
        s = (centers - vertices.min(axis=0)) / np.ptp(vertices, axis=0)
        theta = 2 * np.pi * s[:, 0]
        q = np.column_stack([np.cos(theta), np.sin(theta), 0.5 + s[:, 2]])
    else:
        q = np.tile(rotation(direction)[:, 2], (len(tets), 1))
    return q / np.linalg.norm(q, axis=1)[:, None]


def native(a, b, pre, solver="cg", limit=20000, tol=0.0):
    x = wp.zeros_like(b)
    wp.synchronize()
    start = time.perf_counter()
    nit, _, _ = getattr(linear, solver)(
        a, b, x, M=pre, maxiter=limit, tol=tol, atol=0.0, check_every=0
    )
    wp.synchronize()
    return x, int(nit.numpy()[0]), time.perf_counter() - start


def bracket(contrast, direction, mesh):
    path = Path("data") / mesh
    d = np.load(path / "mesh.npz")
    meta = json.loads((path / "mesh.json").read_text())
    v, t, fixed = d["vertices"], d["tets"], d["fixed"]
    young, nu = meta["young_pa"], meta["poisson"]
    mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
    tau = (contrast - 1) * (lam + 2 * mu)
    q = fiber_field(v, t, direction)
    a, _, volume, free = assemble(
        v, t, fixed, young, nu, meta["density_kg_m3"], fiber=q, reinforcement=tau
    )
    b = wp.array(d["forces"][free], dtype=wp.vec3d)
    op = linear.aslinearoperator(a)
    # Different preconditioners for primary reference and residual correction.
    x, nit, _ = native(
        a,
        b,
        FSAI(a, max_row_size=24, kap_tolerance=0.001, factor_dtype=wp.float64),
        limit=100000,
        tol=2e-12,
    )
    residual = wp.empty_like(b)
    op.matvec(x, b, residual, -1.0, 1.0)
    correction, cnit, _ = native(
        a, residual, BlockJacobi(a, pivot_tolerance=1e-15), limit=100000, tol=1e-10
    )
    ref, delta = x.numpy(), correction.numpy()
    mass = volume.numpy()[free]
    change = np.sqrt(np.sum(mass[:, None] * delta**2) / np.sum(mass[:, None] * ref**2))
    ref += delta
    full = np.zeros_like(v)
    full[free] = ref
    independent = independent_force(v, t, full, young, nu, q, tau)[free]
    rel = np.linalg.norm(independent - b.numpy()) / np.linalg.norm(b.numpy())
    assert change < 1e-7 and rel < 1e-7, (change, rel, nit, cnit)
    check = dict(
        reference_iterations=nit,
        correction_iterations=cnit,
        reference_correction_mass_error=float(change),
        element_relative_residual=float(rel),
        nodes=len(free),
        tets=len(t),
        poisson=nu,
        reinforcement_pa=tau,
        compliance=float(np.sum(ref * b.numpy())),
        maximum_displacement_m=float(np.linalg.norm(ref, axis=1).max()),
        mesh=mesh,
    )
    print("reference", contrast, direction, check, flush=True)
    return a, b, ref, mass, check


class Diagnostics:
    def __init__(self, a, b, ref, mass):
        self.op, self.b, self.ref, self.mass = linear.aslinearoperator(a), b, ref, mass[:, None]
        self.work = wp.empty_like(b)
        self.e = wp.empty_like(b)
        self.field_denom = np.sum(self.mass * ref**2)
        self.op.matvec(wp.array(ref, dtype=wp.vec3d), self.work, self.work, 1.0, 0.0)
        self.energy_denom = np.sum(ref * self.work.numpy())
        self.bnorm = np.linalg.norm(b.numpy())

    def __call__(self, x):
        error = x.numpy() - self.ref
        self.e.assign(error)
        self.op.matvec(self.e, self.work, self.work, 1.0, 0.0)
        energy = np.sqrt(max(0.0, np.sum(error * self.work.numpy())) / self.energy_denom)
        self.op.matvec(x, self.b, self.work, -1.0, 1.0)
        return dict(
            field_error=float(np.sqrt(np.sum(self.mass * error**2) / self.field_denom)),
            energy_error=float(energy),
            true_relative_residual=float(np.linalg.norm(self.work.numpy()) / self.bnorm),
        )


class Enough(Exception):
    pass


def trace(a, b, pre, solver, diag, target, limit):
    x, samples = wp.zeros_like(b), []

    def observe(it, recursive):
        row = dict(
            iteration=int(it), recursive_relative_residual=float(recursive / diag.bnorm), **diag(x)
        )
        samples.append(row)
        if max(row["field_error"], row["energy_error"]) <= target:
            raise Enough

    # Dense near the start; 25-iteration resolution after that. No restarts.
    points = sorted(set([1, 2, 5, 10, 15, 20] + list(range(25, limit + 25, 25))))
    try:
        with sample_iterations(points, observe):
            getattr(linear, solver)(a, b, x, M=pre, maxiter=limit, tol=0.0, atol=0.0, check_every=0)
    except Enough:
        pass
    hit = max(samples[-1]["field_error"], samples[-1]["energy_error"]) <= target
    return samples, samples[-1]["iteration"] if hit else None


def configurations(quick=False):
    configs = [
        dict(family="jacobi"),
        dict(family="block_jacobi", variant="direct"),
        dict(family="block_jacobi", variant="cholesky"),
    ]
    for width in [4, 6] if quick else [4, 6, 12]:
        for step in [1] if quick else [1, 2]:
            configs.append(dict(family="fsai", width=width, step=step))
    for width in [2] if quick else [2, 4]:
        configs.append(dict(family="block_fsai", width=width))
    return configs


def make(a, cfg, reuse=False):
    family = cfg["family"]
    if family == "jacobi":
        return linear.preconditioner(a, "diag")
    if family == "block_jacobi":
        return (
            SpdBlockJacobi(a)
            if cfg["variant"] == "cholesky"
            else BlockJacobi(a, pivot_tolerance=1e-15)
        )
    if family == "fsai":
        return FSAI(
            a,
            max_row_size=cfg["width"],
            max_step_size=cfg["step"],
            kap_tolerance=0.003,
            factor_dtype=wp.float32,
            reuse_pattern=reuse,
        )
    return BsrBlockFSAI(a, max_blocks=cfg["width"], factor_dtype=wp.float32, reuse_pattern=reuse)


def factor_bytes(pre):
    # Count actual retained factor storage, including prototype's scalar copies.
    result, seen = {}, set()
    for cell in getattr(pre.matvec, "__closure__", None) or []:
        array = cell.cell_contents
        if isinstance(array, wp.array):
            result["diagonal"] = int(array.capacity)
            seen.add(array.ptr)
    for name in ["G", "GT", "B", "BT", "inverse_diagonal"]:
        obj = getattr(pre, name, None)
        arrays = (
            [obj]
            if isinstance(obj, wp.array)
            else [getattr(obj, k, None) for k in ["offsets", "columns", "values"]]
        )
        total = 0
        for array in arrays:
            if array is not None and array.ptr not in seen:
                seen.add(array.ptr)
                total += array.capacity
        if total:
            result[name] = int(total)
    return result


def wait_for_gpu():
    while True:
        try:
            require_exclusive_gpu()
            return
        except RuntimeError as error:
            print("Waiting for an isolated GPU:", error, flush=True)
            time.sleep(15)


def guarded_trial(a, b, warm_pre, build, solver, budget):
    while True:
        wait_for_gpu()
        native(a, b, warm_pre, solver, min(budget, 2000))
        try:
            require_exclusive_gpu()
            wp.synchronize()
            start = time.perf_counter()
            pre = build()
            wp.synchronize()
            setup = time.perf_counter() - start
            x, nit, elapsed = native(a, b, pre, solver, budget)
            require_exclusive_gpu()
            return x, nit, elapsed, setup
        except RuntimeError as error:
            if "Other GPU processes" not in str(error):
                raise
            print("Discarded overlapping timing:", error, flush=True)


def run_case(args, contrast, direction, output):
    wait_for_gpu()

    def load(k):
        return (
            coupled(args.grid, k, direction)
            if args.problem == "coupling"
            else bracket(k, direction, args.mesh)
        )

    a, b, ref, mass, checks = load(contrast)
    diag = Diagnostics(a, b, ref, mass)
    a.nnz_sync()
    sig = hashlib.sha256()
    for arr in [
        a.offsets.numpy()[: a.nrow + 1],
        a.columns.numpy()[: a.nnz],
        a.values.numpy()[: a.nnz],
        b.numpy(),
    ]:
        sig.update(arr.tobytes())
    result = dict(
        contrast=contrast,
        direction=direction,
        checks=checks,
        matrix_rhs_sha256=sig.hexdigest(),
        target=args.target,
        candidates={},
    )
    pres, configs = {}, {}
    for cfg in configurations(args.quick):
        pre = make(a, cfg)
        for solver in args.solvers:
            name = "-".join(str(v) for v in cfg.values()) + "-" + solver
            samples, budget = trace(a, b, pre, solver, diag, args.target, args.limit)
            result["candidates"][name] = dict(
                config=cfg,
                solver=solver,
                samples=samples,
                iterations=budget,
                trials=[],
                factor_bytes=factor_bytes(pre),
            )
            print("trace", contrast, direction, name, budget, samples[-1], flush=True)
            if budget is not None:
                pres[name], configs[name] = pre, cfg
    # Warm all candidates first, then alternate ordering; all trials are verified.
    for repeat in range(args.repeats):
        order = list(pres)
        if repeat % 2:
            order.reverse()
        for name in order:
            row = result["candidates"][name]
            x, nit, elapsed, setup = guarded_trial(
                a, b, pres[name], lambda: make(a, configs[name]), row["solver"], row["iterations"]
            )
            errors = diag(x)
            assert max(errors["field_error"], errors["energy_error"]) <= args.target, (name, errors)
            row["trials"].append(
                dict(
                    setup_s=setup,
                    solve_s=elapsed,
                    total_s=setup + elapsed,
                    iterations=nit,
                    **errors,
                )
            )
        print("round", contrast, direction, repeat, flush=True)
    winners = {}
    for family in ["jacobi", "block_jacobi", "fsai", "block_fsai"]:
        valid = [name for name in pres if configs[name]["family"] == family]
        if valid:
            winners[family] = min(
                valid,
                key=lambda name: np.median(
                    [r["total_s"] for r in result["candidates"][name]["trials"]]
                ),
            )
    result["winners"] = winners
    # Fixed graph, changed numerical entries: refit supports selected at the old
    # material versus fresh adaptive setup. All methods still start x at zero.
    if args.updates:
        updated_contrast = contrast * 1.1
        aa, bb, rr, mm, cc = load(updated_contrast)
        dd = Diagnostics(aa, bb, rr, mm)
        updates = dict(contrast=updated_contrast, checks=cc, candidates={})
        for family, name in winners.items():
            cfg, solver = configs[name], result["candidates"][name]["solver"]
            for mode in ["rebuild", "refit"]:
                pre = make(a, cfg, reuse=mode == "refit")

                def update():
                    nonlocal pre
                    if mode == "rebuild" or family == "jacobi":
                        pre = make(aa, cfg)
                    else:
                        pre.update(aa)
                    return pre

                update()
                samples, budget = trace(aa, bb, pre, solver, dd, args.target, args.limit)
                trials = []
                if budget is not None:
                    for _ in range(args.repeats):
                        x, nit, secs, elapsed = guarded_trial(aa, bb, pre, update, solver, budget)
                        errors = dd(x)
                        assert max(errors["field_error"], errors["energy_error"]) <= args.target
                        trials.append(
                            dict(
                                update_s=elapsed,
                                solve_s=secs,
                                total_s=elapsed + secs,
                                iterations=nit,
                                **errors,
                            )
                        )
                updates["candidates"][family + "-" + mode] = dict(
                    config=cfg,
                    solver=solver,
                    iterations=budget,
                    samples=samples,
                    trials=trials,
                    factor_bytes=factor_bytes(pre),
                    note="Jacobi is reconstructed in both modes" if family == "jacobi" else "",
                )
                print("update", contrast, direction, family, mode, budget, flush=True)
        result["updates"] = updates
    output["cases"].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--problem", choices=["coupling", "elasticity"], required=True)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--mesh", default="simjeb-ftetwild")
    p.add_argument("--contrasts", type=float, nargs="+", default=[1, 100, 10000, 1000000])
    p.add_argument("--directions", nargs="+", default=["aligned", "rotated"])
    p.add_argument("--solvers", nargs="+", default=["cg", "cr"], choices=["cg", "cr"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--limit", type=int, default=30000)
    p.add_argument("--target", type=float, default=1e-6)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--updates", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    source = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__),
            Path("benchmarks/block_fsai_candidate.py"),
            Path("benchmarks/elasticity_problem.py"),
            Path("benchmarks/spd_block_jacobi_candidate.py"),
            Path("warp_preconditioners/fsai.py"),
        ]
    }
    output = dict(
        problem=args.problem,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[],
        provenance=dict(
            utc=datetime.now(timezone.utc).isoformat(),
            warp=wp.__version__,
            gpu=wp.get_device().name,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256=source,
            gpu_isolation="Guarded before/after every timed trial",
            timing="JIT-warm setup plus native solve/graph capture; GPU assembly and CPU diagnostics excluded",
            accuracy="Both mass-weighted forward error and energy error; fresh residual also reported",
            selection="Best median setup+solve among tested widths, growth batches and CG/CR; not exhaustive",
            updates="10% contrast increase, fixed graph and zero starting solution; repeated timings of one transition",
        ),
    )
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key, value in output["arguments"].items():
            if key != "resume" and previous["arguments"].get(key) != value:
                raise ValueError("Resume arguments differ: " + key)
        output["cases"] = previous["cases"]
        output["previous_provenance"] = previous.get("previous_provenance", []) + [
            previous["provenance"]
        ]
    for contrast in args.contrasts:
        for direction in args.directions:
            if not any(
                c["contrast"] == contrast and c["direction"] == direction for c in output["cases"]
            ):
                run_case(args, contrast, direction, output)


if __name__ == "__main__":
    main()

"""Diagnose the original Warp magnetostatics example without FSAI.

Run using a Warp checkout's Python. Assembly, interpolation and CR use Warp
on the selected device. NumPy is used only for diagnostic reductions and
seeded control-vector generation; there is no external sparse solver.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.examples.fem import example_magnetostatics as ex
from warp.optim import linear


@fem.integrand
def gradient(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


@fem.integrand
def div_load(s: fem.Sample, v: fem.Field, current: fem.Field):
    return wp.dot(fem.grad(v, s), current(s))


class Captured(Exception):
    pass


def run(res, fp64, maxiter):
    example = ex.Example(resolution=res, fp64=fp64)
    captured = {}
    old = ex.fem_example_utils.bsr_cg

    def capture(A, b, x, **kwargs):
        captured.update(A=A, b=b, x=x)
        raise Captured()

    ex.fem_example_utils.bsr_cg = capture
    try:
        example.step()
    except Captured:
        pass
    finally:
        ex.fem_example_utils.bsr_cg = old
    A, b = captured["A"], captured["b"]
    scalar = A.scalar_type
    space = fem.make_polynomial_space(example.A_field.space.geometry, degree=1, dtype=scalar)
    pos = space.node_positions().numpy()
    boundary = (np.abs(pos[:, 1]) > 2 - 1.0e-5) | (
        np.linalg.norm(pos[:, [0, 2]], axis=1) > 2 - 1.0e-5
    )
    load = fem.integrate(
        div_load,
        fields={"v": fem.make_test(space), "current": example._current_field},
        output_dtype=scalar,
    )
    values = load.numpy()
    values[boundary] = 0
    load_norm = np.linalg.norm(values)
    if not np.isfinite(load_norm) or load_norm == 0:
        raise ValueError("No nonzero finite compatibility probe at this resolution")
    values /= load_norm
    # The covariant Piola mapping preserves reference edge DOFs. Build the
    # gradient on the regular base grid, avoiding the singular mapping axis.
    base = example.A_field.space.geometry.base
    base_scalar = fem.make_polynomial_space(base, degree=1, dtype=scalar)
    base_phi = base_scalar.make_field()
    base_phi.dof_values.assign(values)
    base_edge = fem.make_polynomial_space(
        base,
        degree=1,
        dtype=wp.vec3d if fp64 else wp.vec3,
        element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND,
    )
    qfield = base_edge.make_field()
    fem.interpolate(gradient, dest=qfield, fields={"u": base_phi})
    q = qfield.dof_values
    Atq = sp.bsr_mv(sp.bsr_transposed(A), q)
    # Use float64 host reductions only to inspect assembled Warp vectors.
    qn, bn, aqn = (v.numpy().astype(np.float64) for v in (q, b, Atq))
    av = A.values.numpy()[: A.nnz_sync()].astype(np.float64)
    result = dict(
        resolution=res,
        fp64=fp64,
        n=A.nrow,
        b_norm=float(np.linalg.norm(bn)),
        q_norm=float(np.linalg.norm(qn)),
        matrix_frobenius=float(np.linalg.norm(av)),
        Atq_norm=float(np.linalg.norm(aqn)),
        null_defect=float(np.linalg.norm(aqn) / (np.linalg.norm(av) * np.linalg.norm(qn))),
        b_dot_q=float(bn @ qn),
        null_rhs_fraction=float(abs(bn @ qn) / (np.linalg.norm(bn) * np.linalg.norm(qn))),
        boundary_dofs=int(np.count_nonzero(boundary)),
    )
    # A compatible control load, with the same matrix and zero boundary values.
    rng = np.random.default_rng(20261006)
    x_known = rng.standard_normal(A.nrow).astype(values.dtype)
    edge_pos = base_edge.node_positions().numpy()
    x_known[np.max(np.abs(edge_pos), axis=1) > 2 - 1.0e-5] = 0
    compatible = sp.bsr_mv(A, wp.array(x_known, dtype=scalar))
    M = linear.preconditioner(A, "diag")
    result["solves"] = {}
    for name, rhs in (("original", b), ("compatible_control", compatible)):
        x = wp.zeros_like(rhs)
        info = linear.cr(A, rhs, x, M=M, tol=1.0e-4, atol=0.0, maxiter=maxiter, use_cuda_graph=True)
        # Evaluate true residual in double precision even for float32 assembly.
        A64 = sp.bsr_copy(A, scalar_type=wp.float64)
        x64 = wp.array(x.numpy().astype(np.float64), dtype=wp.float64)
        rhs64 = rhs.numpy().astype(np.float64)
        ax64 = sp.bsr_mv(A64, x64).numpy()
        result["solves"][name] = dict(
            iterations=int(info[0]),
            recursive_residual=float(info[1]),
            true_relative_residual=float(np.linalg.norm(rhs64 - ax64) / np.linalg.norm(rhs64)),
            target_relative_residual=1.0e-4,
        )
    print("RESULT " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    wp.init()
    wp.set_module_options({"enable_backward": False})
    with wp.ScopedDevice(args.device):
        result = run(args.resolution, args.fp64, args.maxiter)
    result.update(warp_version=wp.__version__, device=args.device, maxiter=args.maxiter)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

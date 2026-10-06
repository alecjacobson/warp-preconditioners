"""Explore load assembly variants; no changes to the upstream example."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.examples.fem import example_magnetostatics as ex
from warp.optim import linear


@wp.func
def current_potential(
    pos: Any,
    current: float,
    coil_internal_radius: float,
    coil_external_radius: float,
    coil_height: float,
):
    r = wp.sqrt(pos[0] * pos[0] + pos[2] * pos[2])
    z = pos.dtype(0.0)
    h = pos.dtype(coil_height)
    ir = pos.dtype(coil_internal_radius)
    er = pos.dtype(coil_external_radius)
    amplitude = pos.dtype(current) * wp.clamp(er - r, z, er - ir)
    return type(pos)(z, wp.where(wp.abs(pos[1]) < h, amplitude, z), z)


@fem.integrand
def curl_load(s: fem.Sample, v: fem.Field, potential: fem.Field):
    return wp.dot(potential(s), fem.curl(v, s))


@fem.integrand
def gradient(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


class Captured(Exception):
    pass


def main(args):
    example = ex.Example(resolution=args.resolution, fp64=args.fp64, mesh=args.mesh)
    old = ex.fem_example_utils.bsr_cg
    captured = {}

    def capture(A, b, x, **kwargs):
        captured.update(A=A, b=b)
        raise Captured()

    ex.fem_example_utils.bsr_cg = capture
    try:
        example.step()
    except Captured:
        pass
    finally:
        ex.fem_example_utils.bsr_cg = old
    A = captured["A"]
    space = example.A_field.space
    scalar = A.scalar_type
    geo = space.geometry
    domain = fem.Cells(geo)
    v = fem.make_test(space)
    potential = fem.ImplicitField(
        domain,
        func=current_potential,
        values=dict(
            current=1.0e6, coil_internal_radius=0.3, coil_external_radius=0.4, coil_height=0.25
        ),
    )
    # Boundary projector for each fresh load; K already has boundary conditions.
    boundary = fem.BoundarySides(geo)
    P = fem.integrate(
        ex.mass_form,
        fields={
            "u": fem.make_trial(space, domain=boundary),
            "v": fem.make_test(space, domain=boundary),
        },
        assembly="nodal",
        output_dtype=scalar,
    )
    fem.normalize_dirichlet_projector(P)
    # On the regular Grid3D base, interpolation yields the exact edge
    # differences. Do not use this construction on general nonorthogonal cells.
    GT = None
    if args.mesh == "grid":
        scalar_space = fem.make_polynomial_space(geo.base, degree=1, dtype=scalar)
        edge_space = fem.make_polynomial_space(
            geo.base,
            degree=1,
            dtype=wp.vec3d if args.fp64 else wp.vec3,
            element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND,
        )
        G = sp.bsr_zeros(A.nrow, scalar_space.node_count(), scalar)
        fem.interpolate(
            gradient, dest=G, dest_space=edge_space, fields={"u": fem.make_trial(scalar_space)}
        )
        GT = sp.bsr_transposed(G)
        scalar_pos = scalar_space.node_positions().numpy()
        bd = np.max(np.abs(scalar_pos), axis=1) > 2 - 1.0e-5
    variants = [("original", captured["b"])]
    for order in args.orders:
        quad = fem.RegularQuadrature(domain, order=order) if order >= 0 else None
        for kind in ("direct", "curl"):
            if order == -1 and kind == "direct":
                continue
            fields = (
                {"v": v, "u": example._current_field}
                if kind == "direct"
                else {"v": v, "potential": potential}
            )
            b = fem.integrate(
                ex.mass_form if kind == "direct" else curl_load,
                fields=fields,
                quadrature=quad,
                output_dtype=scalar,
            )
            sp.bsr_mv(P, b, b, alpha=-1.0, beta=1.0)
            variants.append((f"{kind}_{order}", b))
    A64 = sp.bsr_copy(A, scalar_type=wp.float64)
    M = linear.preconditioner(A, "diag")
    records = []
    for name, b in variants:
        bn = b.numpy().astype(np.float64)
        record = dict(name=name, b_norm=float(np.linalg.norm(bn)))
        if GT is not None:
            div = sp.bsr_mv(GT, b).numpy().astype(np.float64)
            div[bd] = 0
            record["gradient_pairing_norm_over_b"] = float(np.linalg.norm(div) / np.linalg.norm(bn))
        for maxiter in args.iterations:
            x = wp.zeros_like(b)
            info = linear.cr(
                A, b, x, M=M, tol=1.0e-4, atol=0.0, maxiter=maxiter, use_cuda_graph=True
            )
            x64 = wp.array(x.numpy().astype(np.float64), dtype=wp.float64)
            res = bn - sp.bsr_mv(A64, x64).numpy()
            record[f"solve_{maxiter}"] = dict(
                iterations=int(info[0]),
                recursive_relative_residual=float(info[1] / np.linalg.norm(bn)),
                true_relative_residual=float(np.linalg.norm(res) / np.linalg.norm(bn)),
            )
        print("RESULT " + json.dumps(record), flush=True)
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(fp64=args.fp64, mesh=args.mesh, resolution=args.resolution, results=records),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--mesh", default="grid")
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--orders", type=int, nargs="+", default=[-1, 4, 8])
    p.add_argument("--iterations", type=int, nargs="+", default=[250, 1000])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    wp.init()
    wp.set_module_options({"enable_backward": False})
    with wp.ScopedDevice("cuda:0"):
        main(args)

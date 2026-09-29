"""Pure squared-Laplacian interpolation on the dragon, with head/tail constraints."""

import argparse
import hashlib
import json
from pathlib import Path

import igl
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from scipy.sparse.csgraph import connected_components
from solve_fields import run


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dir", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/dirichlet"))
    parser.add_argument("--fsai-rtol", type=float, default=1e-12)
    parser.add_argument("--fsai-maxiter", type=int, default=500000)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write assembled inputs without running the solver comparison",
    )
    args = parser.parse_args()
    if args.fsai_rtol <= 0 or args.fsai_maxiter <= 0:
        parser.error("FSAI tolerance and maximum iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    v, f = igl.read_triangle_mesh(str(args.mesh))
    n = len(v)
    mixed = sio.mmread(args.dir / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n]
    L = -mixed[:n, n:]
    del mixed
    assert mass.min() > 0
    cot_error = float(abs(L + igl.cotmatrix(v, f)).max()) / float(abs(L).max())
    assert cot_error < 1e-7
    graph = igl.adjacency_matrix(f).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    assert components == 1

    def patch(mask):
        indices = np.flatnonzero(mask)
        count, labels = connected_components(graph[indices][:, indices], directed=False)
        chosen = indices[labels == np.argmax(np.bincount(labels))]
        return chosen, int(count)

    tail, tail_components = patch((v[:, 0] < -70) & (v[:, 2] > 45))
    head, head_components = patch((v[:, 0] > 70) & (v[:, 2] > 55))
    fixed = np.concatenate([tail, head])
    values = np.concatenate([-np.ones(len(tail)), np.ones(len(head))])
    assert len(np.unique(fixed)) == len(fixed)
    free = np.setdiff1d(np.arange(n), fixed)
    initial = np.zeros(n)
    initial[fixed] = values
    # Form the full energy BEFORE restricting; retain all rows of L in the energy.
    Q = (L @ ss.diags(1 / mass) @ L).tocsr()
    symmetry_error = float(abs(Q - Q.T).max()) / float(abs(Q).max())
    assert symmetry_error < 1e-14
    A = Q[free][:, free].tocsr()
    rhs = -(Q @ initial)[free]
    fields = dict(
        vertices=v,
        faces=f,
        constraints=initial,
        fixed=fixed,
        free=free,
        tail=tail,
        head=head,
        mass=mass,
    )
    meta = dict(
        problem_type="dirichlet",
        problem="minimize 0.5*u^T L M^-1 L u; u[tail]=-1, u[head]=1; no data term",
        initial_guess="zero on free vertices; prescribed values on fixed vertices",
        mesh=str(args.mesh),
        vertices=n,
        triangles=len(f),
        mesh_sha256=digest(v),
        faces_sha256=digest(f),
        matrix_shape=list(A.shape),
        matrix_nnz=A.nnz,
        full_matrix_nnz=Q.nnz,
        connected_components=int(components),
        relative_symmetry_error=symmetry_error,
        relative_cotmatrix_error=cot_error,
        constraints=dict(
            tail=dict(
                selection="largest connected component of x < -70 and z > 45",
                vertices=len(tail),
                candidate_components=tail_components,
                value=-1,
            ),
            head=dict(
                selection="largest connected component of x > 70 and z > 55",
                vertices=len(head),
                candidate_components=head_components,
                value=1,
            ),
            sha256=digest(initial),
            fixed_index_sha256=digest(fixed),
        ),
        convergence=dict(
            fsai_rtol=args.fsai_rtol,
            fsai_maxiter=args.fsai_maxiter,
            criterion="Warp recursive relative residual; independently verify field stability",
        ),
        results={},
        display_order=[],
    )

    def save():
        np.savez_compressed(args.output / "fields.npz", **fields)
        (args.output / "solve.json").write_text(json.dumps(meta, indent=2) + "\n")

    def record(sol, result, multiplier):
        u = initial.copy()
        u[free] = sol
        name = ("fsai_cg_" if "FSAI" in result["preconditioner"] else "jacobi_cr_") + str(
            result["actual_iterations"]
        )
        result.update(
            iteration_multiplier=multiplier,
            field_sha256=digest(u),
            range=[float(u.min()), float(u.max())],
            bending_energy=float(0.5 * np.sum((L @ u) ** 2 / mass)),
            constraint_max_error=float(np.max(np.abs(u[fixed] - values))),
        )
        assert result["constraint_max_error"] == 0
        if meta["display_order"]:
            reference = fields[meta["display_order"][0]]
            result["mass_relative_error_to_fsai"] = float(
                np.sqrt(np.sum(mass * (u - reference) ** 2) / np.sum(mass * reference**2))
            )
        else:
            result["mass_relative_error_to_fsai"] = 0.0
        fields[name] = u
        meta["results"][name] = result
        meta["display_order"].append(name)
        print(name, json.dumps(result), flush=True)
        save()
        return u

    meta["initial_bending_energy"] = float(0.5 * np.sum((L @ initial) ** 2 / mass))
    save()
    print("ASSEMBLY", json.dumps(meta), flush=True)
    if args.prepare_only:
        return
    sol, result = run(A, rhs, "fsai_cg", args.fsai_maxiter, rtol=args.fsai_rtol)
    assert result["reached_stopping_tolerance"], result
    k = result["actual_iterations"]
    meta["convergence"]["k"] = k
    reference = record(sol, result, 1)
    extended, check = run(A, rhs, "fsai_cg", 2 * k)
    stability = float(
        np.sqrt(np.sum(mass[free] * (extended - sol) ** 2) / np.sum(mass * reference**2))
    )
    meta["convergence"]["extended_solve"] = check
    meta["convergence"]["extended_mass_relative_change"] = stability
    print("STABILITY", stability, flush=True)
    save()
    assert stability < 1e-6
    for multiplier in [1, 10, 100]:
        sol, result = run(A, rhs, "jacobi_cr", multiplier * k)
        assert result["actual_iterations"] == multiplier * k, result
        record(sol, result, multiplier)
    meta["scalar_range"] = [
        min(float(fields[name].min()) for name in meta["display_order"]),
        max(float(fields[name].max()) for name in meta["display_order"]),
    ]
    save()


if __name__ == "__main__":
    main()

"""Independently audit the saved SimJEB solve, constraints, loads and stress.

NumPy/SciPy are diagnostic backends only here. No new solution is computed,
no preconditioner is built, and no benchmark timing is changed.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from elasticity_problem import independent_force
from scipy.spatial import cKDTree
from simjeb_problem import SHA256, cards, number


def main():
    root = Path("data/simjeb")
    mesh = np.load(root / "mesh.npz")
    fields = np.load(root / "comparison.npz")
    meta = json.loads((root / "mesh.json").read_text())
    v, t, fixed = mesh["vertices"], mesh["tets"], mesh["fixed"]
    u, vm_warp = fields["reference"], fields["von_mises_reference"]
    deck = (root / "225.fem").read_text()
    assert hashlib.sha256((root / "225.fem").read_bytes()).hexdigest() == SHA256
    case = re.search(r"SUBCASE\s+1\s+(.*?)SUBCASE\s+2", deck, re.S).group(1)
    assert re.search(r"SPC\s*=\s*1", case) and re.search(r"LOAD\s*=\s*2", case)
    all_cards = list(cards(deck))
    ids = np.unique([[int(x) for x in c[3:7]] for c in all_cards if c[0] == "CTETRA"])
    points = {
        int(c[1]): np.array([number(x) * 0.001 for x in c[3:6]])
        for c in all_cards
        if c[0] == "GRID"
    }
    np.testing.assert_array_equal(v, np.array([points[int(i)] for i in ids]))
    constraints = {
        int(c[2])
        for c in all_cards
        if c[0] == "SPC" and c[1] == "1" and c[3] == "123456" and number(c[4]) == 0
    }
    spiders = [c for c in all_cards if c[0] == "RBE2"]
    fixed_from_deck = np.zeros(len(v), dtype=bool)
    for c in spiders:
        assert int(c[2]) in constraints and c[3] == "123456"
        fixed_from_deck |= np.isin(ids, [int(x) for x in c[4:] if x])
    np.testing.assert_array_equal(fixed, fixed_from_deck)
    assert np.max(np.abs(u[fixed])) == 0
    rbe3 = next(c for c in all_cards if c[0] == "RBE3")
    assert rbe3[4] == "123456" and number(rbe3[5]) == 1 and rbe3[6] == "123"
    loaded = np.isin(ids, [int(x) for x in rbe3[7:] if x])
    np.testing.assert_array_equal(loaded, np.any(mesh["forces"] != 0, axis=1))
    load_card = next(c for c in all_cards if c[0] == "FORCE" and c[1] == "2")
    assert int(load_card[2]) == int(rbe3[3]) and int(load_card[3]) == 0
    F = number(load_card[4]) * np.array([number(x) for x in load_card[5:8]])
    ref = points[int(rbe3[3])]
    # Independent six-DOF least-squares construction of the load distribution.
    r = v[loaded] - ref
    B = np.zeros((len(r), 3, 6))
    B[:, :, :3] = np.eye(3)
    for k in range(3):
        B[:, :, 3 + k] = np.cross(np.eye(3)[k], r)
    B = B.reshape(-1, 6)
    independent_load = (B @ np.linalg.solve(B.T @ B, np.r_[F, np.zeros(3)])).reshape(-1, 3)
    load_difference = np.linalg.norm(independent_load - mesh["forces"][loaded]) / np.linalg.norm(
        independent_load
    )
    assert load_difference < 1e-12
    E, nu = meta["young_pa"], meta["poisson"]
    force = independent_force(v, t, u, E, nu)
    reaction = force - mesh["forces"]
    force_balance = reaction[fixed].sum(0) + mesh["forces"].sum(0)
    moment_balance = np.cross(v[fixed], reaction[fixed]).sum(0) + np.cross(v, mesh["forces"]).sum(0)
    assert np.linalg.norm(force_balance) / np.linalg.norm(F) < 1e-11
    assert np.linalg.norm(moment_balance) < 1e-7
    bolt_checks = []
    for c in spiders:
        sel = np.isin(ids, [int(x) for x in c[4:] if x])
        bolt_checks.append(
            dict(
                element=int(c[1]),
                center=int(c[2]),
                vertices=int(sel.sum()),
                reaction_n=reaction[sel].sum(0).tolist(),
                min_m=v[sel].min(0).tolist(),
                max_m=v[sel].max(0).tolist(),
            )
        )
    # Independent tensor recovery; this does not call the Warp stress kernel.
    edges = v[t[:, 1:]] - v[t[:, :1]]
    inverse = np.linalg.inv(edges.transpose(0, 2, 1))
    gradients = np.concatenate([-inverse.sum(1, keepdims=True), inverse], axis=1)
    grad = np.einsum("eia,eib->eab", u[t], gradients)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    sigma = mu * (grad + grad.transpose(0, 2, 1)) + lam * np.trace(grad, axis1=1, axis2=2)[
        :, None, None
    ] * np.eye(3)
    volume = np.abs(np.linalg.det(edges)) / 6
    sums = np.zeros((len(v), 3, 3))
    weight = np.zeros(len(v))
    for k in range(4):
        np.add.at(sums, t[:, k], volume[:, None, None] * sigma)
        np.add.at(weight, t[:, k], volume)
    sums /= weight[:, None, None]
    dev = sums - np.trace(sums, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    stress_difference = np.linalg.norm(vm - vm_warp) / np.linalg.norm(vm)
    assert stress_difference < 1e-12
    surface = np.unique(mesh["faces"])
    distance = cKDTree(v[fixed]).query(v)[0]
    hottest = surface[vm_warp[surface] >= np.percentile(vm_warp[surface], 99.9)]
    peak = int(vm_warp.argmax())
    result = dict(
        deck_sha256=SHA256,
        subcase=1,
        spc=1,
        load=2,
        exact_constraint_mask_match=True,
        fixed_vertices=int(fixed.sum()),
        exact_loaded_vertex_mask_match=True,
        loaded_vertices=int(loaded.sum()),
        applied_force_n=F.tolist(),
        applied_reference_m=ref.tolist(),
        independent_load_relative_difference=float(load_difference),
        maximum_fixed_displacement_m=float(np.abs(u[fixed]).max()),
        bolts=bolt_checks,
        force_balance_n=force_balance.tolist(),
        moment_balance_nm=moment_balance.tolist(),
        free_relative_force_residual=float(
            np.linalg.norm(reaction[~fixed]) / np.linalg.norm(mesh["forces"][~fixed])
        ),
        independent_stress_relative_difference=float(stress_difference),
        reference_displacement_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
        reference_stress_sha256=hashlib.sha256(vm_warp.tobytes()).hexdigest(),
        maximum_displacement_mm=float(np.linalg.norm(u, axis=1).max() * 1000),
        peak=dict(
            node=peak,
            nastran_id=int(ids[peak]),
            position_mm=(v[peak] * 1000).tolist(),
            fixed=bool(fixed[peak]),
            stress_mpa=float(vm_warp[peak] / 1e6),
        ),
        surface_stress_percentiles_mpa={
            str(p): float(np.percentile(vm_warp[surface], p) / 1e6)
            for p in [50, 90, 95, 99, 99.9, 100]
        },
        hottest_0_1_percent=dict(
            nodes=len(hottest),
            max_distance_to_fixed_mm=float(distance[hottest].max() * 1000),
            fixed_nodes=int(fixed[hottest].sum()),
        ),
        gray_masked_surface_triangles_in_previous_render=int(
            np.all(fixed[mesh["faces"]], axis=1).sum()
        ),
        limitations="Audits consistency with the imported SimJEB deck and independent stress evaluation; not a mesh-convergence study or a validation of every OptiStruct RBE3 implementation detail.",
    )
    Path("results/simjeb-stress-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

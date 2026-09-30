"""Export exact input node sets for the boundary-condition illustration.

Reads the original FEM deck, compares its sets to the solver input, and exports
undeformed geometry. Does not read or compute a displacement/stress solution.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from simjeb_problem import SHA256, cards, number


def main():
    root = Path("data/simjeb")
    mesh = np.load(root / "mesh.npz")
    deck = root / "225.fem"
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == SHA256
    records = list(cards(deck.read_text()))
    ids = np.unique([[int(x) for x in c[3:7]] for c in records if c[0] == "CTETRA"])
    points = {int(c[1]): [number(x) * 0.001 for x in c[3:6]] for c in records if c[0] == "GRID"}
    np.testing.assert_array_equal(mesh["vertices"], [points[int(i)] for i in ids])
    fixed_group = np.zeros(len(ids), dtype=np.int32)
    centers, bolts = [], []
    for k, c in enumerate(c for c in records if c[0] == "RBE2"):
        center = int(c[2])
        assert c[3] == "123456"
        spc = next(s for s in records if s[0] == "SPC" and s[1] == "1" and int(s[2]) == center)
        assert spc[3] == "123456" and number(spc[4]) == 0
        sel = np.isin(ids, [int(x) for x in c[4:] if x])
        assert not np.any(fixed_group[sel])
        fixed_group[sel] = k + 1
        centers.append(points[center])
        bolts.append(
            dict(
                label=f"B{k + 1}",
                rbe2=int(c[1]),
                center_grid=center,
                physical_nodes=int(sel.sum()),
                center_mm=(np.array(points[center]) * 1000).tolist(),
                prescribed_center_dofs="ux=uy=uz=rx=ry=rz=0",
                prescribed_solid_node_dofs="ux=uy=uz=0",
            )
        )
    np.testing.assert_array_equal(fixed_group > 0, mesh["fixed"])
    rbe = next(c for c in records if c[0] == "RBE3")
    assert rbe[4] == "123456" and number(rbe[5]) == 1 and rbe[6] == "123"
    loaded = np.isin(ids, [int(x) for x in rbe[7:] if x])
    np.testing.assert_array_equal(loaded, np.any(mesh["forces"] != 0, axis=1))
    assert not np.any(loaded & (fixed_group > 0))
    center = int(rbe[3])
    force = next(c for c in records if c[0] == "FORCE" and c[1] == "2")
    assert int(force[2]) == center and force[3] == "0"
    resultant = number(force[4]) * np.array([number(x) for x in force[5:8]])
    np.testing.assert_allclose(mesh["forces"].sum(0), resultant, atol=1e-9)
    np.savez_compressed(
        root / "boundary-conditions.npz",
        vertices=mesh["vertices"],
        faces=mesh["faces"],
        fixed_group=fixed_group,
        loaded=loaded,
        grid_ids=ids,
        bolt_centers=np.array(centers),
        load_center=np.array(points[center]),
        resultant=resultant,
        forces=mesh["forces"],
    )
    loaded_norm = np.linalg.norm(mesh["forces"][loaded], axis=1)
    meta = dict(
        source="225.fem",
        source_sha256=SHA256,
        subcase=1,
        spc=1,
        load=2,
        geometry="Undeformed original tetrahedral boundary; no solution fields loaded",
        boundary_npz_sha256=hashlib.sha256(
            (root / "boundary-conditions.npz").read_bytes()
        ).hexdigest(),
        exact_fixed_mask_match=True,
        exact_loaded_mask_match=True,
        fixed_nodes=int(mesh["fixed"].sum()),
        loaded_nodes=int(loaded.sum()),
        bolts=bolts,
        load_interface=dict(
            rbe3=int(rbe[1]),
            center_grid=center,
            center_mm=(np.array(points[center]) * 1000).tolist(),
            resultant_n=resultant.tolist(),
            distribution="Equal-weight RBE3-style force/moment-preserving distribution",
            nodal_force_magnitude_range_n=[float(loaded_norm.min()), float(loaded_norm.max())],
        ),
        other_conditions="Remaining displacements are unknown. Other exposed surfaces have no prescribed traction. No gravity, pressure, contact, or whole-base clamp.",
        source_cards=[
            " ".join(c).strip() for c in records if c[0] == "SPC" or c[0] == "FORCE" and c[1] == "2"
        ],
    )
    (root / "boundary-conditions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

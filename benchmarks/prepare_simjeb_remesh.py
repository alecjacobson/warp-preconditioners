"""Transfer original SimJEB boundary regions and measure tetrahedron quality.

Boundary face centroids are matched to the original tagged surface triangles.
No coordinate-based bolt guesses or whole-base constraints are introduced.
"""

import argparse
import hashlib
import json
from pathlib import Path

import igl
import numpy as np
import warp as wp
from simjeb_problem import distribute


def boundary(t):
    faces = t[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, index, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    assert counts.max() == 2
    return faces[index[counts == 1]]


def areas(v, f):
    return np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2


def quality(v, t):
    points = v[t]
    edges = points[:, 1:] - points[:, :1]
    volume = np.abs(np.linalg.det(edges)) / 6
    length2 = np.stack(
        [
            np.sum((points[:, i] - points[:, j]) ** 2, axis=1)
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        ],
        axis=1,
    )
    meanratio = 12 * (3 * volume) ** (2 / 3) / length2.sum(1)
    normals = []
    area = []
    for i, j, k in [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]:
        n = np.cross(points[:, j] - points[:, i], points[:, k] - points[:, i])
        mag = np.linalg.norm(n, axis=1)
        normals.append(n / mag[:, None])
        area.append(mag / 2)
    angles = np.stack(
        [
            np.degrees(np.arccos(np.clip(-np.sum(normals[i] * normals[j], axis=1), -1, 1)))
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        ],
        axis=1,
    )
    circumcenter = np.linalg.solve(edges, 0.5 * np.sum(edges**2, axis=2)[..., None])[..., 0]
    radiusratio = 3 * (3 * volume / np.sum(area, axis=0)) / np.linalg.norm(circumcenter, axis=1)

    def stats(a):
        return {str(q): float(np.percentile(a, q)) for q in [0, 1, 5, 50, 95, 99, 100]}

    return (
        dict(
            vertices=len(v),
            tetrahedra=len(t),
            volume_mm3=float(volume.sum() * 1e9),
            mean_ratio=stats(meanratio),
            radius_ratio=stats(radiusratio),
            min_dihedral_degrees=stats(angles.min(1)),
            fraction_dihedral_below_10=float(np.mean(angles.min(1) < 10)),
            edge_length_mm=stats(np.sqrt(length2).ravel() * 1000),
        ),
        meanratio,
        angles.min(1),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path("data/simjeb-ftetwild"))
    args = p.parse_args()
    original = np.load("data/simjeb/mesh.npz")
    bc = np.load("data/simjeb/boundary-conditions.npz")
    oldmeta = json.loads(Path("data/simjeb/mesh.json").read_text())
    raw = np.load(args.data / "raw-mesh.npz")
    v, t = raw["vertices"], raw["tets"]
    # Remove any unused output vertices before assembling unknowns.
    used = np.unique(t)
    index = np.full(len(v), -1, dtype=np.int32)
    index[used] = np.arange(len(used))
    v, t = v[used], index[t]
    f = boundary(t)
    ov, of = original["vertices"], original["faces"]
    oldtags = np.zeros(len(of), dtype=int)
    for k in range(1, 5):
        oldtags[np.all(bc["fixed_group"][of] == k, axis=1)] = k
    oldtags[np.all(bc["loaded"][of], axis=1)] = 5
    distance, nearest, _ = igl.point_mesh_squared_distance(v[f].mean(1), ov, of)
    tags = oldtags[nearest]
    group = np.zeros(len(v), dtype=np.int32)
    for k in range(1, 5):
        group[np.unique(f[tags == k])] = k
    loaded = np.zeros(len(v), dtype=bool)
    loaded[np.unique(f[tags == 5])] = True
    fixed = group > 0
    assert not np.any(fixed & loaded)
    sourcearea, newarea = areas(ov, of), areas(v, f)
    patch = []
    for k in range(1, 6):
        olda = float(sourcearea[oldtags == k].sum())
        newa = float(newarea[tags == k].sum())
        assert olda > 0 and abs(newa / olda - 1) < 0.05, (k, olda, newa)
        patch.append(
            dict(
                label="pin" if k == 5 else f"B{k}",
                original_area_mm2=olda * 1e6,
                new_area_mm2=newa * 1e6,
                relative_area_change=newa / olda - 1,
                nodes=int(loaded.sum() if k == 5 else np.sum(group == k)),
            )
        )
    distance_v, _, _ = igl.point_mesh_squared_distance(v[np.unique(f)], ov, of)
    # Correspondence checks on the original surface in the other direction too.
    reverse, _, _ = igl.point_mesh_squared_distance(ov[np.unique(of)], v, f)
    points = v[loaded]
    forces = np.zeros_like(v)
    out = wp.empty(len(points), dtype=wp.vec3d, device="cuda:0")
    wp.launch(
        distribute,
        1,
        [
            wp.array(points, dtype=wp.vec3d, device="cuda:0"),
            wp.vec3d(points.mean(0)),
            wp.vec3d(bc["load_center"]),
            wp.vec3d(bc["resultant"]),
            out,
        ],
        device="cuda:0",
    )
    forces[loaded] = out.numpy()
    np.testing.assert_allclose(forces.sum(0), bc["resultant"], atol=1e-8)
    np.testing.assert_allclose(np.cross(v - bc["load_center"], forces).sum(0), 0, atol=1e-8)
    np.savez_compressed(
        args.data / "mesh.npz",
        vertices=v,
        tets=t,
        fixed=fixed,
        faces=f,
        forces=forces,
        fixed_group=group,
        loaded=loaded,
        boundary_tags=tags,
    )
    oq, _, _ = quality(ov, original["tets"])
    nq, ratio, angle = quality(v, t)
    np.savez_compressed(args.data / "quality.npz", mean_ratio=ratio, min_dihedral=angle)
    remesh = json.loads((args.data / "remesh.json").read_text())
    assert abs(nq["volume_mm3"] / oq["volume_mm3"] - 1) < 0.005
    assert np.sqrt(distance_v.max()) * 1000 < 2 * remesh["envelope_mm"]
    assert np.sqrt(reverse.max()) * 1000 < 2 * remesh["envelope_mm"]
    transfer = dict(
        method="Nearest original surface triangle to each new boundary-face centroid; transfer triangle tag, then constrain/load incident vertices",
        patches=patch,
        max_vertex_distance_to_original_mm=float(np.sqrt(distance_v.max()) * 1000),
        max_original_vertex_distance_to_new_mm=float(np.sqrt(reverse.max()) * 1000),
        max_centroid_distance_to_original_mm=float(np.sqrt(distance.max()) * 1000),
        load="Same reference point, resultant and zero reference moment; equal-weight distribution over new interface nodes",
        caveat="Remeshing changes discrete node positions and equal-weight RBE3 nodal sampling; surface regions and total force/moment are preserved, not individual original nodal forces.",
    )
    meta = dict(oldmeta)
    meta.update(
        name=f"SimJEB #225 — fTetWild {remesh['target_edge_mm']} mm",
        vertices=len(v),
        tetrahedra=len(t),
        fixed_vertices=int(fixed.sum()),
        loaded_vertices=int(loaded.sum()),
        free_dofs=int(3 * (~fixed).sum()),
        mesh_sha256=hashlib.sha256((args.data / "mesh.npz").read_bytes()).hexdigest(),
        remeshing=remesh,
        transfer=transfer,
    )
    (args.data / "mesh.json").write_text(json.dumps(meta, indent=2) + "\n")
    report = dict(original=oq, ftetwild=nq, transfer=transfer, meshing=remesh)
    (args.data / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

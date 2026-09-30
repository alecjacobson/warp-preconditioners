"""Import SimJEB #225's original linear tetrahedra and vertical load in SI units.

Geometry: Michael Jenkins, improved-ge-bracket-1. Dataset: Whalen et al.,
https://simjeb.github.io/ (ODC-By; original CAD subject to GrabCAD terms).
No external numerical solver is used. Load distribution runs in Warp.
"""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import warp as wp

URL = (
    "https://raw.githubusercontent.com/nora-alshareef/jeb-dataset-forge/"
    "e571104fc20c7cd34dc9be5b7f038b56fbef4981/data/raw/225.fem"
)
SHA256 = "1dc777d4e6d7a8f10aac7bf3994df849b8b38f7853cf3374ea46189e586c1b33"


def number(s):
    return float(re.sub(r"(?<=[\d.])([+-]\d+)$", r"e\1", s.strip()))


def cards(text):
    current = None
    for line in text.splitlines():
        if not line or line.startswith("$"):
            continue
        fields = [line[i : i + 8].strip() for i in range(0, 72, 8)]
        if fields[0] == "+":
            if current is not None:
                current.extend(fields[1:])
        else:
            if current is not None:
                yield current
            current = fields
    if current is not None:
        yield current


@wp.kernel
def distribute(
    points: wp.array(dtype=wp.vec3d),
    center: wp.vec3d,
    reference: wp.vec3d,
    force: wp.vec3d,
    out: wp.array(dtype=wp.vec3d),
):
    # Equal-weight rigid-motion least-squares interpolation, transposed to
    # distribute force. Preserves both resultant and moment about any origin.
    inertia = wp.mat33d(wp.float64(0))
    for j in range(points.shape[0]):
        r = points[j] - center
        inertia += wp.dot(r, r) * wp.identity(n=3, dtype=wp.float64) - wp.outer(r, r)
    omega = wp.inverse(inertia) * wp.cross(reference - center, force)
    for j in range(points.shape[0]):
        out[j] = force / wp.float64(points.shape[0]) + wp.cross(omega, points[j] - center)


def main():
    root = Path("data/simjeb")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "225.fem"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
    nodes, elements, fixed_ids = {}, [], []
    constraints = set()
    spiders = []
    for c in cards(path.read_text()):
        if c[0] == "GRID":
            assert not c[2] or int(c[2]) == 0
            nodes[int(c[1])] = [number(x) * 0.001 for x in c[3:6]]
        elif c[0] == "CTETRA":
            elements.append([int(x) for x in c[3:7]])
        elif c[0] == "RBE2":
            assert c[3] == "123456"
            spiders.append((int(c[2]), [int(x) for x in c[4:] if x]))
        elif c[0] == "RBE3":
            assert c[4] == "123456" and number(c[5]) == 1 and c[6] == "123"
            load_center = int(c[3])
            load_ids = [int(x) for x in c[7:] if x]
        elif c[0] == "SPC" and c[1] == "1":
            assert c[3] == "123456" and number(c[4]) == 0
            constraints.add(int(c[2]))
        elif c[0] == "MAT1":
            young, poisson, density = number(c[2]) * 1e6, number(c[4]), number(c[5]) * 1e12
        elif c[0] == "FORCE" and c[1] == "2":
            assert int(c[2]) == load_center and int(c[3]) == 0
            force = number(c[4]) * np.array([number(s) for s in c[5:8]])
    for center, ids in spiders:
        assert center in constraints
        fixed_ids.extend(ids)
    ids = np.unique(elements)
    remap = {int(n): i for i, n in enumerate(ids)}
    v = np.array([nodes[int(n)] for n in ids])
    t = np.array([[remap[n] for n in e] for e in elements], dtype=np.int32)
    det = np.linalg.det((v[t[:, 1:]] - v[t[:, :1]]).transpose(0, 2, 1))
    assert np.all(np.abs(det) > 1e-18)
    flip = det < 0
    t[flip, :2] = t[flip, 1::-1]
    fixed = np.isin(ids, fixed_ids)
    surface = t[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first, counts = np.unique(
        np.sort(surface, axis=1), axis=0, return_index=True, return_counts=True
    )
    assert counts.max() == 2
    faces = surface[first[counts == 1]]
    load = np.array([remap[n] for n in load_ids], dtype=np.int32)
    points = v[load]
    out = wp.empty(len(load), dtype=wp.vec3d, device="cuda:0")
    wp.launch(
        distribute,
        1,
        [
            wp.array(points, dtype=wp.vec3d, device="cuda:0"),
            wp.vec3d(points.mean(0)),
            wp.vec3d(nodes[load_center]),
            wp.vec3d(force),
            out,
        ],
        device="cuda:0",
    )
    forces = np.zeros_like(v)
    forces[load] = out.numpy()
    np.testing.assert_allclose(forces.sum(0), force, atol=1e-9)
    np.testing.assert_allclose(np.cross(v - nodes[load_center], forces).sum(0), 0, atol=1e-9)
    assert not np.any(fixed[load])
    np.savez_compressed(
        root / "mesh.npz", vertices=v, tets=t, fixed=fixed, faces=faces, forces=forces
    )
    meta = dict(
        name="SimJEB #225 — Michael Jenkins",
        source=URL,
        source_sha256=SHA256,
        dataset="https://simjeb.github.io/",
        design="https://grabcad.com/library/improved-ge-bracket-1",
        vertices=len(v),
        tetrahedra=len(t),
        fixed_vertices=int(fixed.sum()),
        free_dofs=int(3 * (~fixed).sum()),
        mesh_sha256=hashlib.sha256((root / "mesh.npz").read_bytes()).hexdigest(),
        young_pa=young,
        poisson=poisson,
        density_kg_m3=density,
        load_force_n=force.tolist(),
        load_reference_m=nodes[load_center],
        loaded_vertices=len(load),
        boundary="All translations fixed on four RBE2 bolt-hole surfaces",
        load="Original vertical FORCE; equal-weight rigid-motion least-squares distribution over RBE3 nodes, preserving force and moment",
        units="m, N, Pa; original deck in mm, N, MPa",
        disclaimer="Reproduction with linear tetrahedra, not a validation of OptiStruct RBE3 internal implementation",
    )
    (root / "mesh.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

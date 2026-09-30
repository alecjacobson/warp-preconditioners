"""Prepare a watertight, simplified dragon and a quality TetGen volume mesh."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import igl
import numpy as np
import tetgen


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--surface", type=Path, default=Path("data/dirichlet-tuned/fields.npz"))
    p.add_argument("--output", type=Path, default=Path("data/elasticity"))
    p.add_argument("--faces", type=int, default=60000)
    p.add_argument("--max-volume", type=float, default=1e-6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.surface.suffix == ".npz":
        d = np.load(args.surface)
        v, f = d["vertices"], d["faces"]
    else:
        v, f = igl.read_triangle_mesh(str(args.surface))
    original_hash = hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()
    start = time.perf_counter()
    print("Input", v.shape, f.shape, flush=True)
    cache = args.output / "surface.npz"
    if cache.exists():
        d = np.load(cache)
        assert str(d["source_sha256"]) == original_hash and int(d["target_faces"]) == args.faces
        v, f = d["vertices"], d["faces"]
    else:
        v, f, _, _ = igl.decimate(v, f.astype(np.int32), args.faces, True)
        np.savez_compressed(
            cache, vertices=v, faces=f, source_sha256=original_hash, target_faces=args.faces
        )
    print("Simplified", v.shape, f.shape, "seconds", time.perf_counter() - start, flush=True)
    # Use SI coordinates with a one-metre long dragon. Preserve the old patch selections.
    scale = 1.0 / np.ptp(v[:, 0])
    mesher = tetgen.TetGen(np.ascontiguousarray(v * scale), np.ascontiguousarray(f, dtype=np.int32))
    tv, t, _, _ = mesher.tetrahedralize(
        minratio=1.4,
        mindihedral=8,
        fixedvolume=True,
        maxvolume=args.max_volume,
        steinerleft=-1,
        quiet=False,
    )
    t = t.astype(np.int32)
    edges = tv[t[:, 1:]] - tv[t[:, :1]]
    det = np.linalg.det(edges)
    negative = det < 0
    t[negative, :2] = t[negative, 1::-1]
    assert np.all(np.abs(det) > 1e-18)
    faces = igl.boundary_facets(t)[0].astype(np.int32)
    boundary = np.unique(faces)
    fixed = np.zeros(len(tv), dtype=bool)
    raw = tv / scale
    fixed[boundary] = ((raw[boundary, 0] < -70) & (raw[boundary, 2] > 45)) | (
        (raw[boundary, 0] > 70) & (raw[boundary, 2] > 55)
    )
    assert fixed.sum() > 0
    np.savez_compressed(
        args.output / "mesh.npz",
        vertices=tv,
        tets=t,
        faces=faces,
        fixed=fixed,
        original_scale=scale,
    )
    meta = dict(
        source=str(args.surface),
        source_sha256=original_hash,
        tetgen=tetgen.__version__,
        simplified_vertices=len(v),
        simplified_faces=len(f),
        vertices=len(tv),
        tetrahedra=len(t),
        boundary_faces=len(faces),
        fixed_vertices=int(fixed.sum()),
        free_dofs=3 * int((~fixed).sum()),
        length_metres=float(np.ptp(tv[:, 0])),
        max_volume=args.max_volume,
        minratio=1.4,
        mindihedral=8,
        volume=float(np.abs(det).sum() / 6),
        minimum_tet_volume=float(np.abs(det).min() / 6),
        mesh_sha256=hashlib.sha256(tv.tobytes() + t.tobytes() + fixed.tobytes()).hexdigest(),
        seconds=time.perf_counter() - start,
    )
    (args.output / "mesh.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

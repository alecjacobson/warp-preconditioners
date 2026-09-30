"""Remesh the original SimJEB surface with fTetWild; save raw tets for audit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import wildmeshing as wm


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--edge-mm", type=float, default=2.0)
    p.add_argument("--epsilon-mm", type=float, default=0.02)
    p.add_argument("--output", type=Path, default=Path("data/simjeb-ftetwild"))
    args = p.parse_args()
    source = Path("data/simjeb/mesh.npz")
    mesh = np.load(source)
    v, f = mesh["vertices"], mesh["faces"]
    used = np.unique(f)
    index = np.full(len(v), -1, dtype=np.int32)
    index[used] = np.arange(len(used))
    v, f = v[used], index[f]
    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    options = dict(
        stop_quality=10,
        max_its=80,
        max_threads=4,
        epsilon=args.epsilon_mm * 0.001 / diagonal,
        edge_length_r=args.edge_mm * 0.001 / diagonal,
        skip_simplify=False,
        coarsen=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # fTetWild writes a diagnostic STL in its working directory.
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output = args.output.resolve()
    os.chdir(args.output)
    start = time.perf_counter()
    tetra = wm.Tetrahedralizer(**options)
    tetra.set_mesh(v, f.astype(np.int32))
    tetra.tetrahedralize()
    result = tetra.get_tet_mesh()
    vertices, tets = result[:2]
    print(
        "output",
        vertices.shape,
        tets.shape,
        [getattr(x, "shape", None) for x in result],
        flush=True,
    )
    det = np.linalg.det((vertices[tets[:, 1:]] - vertices[tets[:, :1]]).transpose(0, 2, 1))
    assert np.all(np.abs(det) > 1e-20)
    tets = tets.astype(np.int32)
    flip = det < 0
    tets[flip, :2] = tets[flip, 1::-1]
    np.savez_compressed(args.output / "raw-mesh.npz", vertices=vertices, tets=tets)
    meta = dict(
        mesher="fTetWild via wildmeshing",
        version=importlib.metadata.version("wildmeshing"),
        options=options,
        target_edge_mm=args.edge_mm,
        envelope_mm=args.epsilon_mm,
        source_sha256=source_hash,
        seconds=time.perf_counter() - start,
        vertices=len(vertices),
        tetrahedra=len(tets),
        stats=tetra.get_stats(),
    )
    (args.output / "remesh.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)


if __name__ == "__main__":
    main()

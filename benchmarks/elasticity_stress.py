"""Recover nodal von Mises stress from the saved physical displacements in Warp.

Volume-average the element stress tensors at each vertex, then evaluate
sqrt(3/2 dev(sigma):dev(sigma)). This is visualization postprocessing only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from elasticity_problem import mat43d


@wp.kernel(enable_backward=False)
def accumulate_stress(
    vertices: wp.array(dtype=wp.vec3d),
    tets: wp.array(dtype=wp.vec4i),
    u: wp.array(dtype=wp.vec3d),
    lame: wp.float64,
    mu: wp.float64,
    stress_sum: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    element_vm: wp.array(dtype=wp.float64),
):
    e = wp.tid()
    t = tets[e]
    p = vertices[t[0]]
    dm = wp.matrix_from_cols(vertices[t[1]] - p, vertices[t[2]] - p, vertices[t[3]] - p)
    inv = wp.inverse(dm)
    g = mat43d(wp.float64(0))
    g[0] = -(inv[0] + inv[1] + inv[2])
    g[1] = inv[0]
    g[2] = inv[1]
    g[3] = inv[2]
    grad = wp.mat33d(wp.float64(0))
    for i in range(4):
        grad += wp.outer(u[t[i]], g[i])
    stress = mu * (grad + wp.transpose(grad)) + lame * wp.trace(grad) * wp.identity(
        n=3, dtype=wp.float64
    )
    mean = wp.trace(stress) / wp.float64(3)
    norm_sq = wp.float64(0)
    for a in range(3):
        for b in range(3):
            value = stress[a, b]
            if a == b:
                value -= mean
            norm_sq += value * value
    element_vm[e] = wp.sqrt(wp.float64(1.5) * norm_sq)
    volume = wp.abs(wp.determinant(dm)) / wp.float64(6)
    for i in range(4):
        wp.atomic_add(weights, t[i], volume)
        for a in range(3):
            for b in range(3):
                wp.atomic_add(stress_sum, t[i], 3 * a + b, volume * stress[a, b])


@wp.kernel(enable_backward=False)
def nodal_von_mises(
    stress_sum: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    inv = wp.float64(1) / weights[i]
    mean = (stress_sum[i, 0] + stress_sum[i, 4] + stress_sum[i, 8]) * inv / wp.float64(3)
    norm_sq = wp.float64(0)
    for a in range(3):
        for b in range(3):
            value = stress_sum[i, 3 * a + b] * inv
            if a == b:
                value -= mean
            norm_sq += value * value
    output[i] = wp.sqrt(wp.float64(1.5) * norm_sq)


def recover(vertices, tets, displacement, young, poisson, device="cuda:0"):
    with wp.ScopedDevice(device):
        sums = wp.zeros((len(vertices), 9), dtype=wp.float64)
        weights = wp.zeros(len(vertices), dtype=wp.float64)
        element = wp.empty(len(tets), dtype=wp.float64)
        out = wp.empty(len(vertices), dtype=wp.float64)
        wp.launch(
            accumulate_stress,
            len(tets),
            [
                wp.array(vertices, dtype=wp.vec3d),
                wp.array(tets, dtype=wp.vec4i),
                wp.array(displacement, dtype=wp.vec3d),
                young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
                young / (2 * (1 + poisson)),
                sums,
                weights,
                element,
            ],
        )
        wp.launch(nodal_von_mises, len(vertices), [sums, weights, out])
    return out.numpy(), element.numpy()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/simjeb"))
    parser.add_argument("--output", type=Path, default=Path("results/elasticity.json"))
    args = parser.parse_args()
    root = args.data
    fields = dict(np.load(root / "comparison.npz"))
    mesh = np.load(root / "mesh.npz")
    path = args.output
    meta = json.loads(path.read_text())
    stress = {}
    for name in ["reference", "jacobi", "block", "fsai"]:
        nodal, element = recover(
            fields["vertices"], mesh["tets"], fields[name], meta["young_pa"], meta["poisson"]
        )
        assert np.all(np.isfinite(nodal)) and np.all(nodal >= 0)
        fields["von_mises_" + name] = nodal
        stress[name] = dict(
            nodal_max_pa=float(nodal.max()),
            element_max_pa=float(element.max()),
            nodal_sha256=hashlib.sha256(nodal.tobytes()).hexdigest(),
        )
        print(name, stress[name], flush=True)
    meta["stress"] = dict(
        recovery="Volume-average element Cauchy stress tensors at vertices, then compute von Mises invariant",
        units="Pa",
        displacement_amplification=1,
        fields=stress,
    )
    np.savez_compressed(root / "comparison.npz", **fields)
    path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

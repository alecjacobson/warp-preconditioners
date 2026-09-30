"""Validate smoothing fields with a long-double factored-gradient refinement."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sksparse.cholmod import cholesky


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/visualization-tuned"))
    parser.add_argument("--original", type=Path, default=Path("data/visualization"))
    parser.add_argument("--dump", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--output", type=Path, default=Path("results/smoothing-verification.json"))
    args = parser.parse_args()
    d = np.load(args.data / "fields.npz")
    meta = json.loads((args.data / "solve.json").read_text())
    n = len(d["vertices"])
    mixed = sio.mmread(args.dump / "k4_Q.mtx").tocsr()
    mass = mixed.diagonal()[:n].astype(np.longdouble)
    L = (-mixed[:n, n:]).astype(np.longdouble)
    weight = np.longdouble(meta["data_weight"])
    target = d["target"].astype(np.longdouble)
    a = L.T @ ss.diags(1 / mass) @ L + ss.diags(weight * mass)
    scale = 1 / np.sqrt(a.diagonal())
    factor = cholesky(
        (ss.diags(scale) @ a @ ss.diags(scale)).astype(np.float64).tocsc(), ordering_method="amd"
    )

    def solve(rhs):
        return scale * factor(np.asarray(scale * rhs, dtype=np.float64)).astype(np.longdouble)

    u = solve(weight * mass * target)
    corrections = []
    for _ in range(6):
        dx = solve(weight * mass * (target - u) - L.T @ ((L @ u) / mass))
        u += dx
        corrections.append(float(np.sqrt(np.sum(mass * dx**2) / np.sum(mass * u**2))))
    assert corrections[-1] < 1e-12, corrections

    def error(v):
        return float(np.sqrt(np.sum(mass * (v - u) ** 2) / np.sum(mass * u**2)))

    original = np.load(args.original / "fields.npz")
    original_meta = json.loads((args.original / "solve.json").read_text())
    np.testing.assert_array_equal(d["target"], original["target"])
    np.testing.assert_array_equal(d["vertices"], original["vertices"])
    np.testing.assert_array_equal(d["faces"], original["faces"])
    report = dict(
        reference="Long-double factored energy gradient, equilibrated float64 AMD Cholesky corrections",
        long_double_epsilon=float(np.finfo(np.longdouble).eps),
        relative_mass_norm_corrections=corrections,
        tuned_fields={name: error(d[name]) for name in meta["display_order"]},
        original_fields={name: error(original[name]) for name in original_meta["display_order"]},
        unchanged_inputs="Target, vertices and triangles are bitwise identical to original smoothing figure",
    )
    assert report["tuned_fields"][meta["display_order"][0]] < 1e-6, report
    np.save(args.data / "refined_reference.npy", np.asarray(u, dtype=np.float64))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

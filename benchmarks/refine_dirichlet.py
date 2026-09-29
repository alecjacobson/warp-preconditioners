"""Higher-precision energy reference using Cholesky and long-double refinement."""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sksparse.cholmod import cholesky


def main():
    p = Path("data/dirichlet")
    d = np.load(p / "fields.npz")
    n = len(d["vertices"])
    free = d["free"]
    mixed = sio.mmread("/tmp/dump/k4_Q.mtx").tocsr()
    L = (-mixed[:n, n:]).astype(np.longdouble)
    mass = d["mass"].astype(np.longdouble)
    initial = d["constraints"].astype(np.longdouble)
    Q = L.T @ ss.diags(1 / mass) @ L
    a = Q[free][:, free].tocsr()
    b = -(Q @ initial)[free]
    scale = 1 / np.sqrt(a.diagonal())
    ahat = ss.diags(scale) @ a @ ss.diags(scale)
    factor = cholesky(ahat.astype(np.float64).tocsc(), ordering_method="amd")

    def solve(rhs):
        return scale * factor(np.asarray(scale * rhs, dtype=np.float64)).astype(np.longdouble)

    x = solve(b)
    records = []
    for step in range(6):
        full = initial.copy()
        full[free] = x
        residual = -(L.T @ ((L @ full) / mass))[free]
        dx = solve(residual)
        x += dx
        full = initial.copy()
        full[free] = x
        correction = float(np.sqrt(np.sum(mass[free] * dx**2) / np.sum(mass * full**2)))
        records.append(dict(step=step + 1, mass_relative_correction=correction))
        print(records[-1], flush=True)
    full = initial.copy()
    full[free] = x
    reference = np.asarray(full, dtype=np.float64)
    old = np.load(p / "cholesky.npy")

    def error(u):
        return float(np.sqrt(np.sum(mass * (u - full) ** 2) / np.sum(mass * full**2)))

    result = dict(
        arithmetic="L and M promoted before assembly; long-double factored energy gradient L.T((Lu)/mass) and updates; diagonally equilibrated float64 AMD Cholesky corrections",
        long_double_epsilon=float(np.finfo(np.longdouble).eps),
        iterations=records,
        original_cholesky_mass_relative_error=error(old),
        energy=float(0.5 * np.sum((L @ full) ** 2 / mass)),
        range=[float(full.min()), float(full.max())],
        original_fields={
            name: error(d[name])
            for name in json.loads((p / "solve.json").read_text())["display_order"]
        },
    )
    np.save("data/tuning/refined_reference.npy", reference)
    # This RHS is for matrix-free experiments; it does not overwrite the benchmark RHS.
    np.save("data/tuning/refined_rhs.npy", np.asarray(b, dtype=np.float64))
    Path("data/tuning/refinement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

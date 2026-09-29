"""Measure generalized spectral endpoints with AMD Cholesky shift-invert."""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import cholesky
from tune_dirichlet import load_problem


def main():
    a, _, mass, _, _, _ = load_problem()
    m = ss.diags(mass)
    factor = cholesky(a.tocsc(), ordering_method="amd")
    inv = LinearOperator(a.shape, matvec=factor, dtype=np.float64)
    low, u = eigsh(a, k=1, M=m, sigma=0, which="LM", OPinv=inv, tol=1e-8)
    high, _ = eigsh(a, k=1, M=m, which="LA", tol=1e-8)
    d = np.load("data/dirichlet/fields.npz")
    n = len(d["vertices"])
    mixed = sio.mmread("/tmp/dump/k4_Q.mtx").tocsr()
    L = -mixed[:n, n:]
    full = np.zeros(n)
    full[d["free"]] = u[:, 0]
    rayleigh = float(np.sum((L @ full) ** 2 / d["mass"]) / np.sum(mass * u[:, 0] ** 2))
    residual = a @ u[:, 0] - low[0] * mass * u[:, 0]
    backward = float(
        np.max(
            abs(residual)
            / np.maximum(abs(a) @ abs(u[:, 0]) + abs(low[0] * mass * u[:, 0]), np.finfo(float).tiny)
        )
    )
    result = dict(
        generalized_lambda_min=float(low[0]),
        lambda_min_energy_rayleigh=rayleigh,
        generalized_lambda_max=float(high[0]),
        min_eigenvector_backward_error=backward,
        method="ARPACK generalized eigsh; smallest via sigma=0 Cholesky-AMD shift-invert, largest algebraic",
        shifts=[
            dict(alpha=alpha, condition_estimate=float((high[0] + alpha) / (rayleigh + alpha)))
            for alpha in [1.0, 0.01, 0.0001, 0.000001, 0.0]
        ],
    )
    Path("data/tuning/spectrum.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

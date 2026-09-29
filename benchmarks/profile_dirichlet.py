"""Repeated graph-replay timings for A, factors, and complete solver iterations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from dragon import upload
from ell_candidate import ELLFSAI
from tune_dirichlet import load_problem, permutation
from warp.optim import linear

from warp_preconditioners import FSAI


def measure(fn, count=100, repeats=5):
    fn()
    wp.synchronize()
    with wp.ScopedCapture(device="cuda:0") as cap:
        for _ in range(count):
            fn()
    wp.capture_launch(cap.graph)
    wp.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        times.append((time.perf_counter() - start) / count * 1e6)
    return dict(median_us=float(np.median(times)), samples_us=times)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--ordering", default="natural")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b_np, _, _, _, v = load_problem()
    order = permutation(a, v, args.ordering)
    a = a[order][:, order].tocsr()
    A = upload(a, "cuda:0")
    b = wp.array(b_np[order], dtype=wp.float64, device="cuda:0")
    x = wp.ones_like(b)
    y = wp.zeros_like(b)
    z = wp.zeros_like(b)
    csr = FSAI(A, max_row_size=args.width)
    ell = ELLFSAI(A, max_row_size=args.width)
    jacobi = linear.preconditioner(A, "diag")
    csr.matvec(x, y, y, 1.0, 0.0)
    ell.matvec(x, z, z, 1.0, 0.0)
    np.testing.assert_allclose(z.numpy(), y.numpy(), rtol=1e-12, atol=1e-12)
    records = {}
    for name, fn in [
        ("A_csr", lambda: sp.bsr_mv(A, x, y)),
        ("G_csr", lambda: sp.bsr_mv(csr.G, x, y)),
        ("GT_csr", lambda: sp.bsr_mv(csr.GT, x, y)),
        ("FSAI_csr", lambda: csr.matvec(x, y, y, 1.0, 0.0)),
        ("FSAI_ell", lambda: ell.matvec(x, y, y, 1.0, 0.0)),
        ("Jacobi", lambda: jacobi.matvec(x, y, y, 1.0, 0.0)),
    ]:
        records[name] = measure(fn)
        print(name, records[name], flush=True)
    for name, solver, pre in [
        ("csr_cg", linear.cg, csr),
        ("ell_cg", linear.cg, ell),
        ("jacobi_cg", linear.cg, jacobi),
        ("jacobi_cr", linear.cr, jacobi),
    ]:
        solver(A, b, x, M=pre, tol=0.0, atol=0.0, maxiter=10, check_every=0)
        samples = []
        for _ in range(3):
            x.zero_()
            wp.synchronize()
            start = time.perf_counter()
            solver(A, b, x, M=pre, tol=0.0, atol=0.0, maxiter=5000, check_every=0)
            wp.synchronize()
            samples.append((time.perf_counter() - start) / 5000 * 1e6)
        records[name] = dict(median_us=float(np.median(samples)), samples_us=samples)
        print(name, records[name], flush=True)
    args.output.write_text(
        json.dumps(dict(width=args.width, ordering=args.ordering, results=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

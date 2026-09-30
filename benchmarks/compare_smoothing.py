"""Repeated original/tuned FSAI measurements on the smoothing figure's exact data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualization"))
from solve_fields import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/visualization"))
    parser.add_argument("--dump", type=Path, default=Path("/tmp/dump"))
    parser.add_argument("--output", type=Path, default=Path("results/smoothing-tuning.json"))
    args = parser.parse_args()
    d = np.load(args.data / "fields.npz")
    meta = json.loads((args.data / "solve.json").read_text())
    n = len(d["vertices"])
    mixed = sio.mmread(args.dump / "k4_Q.mtx").tocsr()
    mass, L = mixed.diagonal()[:n], -mixed[:n, n:]
    weight = meta["data_weight"]
    a = sio.mmread(args.dump / "k2_Q.mtx").tocsr() + ss.diags((weight - 1) * mass)
    b = weight * mass * d["target"]
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    records = []
    fields = {}
    for trial in range(3):
        for tuned in [False, True] if trial % 2 == 0 else [True, False]:
            u, record = run(
                a,
                b,
                "fsai_cg",
                100000,
                rtol=meta["convergence"]["fsai_rtol"],
                initial=d["target"],
                tuned=tuned,
                laplacian=L,
                mass=mass,
                data_weight=weight,
            )
            assert record["reached_stopping_tolerance"], record
            record.update(trial=trial, tuned=tuned, total_s=record["setup_s"] + record["solve_s"])
            record["factored_relative_residual"] = float(
                np.linalg.norm(b - (L.T @ ((L @ u) / mass) + weight * mass * u)) / np.linalg.norm(b)
            )
            fields["tuned" if tuned else "original"] = u
            records.append(record)
            print(json.dumps(record), flush=True)
    discrepancy = float(
        np.sqrt(
            np.sum(mass * (fields["tuned"] - fields["original"]) ** 2)
            / np.sum(mass * fields["tuned"] ** 2)
        )
    )
    report = dict(
        warp=wp.__version__,
        gpu=wp.get_device().name,
        target=meta["target"],
        data_weight=weight,
        initial_guess="data f",
        repeats=3,
        timing="Warmed single RHS; setup includes operator and preconditioner; matrix upload excluded for both",
        mass_relative_difference_between_configurations=discrepancy,
        results=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FIELD DIFFERENCE", discrepancy, flush=True)


if __name__ == "__main__":
    main()

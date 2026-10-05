"""Check whether wider supports reverse the strongest elasticity comparisons.

Reuses the guarded benchmark protocol, with paired scalar-FSAI controls and
larger scalar/block supports. This is an additional tuning check, not a new
preconditioner implementation.
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import block_challenge as challenge
import warp as wp


def configurations(quick=False):
    return [
        dict(family="fsai", width=w, step=s) for w, s in [(6, 1), (12, 1), (24, 1), (24, 2)]
    ] + [dict(family="block_fsai", width=w) for w in [4, 8]]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--directions", nargs="+", default=["aligned", "varying"])
    p.add_argument("--output", type=Path, default=Path("results/block-wide-check.json"))
    args = p.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    challenge.configurations = configurations
    args.problem, args.mesh, args.grid = "elasticity", "simjeb-ftetwild", 32
    args.quick, args.updates, args.solvers = False, False, ["cg", "cr"]
    args.target, args.limit, args.repeats = 1e-4, 30000, 5
    paths = [
        Path(__file__),
        Path(challenge.__file__),
        Path("benchmarks/block_fsai_candidate.py"),
        Path("benchmarks/elasticity_problem.py"),
        Path("warp_preconditioners/fsai.py"),
    ]
    output = dict(
        problem=args.problem,
        cases=[],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        provenance=dict(
            utc=datetime.now(timezone.utc).isoformat(),
            warp=wp.__version__,
            gpu=wp.get_device().name,
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            protocol="Same guarded warm setup+solve and independently verified error protocol as block_challenge.py; five interleaved trials",
        ),
    )
    for direction in args.directions:
        challenge.run_case(args, 1000.0, direction, output)


if __name__ == "__main__":
    main()

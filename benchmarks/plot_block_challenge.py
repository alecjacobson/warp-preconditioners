"""Plot verified block-challenge results; PNG assets are tracked through Git LFS."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["jacobi", "block_jacobi", "fsai", "block_fsai"]
LABELS = ["Scalar Jacobi", "Block Jacobi", "Scalar FSAI", "Block FSAI"]
COLORS = ["#777777", "#0072B2", "#D55E00", "#009E73"]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--directions", nargs=2, default=["aligned", "rotated"])
    args = p.parse_args()
    docs = [json.loads(path.read_text()) for path in args.inputs]
    cases = [case for d in docs for case in d["cases"]]
    elastic = docs[0]["problem"] == "elasticity"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, direction in zip(axes[:2], args.directions):
        selected = sorted(
            [c for c in cases if c["direction"] == direction], key=lambda c: c["contrast"]
        )
        for family, label, color in zip(FAMILIES, LABELS, COLORS):
            xx, yy, low, high = [], [], [], []
            for case in selected:
                name = case["winners"].get(family)
                if name is None:
                    continue
                times = np.array([t["total_s"] for t in case["candidates"][name]["trials"]]) * 1000
                xx.append(case["contrast"])
                yy.append(np.median(times))
                low.append(times.min())
                high.append(times.max())
            ax.plot(xx, yy, "o-", color=color, label=label, lw=2)
            ax.fill_between(xx, low, high, color=color, alpha=0.12)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Reinforcement contrast" if elastic else "Component contrast κ",
            ylabel="Setup + solve (ms)",
            title=direction.capitalize() + (" fibers" if elastic else " components"),
        )
        ax.grid(alpha=0.2, which="both")
    hardest = max(
        [c for c in cases if c["direction"] == args.directions[1]], key=lambda c: c["contrast"]
    )
    ax = axes[2]
    for family, label, color in zip(FAMILIES, LABELS, COLORS):
        name = hardest["winners"].get(family)
        # Also show a failed family: its final error matters, not just winners.
        if name is None:
            valid = [
                (n, c) for n, c in hardest["candidates"].items() if c["config"]["family"] == family
            ]
            name = min(valid, key=lambda pair: pair[1]["samples"][-1]["field_error"])[0]
        c = hardest["candidates"][name]
        points = c["samples"]
        ax.plot(
            [s["iteration"] + 1 for s in points],
            [s["field_error"] for s in points],
            color=color,
            lw=2,
            label=label + " + " + c["solver"].upper(),
        )
    ax.axhline(hardest["target"], color="black", linestyle=":", lw=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Iterations + 1 (includes the zero start)",
        ylabel="Relative displacement error" if elastic else "Relative solution error",
        title=f"{args.directions[1].capitalize()}, contrast {hardest['contrast']:g}",
    )
    ax.grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    title = (
        "Reinforced SimJEB bracket · fTetWild mesh"
        if elastic
        else "Controlled block coupling · A = K ⊗ C · 98,304 unknowns"
    )
    fig.suptitle(
        title
        + "\nL40 / pure Warp · best tested settings · medians and min–max bands · verified errors",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()

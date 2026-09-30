"""Plot measured relative convergence and repeat timings under force scaling."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads(Path("results/simjeb-load-scaling.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.23, top=0.77, wspace=0.28)
    colors = dict(jacobi="#D17632", block="#8056B3", fsai="#087E8B")
    labels = dict(jacobi="Scalar Jacobi + CG", block="Block Jacobi + CG", fsai="FSAI + CG")
    for family, color in colors.items():
        for case, style in [(data["cases"][0], "-"), (data["cases"][-1], "--")]:
            samples = [r for r in case["methods"][family]["samples"] if r["iteration"] > 0]
            for axis, key in zip(axes[:2], ["mass_error", "energy_error"]):
                axis.loglog(
                    [r["iteration"] for r in samples],
                    [r[key] for r in samples],
                    color=color,
                    ls=style,
                    lw=2 if style == "-" else 1.4,
                    label=labels[family] if style == "-" else None,
                )
        values = np.array(
            [
                [c["methods"][family][k] * 1000 for k in ["total_s", "total_s_min", "total_s_max"]]
                for c in data["cases"]
            ]
        )
        axes[2].errorbar(
            data["scales"],
            values[:, 0],
            yerr=np.vstack((values[:, 0] - values[:, 1], values[:, 2] - values[:, 0])),
            color=color,
            marker="o",
            capsize=4,
            lw=1.8,
            label=labels[family],
        )
    for axis, key in zip(axes[:2], ["Displacement", "Energy"]):
        axis.set_xlabel("Iterations")
        axis.set_ylabel(f"Relative {key.lower()} error")
        axis.axhline(data["target"], color="#8894A4", ls=":", lw=1)
        axis.set_ylim(1e-6, 2)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    axes[2].set_xlabel("Force multiplier")
    axes[2].set_ylabel("Setup + solve time to target (ms)")
    axes[2].set_xticks(data["scales"])
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(
        0.065,
        0.94,
        "Larger forces do not separate these linear-elasticity solvers",
        fontsize=22,
        weight="bold",
        color="#263448",
    )
    fig.text(
        0.065,
        0.875,
        "SimJEB #225 · fTetWild · NVIDIA L40 · Same stiffness, supports, preconditioner settings and zero initial guess",
        color="#657083",
    )
    fig.text(
        0.065,
        0.115,
        "Left / middle: 1× solid and 10× dashed nearly coincide. Right: median of five interleaved repeats; whiskers show min–max.",
        color="#657083",
    )
    counts = " · ".join(
        f"{labels[k]}: {data['cases'][0]['methods'][k]['iterations']:,}" for k in colors
    )
    fig.text(
        0.065,
        0.065,
        "Iterations to the sampled 10⁻⁴ target at every load: " + counts,
        color="#657083",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.02,
        "Both displacement and energy must meet the relative target. This experiment scales the load of a fixed small-strain model.",
        color="#657083",
        fontsize=10,
    )
    fig.savefig("assets/simjeb-load-scaling.png", dpi=190, facecolor="white")


if __name__ == "__main__":
    main()

"""Compare adaptive rebuilds, fixed-support refits and stale factors."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads(Path("results/numerical-updates-w8.json").read_text())
    methods = ["fresh", "refit", "stale", "block_update"]
    labels = ["Fresh FSAI", "Refit FSAI", "Stale FSAI", "Updated block Jacobi"]
    colors = ["#4176B6", "#078878", "#DC843B", "#8056B3"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.76, wspace=0.3)
    steps = np.arange(1, len(data["sequence"]) + 1)
    for name, label, color in zip(methods, labels, colors):
        rows = [s["methods"][name] for s in data["sequence"]]
        if name != "stale":
            axes[0].semilogy(
                steps, [1000 * r["setup_s"] for r in rows], "-o", color=color, label=label, ms=3
            )
        axes[1].plot(steps, [r["iterations"] for r in rows], "-o", color=color, label=label, ms=3)
        initial = data["totals"][name]["initial_setup_s"]
        axes[2].plot(
            steps,
            initial + np.cumsum([r["total_s"] for r in rows]),
            "-o",
            color=color,
            label=label,
            ms=3,
        )
        if name == "refit":
            axes[2].plot(
                steps,
                initial + np.cumsum([r["total_s"] + r["quality_s"] for r in rows]),
                "--",
                color=color,
                lw=1,
                label="Refit + quality check",
            )
    for ax in axes:
        ax.grid(color="#E4E9ED", lw=0.6)
        ax.set_xlabel("Matrix in sequence")
        ax.set_xticks([1, 3, 5, 7, 10])
    axes[0].set_ylabel("Numerical setup / update (ms)")
    axes[1].set_ylabel("CG iterations to relative tolerance 1e-8")
    axes[2].set_ylabel("Cumulative setup + solve time (s)")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.06,
        0.94,
        "Reusing FSAI supports as material stiffness changes",
        fontsize=21,
        weight="bold",
        color="#263448",
    )
    fig.text(
        0.06,
        0.865,
        "SimJEB #225 · width 8 · float64 factors · NVIDIA L40 · identical BSR sparsity throughout",
        fontsize=11,
        color="#66758A",
    )
    fig.text(
        0.06,
        0.10,
        "Ten spatially varying modulus fields, ending at 263× contrast. Refits retain nearly the same iterations as fresh adaptive builds.",
        color="#66758A",
    )
    fig.text(
        0.06,
        0.045,
        "Solve times: medians of three runs from zero. Cumulative totals include the first build; assembly, uploads and JIT are excluded.",
        color="#66758A",
    )
    fig.savefig("assets/simjeb-numerical-updates.png", dpi=190, facecolor="white")


if __name__ == "__main__":
    main()

"""Render a log-log convergence plot from measured, unrestarted Warp trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/dirichlet-residuals.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/dragon-dirichlet-residuals.png")
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert set(data["results"]) == {"fsai_cg", "fsai_cr", "jacobi_cr", "jacobi_cg"}, (
        "All four completed trajectories are required"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263448",
            "text.color": "#263448",
            "xtick.color": "#526174",
            "ytick.color": "#526174",
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 7.8), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.81)
    colors = {
        "fsai_cg": "#087E8B",
        "fsai_cr": "#3267C5",
        "jacobi_cr": "#D17632",
        "jacobi_cg": "#8056B3",
    }
    labels = {
        "fsai_cg": "Tuned FSAI-CG",
        "fsai_cr": "Tuned FSAI-CR",
        "jacobi_cr": "Jacobi-CR",
        "jacobi_cg": "Jacobi-CG",
    }
    for name in ["jacobi_cg", "jacobi_cr", "fsai_cg", "fsai_cr"]:
        samples = [s for s in data["results"][name]["samples"] if s["iteration"] > 0]
        iteration = np.array([s["iteration"] for s in samples])
        recomputed = np.array([s["relative_residual"] for s in samples])
        recursive = np.array([s["recursive_relative_residual"] for s in samples])
        assert np.all(np.diff(iteration) > 0) and np.all(recomputed > 0) and np.all(recursive > 0)
        ax.loglog(iteration, recursive, color=colors[name], ls=(0, (3, 3)), lw=1.7, alpha=0.8)
        ax.loglog(iteration, recomputed, color=colors[name], lw=2.2, label=labels[name], zorder=4)
        ax.scatter(iteration[-1], recomputed[-1], s=35, color=colors[name], zorder=5)
    k = data["k"]
    for mult in [1, 10, 100]:
        x = mult * k
        ax.axvline(x, color="#CBD1D9", lw=0.9, ls=(0, (2, 4)), zorder=0)
        ax.text(
            x,
            1.04,
            "k" if mult == 1 else f"{mult}k",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#738094",
        )
    ax.axhline(1e-12, color="#98A2B1", lw=1, ls=(0, (6, 4)), zorder=0)
    ax.text(1.5, 1.6e-12, r"Recursive stopping tolerance: $10^{-12}$", fontsize=10, color="#738094")
    fsai = data["results"]["fsai_cg"]["samples"][-1]
    jacobi = data["results"]["jacobi_cr"]["samples"][-1]
    ax.annotate(
        f"FSAI-CG: {fsai['relative_residual']:.2e}\nat {k:,} iterations",
        xy=(k, fsai["relative_residual"]),
        xytext=(k / 15, 3e-10),
        fontsize=10,
        color=colors["fsai_cg"],
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "color": colors["fsai_cg"], "lw": 0.9},
    )
    fsai_cr = data["results"]["fsai_cr"]["samples"][-1]
    ax.annotate(
        f"FSAI-CR: {fsai_cr['relative_residual']:.2e}\nat {fsai_cr['iteration']:,} iterations",
        xy=(fsai_cr["iteration"], fsai_cr["relative_residual"]),
        xytext=(2e5, 2e-11),
        fontsize=10,
        color=colors["fsai_cr"],
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "color": colors["fsai_cr"], "lw": 0.9},
    )
    ax.annotate(
        f"Jacobi-CR: {jacobi['relative_residual']:.2e}",
        xy=(100 * k, jacobi["relative_residual"]),
        xytext=(100 * k / 1.3, 1e-5),
        fontsize=10,
        color=colors["jacobi_cr"],
        ha="right",
        arrowprops={"arrowstyle": "-", "color": colors["jacobi_cr"], "lw": 0.9},
    )
    ax.set_xlim(1, 100 * k * 1.5)
    ax.set_ylim(1e-17, 1.3)
    ax.set_xlabel("Iterations (log scale)", labelpad=12)
    ax.set_ylabel(r"Relative residual $\|r_k\|_2\,/\,\|b\|_2$ (log scale)", labelpad=12)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#E5E9EF", lw=0.8)
    ax.set_axisbelow(True)
    method_legend = ax.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor="white", fontsize=12
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#536174", lw=2.2, label="Independently recomputed"),
            Line2D(
                [0], [0], color="#536174", lw=1.7, ls=(0, (3, 3)), label="Solver recursive residual"
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.10, 0.945, "Biharmonic residual convergence", fontsize=23, fontweight="bold", ha="left"
    )
    fig.text(
        0.10,
        0.896,
        "Dragon Dirichlet problem  ·  Same operator and zero free initial guess",
        fontsize=12,
        color="#66758A",
    )
    fig.text(
        0.10,
        0.061,
        "Solid curves recompute the full energy gradient; dashed curves show Warp’s internal recurrence.",
        fontsize=10,
        color="#66758A",
    )
    fig.text(
        0.10,
        0.031,
        "Real sampled iterates; no restarts or smoothing. Iteration 0 and recursive values below 10⁻¹⁷ are outside the axes.",
        fontsize=10,
        color="#66758A",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

"""Plot verified displacement and energy errors for the three tuned finalists."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("results/elasticity.json").read_text())
    colors = dict(jacobi="#D17632", block="#8056B3", fsai="#087E8B")
    labels = dict(jacobi="Scalar Jacobi", block="Block Jacobi", fsai="FSAI")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.22, top=0.77, wspace=0.25)
    for family, row in data["finalists"].items():
        label = labels[family] + " + " + row["config"]["solver"].upper()
        for ax, values, xkey in [
            (axes[0], row["samples"], "iteration"),
            (axes[1], row["timed_prefixes"], "total_s"),
        ]:
            values = [s for s in values if s[xkey] > 0]
            ax.loglog(
                [s[xkey] for s in values],
                [s["mass_error"] for s in values],
                color=colors[family],
                label=label,
                lw=2.1,
            )
            ax.loglog(
                [s[xkey] for s in values],
                [s["energy_error"] for s in values],
                color=colors[family],
                ls="--",
                alpha=0.65,
                lw=1.3,
            )
    for ax in axes:
        ax.axhline(data["target"], color="#8794A4", ls=":", lw=1.2)
        ax.set_ylabel("Relative error against verified reference")
        ax.grid(which="major", color="#E5E9EF", lw=0.7)
        ax.set_ylim(1e-6, 2)
    for family, row in data["finalists"].items():
        snap = row["snapshot"]
        axes[1].scatter(
            snap["total_s"],
            snap["mass_error"],
            color=colors[family],
            edgecolor="white",
            linewidth=0.8,
            s=45,
            zorder=5,
        )
    axes[0].set_xlabel("Iterations")
    axes[1].set_xlabel("Setup + solve wall time (seconds)")
    axes[1].axvline(data["snapshot_budget_s"], color="#536174", ls="-.", lw=1)
    axes[1].text(
        data["snapshot_budget_s"],
        1.02,
        "Render budget",
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        color="#536174",
    )
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    fig.text(
        0.075,
        0.94,
        "SimJEB #225 bracket under vertical pin load",
        fontsize=21,
        weight="bold",
        color="#263448",
    )
    fig.text(
        0.075,
        0.875,
        f"{data['mesh']['tetrahedra']:,} tetrahedra · {data['mesh']['free_dofs']:,} free DOFs · Four bolt holes fixed · NVIDIA L40",
        fontsize=11,
        color="#66758A",
    )
    fig.text(
        0.075,
        0.09,
        "Solid: lumped-mass displacement error. Dashed: energy error. Target: both ≤ 10⁻⁴.",
        fontsize=10,
        color="#66758A",
    )
    fig.text(
        0.075,
        0.045,
        "Same 3×3 BSR operator and zero initial guess. Timed prefixes exclude JIT, uploads, and independent verification.",
        fontsize=10,
        color="#66758A",
    )
    out = Path("assets/simjeb-elasticity-convergence.png")
    fig.savefig(out, dpi=190, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()

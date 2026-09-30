"""Plot successful mixed-biharmonic timings and the unresolved triharmonic residuals."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.2), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(top=0.73, bottom=0.24, left=0.065, right=0.98)
    for ax, filename, title in zip(
        axes[:2],
        ["indefinite-original-repeats", "indefinite-hard-repeats"],
        [
            "Original mixed biharmonic\nα = 1 · all three coordinate RHSs",
            "Harder mixed biharmonic\nα = 10⁻⁴ · one nonlinear RHS",
        ],
    ):
        records = json.loads(Path(f"results/{filename}.json").read_text())["results"]
        for i, method in enumerate(["shift8", "shift48"]):
            trials = [r for r in records if r["method"] == method]
            assert len(trials) == 3 and all(r["validated"] for r in trials)
            r = sorted(trials, key=lambda r: r["total_s"])[1]
            setup = r["operator_setup_s"] + r["setup_s"]
            solve = sum(c["solve_s"] for c in r["columns"])
            ax.bar(
                i,
                setup,
                width=0.55,
                color="#A3B9C4",
                label="Operator + preconditioner setup" if i == 0 else None,
            )
            ax.bar(
                i,
                solve,
                bottom=setup,
                width=0.55,
                color="#087E8B",
                label="GMRES solve(s)" if i == 0 else None,
            )
            ax.text(
                i,
                r["total_s"],
                f"{r['total_s']:.3f} s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_xticks([0, 1], ["Shifted block\nFSAI width 8", "Shifted block\nFSAI width 48"])
        ax.set_title(title, pad=18, fontsize=12)
        ax.set_ylabel("Total seconds (median of 3 trials)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False, fontsize=9)
    records = json.loads(Path("results/indefinite-second-probe.json").read_text())["results"]
    ax = axes[2]
    for i, method in enumerate(["block-jacobi", "shift8", "shift48"]):
        r = next(r for r in records if r["order"] == 3 and r["method"] == method)
        assert not r["validated"]
        value = r["columns"][0]["relative_residual"]
        ax.bar(i, value, width=0.55, color="#B97349")
        ax.text(i, value * 1.4, f"{value:.2e}", ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1)
    ax.axhline(1e-8, ls="--", lw=1, color="#526174")
    ax.text(-0.4, 1.6e-8, "Residual target: 10⁻⁸", fontsize=9, color="#526174")
    ax.set_xticks([0, 1, 2], ["Block\nJacobi", "Shifted\nFSAI 8", "Shifted\nFSAI 48"])
    ax.set_ylabel("Original-system relative residual")
    ax.set_title(
        "Original mixed triharmonic\n3,100 iterations · all fail validation", pad=18, fontsize=12
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(
        0.065, 0.94, "Preconditioning the indefinite dragon systems", fontsize=22, fontweight="bold"
    )
    fig.text(
        0.065,
        0.875,
        "NVIDIA L40 · Pure-Warp GMRES · Fixed linear preconditioners · No multigrid",
        fontsize=12,
        color="#526174",
    )
    fig.text(
        0.065,
        0.04,
        "Biharmonic: verified solves; harder case uses symmetric block equilibration. Triharmonic: residual reduction is not convergence.",
        fontsize=10,
        color="#526174",
    )
    out = Path("assets/indefinite-preconditioners.png")
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

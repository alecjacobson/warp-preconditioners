"""Plot the independent zero-versus-data initialization experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/visualization"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/dragon-initialization-comparison.png")
    )
    args = parser.parse_args()
    report = json.loads((args.data / "initialization_audit.json").read_text())
    rows = {r["name"]: r for r in report["results"]}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1, 1.65]})
    colors = {"zero": "#667fb5", "data": "#249b8a"}
    for i, guess in enumerate(["zero", "data"]):
        r = rows[f"fsai_{guess}"]
        axes[0].bar(i, r["actual_iterations"], color=colors[guess], width=0.58)
        axes[0].text(
            i,
            r["actual_iterations"] + 500,
            f"{r['actual_iterations']:,}\n{r['solve_s']:.2f} s",
            ha="center",
            va="bottom",
        )
        points = [
            (0, 100 * report["initial_fields"][guess]["relative_mass_norm_difference_from_fsai"])
        ]
        points += [
            (r["actual_iterations"], 100 * r["relative_mass_norm_difference_from_fsai"])
            for r in report["results"]
            if r["name"].startswith(f"jacobi_{guess}_")
        ]
        points.sort()
        axes[1].plot(
            *zip(*points),
            "o--",
            color=colors[guess],
            lw=2,
            markersize=7,
            label=f"Start from {guess}",
        )
        axes[1].annotate(
            f"{points[-1][1]:.1f}%",
            points[-1],
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=colors[guess],
        )
    axes[0].set_xticks([0, 1], ["Start from zero", "Start from data"])
    axes[0].set_ylim(0, 36000)
    axes[0].set_ylabel("Iterations to recursive relative tolerance 1e−8")
    axes[0].set_title("FSAI-CG: modest iteration savings", pad=15)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    axes[1].set_xscale("symlog", linthresh=100)
    k = rows["fsai_data"]["actual_iterations"]
    axes[1].set_xticks([0, 1000, k], ["0", "1,000", f"{k:,}"])
    axes[1].set_xlim(-5, k * 2)
    axes[1].set_ylim(0, 108)
    axes[1].set_xlabel("Jacobi-CR iterations (sampled checkpoints)")
    axes[1].set_ylabel("Relative mass-norm field difference from FSAI (%)")
    axes[1].set_title("Jacobi-CR: substantially better early fields", pad=15)
    axes[1].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e8eb")
    fig.suptitle(
        "Initial guess experiment · same nonlinear target, equation and preconditioners",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.025,
        "Data weight = 1e−4 · independent paired solves from zero/data · dashed lines join measured checkpoints only",
        ha="center",
        fontsize=10,
        color="#57606d",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

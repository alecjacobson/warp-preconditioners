"""Publish repeated-trial summary and a figure from committed measurement JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("results/dirichlet-tuning")

    def read(name):
        return json.loads((root / f"{name}.json").read_text())["results"]

    trials = read("repeated")
    keys = ["baseline", "total48", "total64", "lean64", "jacobi_stored", "jacobi_factored"]
    summary = {}
    for key in keys:
        rows = [r for r in trials if r["config"]["label"] == key]
        summary[key] = {"config": rows[0]["config"], "repeats": len(rows)}
        for field in (
            "setup_s",
            "operator_setup_s",
            "solve_s",
            "total_s",
            "us_per_iteration",
            "iterations",
            "physical_mass_relative_error",
            "operator_relative_residual",
        ):
            values = [r[field] for r in rows]
            summary[key][field] = dict(
                median=float(np.median(values)), min=min(values), max=max(values)
            )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax = axes[0, 0]
    rows = [
        r for r in read("regularization") if r["case"].startswith("fixed_patches") and r["tol"] > 0
    ]
    colors = ["#81b8d6", "#3c88ac", "#194d68"]
    bars = ax.bar(
        [r"$\alpha=1$", r"$\alpha=10^{-4}$", r"$\alpha=0$"],
        [r["solve_s"] for r in rows],
        color=colors,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.05, 150)
    ax.set_ylabel("Solve time (seconds, logarithmic)")
    ax.set_title("Why the problem became harder\nSame constraints, RHS, width 8, tolerance")
    for bar, r in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r["solve_s"] * 1.15,
            f"{r['solve_s']:.3g} s\n{r['iterations']:,} iterations",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[0, 1]
    names = ["Original\nwidth 8", "Tuned\nwidth 48", "Tuned\nwidth 64", "Cheaper steps\nwidth 64"]
    x = np.arange(4)
    setup = np.array(
        [
            summary[k]["setup_s"]["median"] + summary[k]["operator_setup_s"]["median"]
            for k in keys[:4]
        ]
    )
    solve = np.array([summary[k]["solve_s"]["median"] for k in keys[:4]])
    ax.bar(x, solve, color="#3c88ac", label="Solve")
    ax.bar(x, setup, bottom=solve, color="#e4a051", label="Operator + FSAI setup")
    for i, key in enumerate(keys[:4]):
        r = summary[key]["total_s"]
        ax.plot([i, i], [r["min"], r["max"]], color="black", linewidth=2)
        ax.text(i, r["max"] + 0.8, f"{r['median']:.2f} s", ha="center")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Seconds; median of three trials")
    ax.set_title("Same unregularized energy and zero initial guess")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    order = ["baseline", "jacobi_stored", "total48", "lean64", "jacobi_factored"]
    labels = [
        "Original FSAI-CG",
        "Original Jacobi-CR",
        "Tuned FSAI-CG",
        "Cheaper-step FSAI-CG",
        "Factored Jacobi-CR",
    ]
    values = [summary[k]["us_per_iteration"]["median"] for k in order]
    ax.barh(labels, values, color=["#194d68", "#aaa", "#3c88ac", "#81b8d6", "#aaa"])
    for i, v in enumerate(values):
        ax.text(v + 4, i, f"{v:.0f}", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("Microseconds per complete solver iteration")
    ax.set_title(
        "FSAI remains more expensive than Jacobi\nJacobi measured over 5,000 steps; not converged"
    )

    ax = axes[1, 1]
    for files, color, label in [
        (["widths", "kap", "order", "solvers"], "#aaa", "Parameter / layout trials"),
        (["kernels", "finalists"], "#e4a051", "Cooperative products / factored trials"),
        (
            ["matrix-free", "production", "nearby", "lean"],
            "#3c88ac",
            "Factored operator + tuned factors",
        ),
    ]:
        rows = [r for file in files for r in read(file) if r["config"].get("width", 8) > 0]
        ax.scatter(
            [r["us_per_iteration"] for r in rows],
            [r["setup_s"] + r["solve_s"] for r in rows],
            s=24,
            c=color,
            label=label,
        )
    ax.set_xlabel("Microseconds per iteration")
    ax.set_ylabel("FSAI setup + solve (seconds)")
    ax.set_title("Fewer iterations alone is not the objective")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig("assets/dragon-dirichlet-tuning.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

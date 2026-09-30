"""Compare independently solved reference fields on the original surface."""

import argparse
import json
from pathlib import Path

import igl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare_simjeb_remesh import areas, quality
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path("data/simjeb-ftetwild"))
    args = p.parse_args()
    old = np.load("data/simjeb/mesh.npz")
    new = np.load(args.data / "mesh.npz")
    a = np.load("data/simjeb/comparison.npz")
    b = np.load(args.data / "comparison.npz")
    oldresult = json.loads(Path("results/elasticity.json").read_text())
    newresult = json.loads(Path("results/simjeb-ftetwild.json").read_text())
    report = json.loads((args.data / "quality.json").read_text())
    surface = np.unique(old["faces"])
    points = old["vertices"][surface]
    dist, idx, closest = igl.point_mesh_squared_distance(points, new["vertices"], new["faces"])
    triangle = new["vertices"][new["faces"][idx]]
    bary = igl.barycentric_coordinates(closest, triangle[:, 0], triangle[:, 1], triangle[:, 2])
    np.testing.assert_allclose(bary.sum(1), 1, atol=1e-12)
    assert bary.min() > -1e-8

    def interpolate(field):
        return np.einsum("ij,ijk->ik", bary, field[new["faces"][idx]])

    u_new = interpolate(b["reference"])
    u_old = a["reference"][surface]
    s_new = interpolate(b["von_mises_reference"][:, None])[:, 0]
    s_old = a["von_mises_reference"][surface]
    weights = np.zeros(len(old["vertices"]))
    for k in range(3):
        np.add.at(weights, old["faces"][:, k], areas(old["vertices"], old["faces"]) / 3)
    weights = weights[surface]

    def relative(x, y, w):
        return float(np.sqrt(np.sum(w[:, None] * (x - y) ** 2) / np.sum(w[:, None] * y**2)))

    nearfixed = cKDTree(old["vertices"][old["fixed"]]).query(points)[0]
    away = nearfixed > 0.005

    def summary(mesh, fields, state):
        surface_nodes = np.unique(mesh["faces"])
        stress = fields["von_mises_reference"]
        distance_fixed = cKDTree(mesh["vertices"][mesh["fixed"]]).query(
            mesh["vertices"][surface_nodes]
        )[0]
        hottest = stress[surface_nodes] >= np.percentile(stress[surface_nodes], 99.9)
        peak = int(np.argmax(stress))
        return dict(
            max_displacement_mm=float(np.linalg.norm(fields["reference"], axis=1).max() * 1000),
            compliance_nm=float(np.sum(mesh["forces"] * fields["reference"])),
            nodal_peak_stress_mpa=float(fields["von_mises_reference"].max() / 1e6),
            peak_position_mm=(mesh["vertices"][peak] * 1000).tolist(),
            peak_node_fixed=bool(mesh["fixed"][peak]),
            hottest_surface_0_1_percent_max_distance_to_fixed_mm=float(
                distance_fixed[hottest].max() * 1000
            ),
            element_peak_stress_mpa=state["stress"]["fields"]["reference"]["element_max_pa"] / 1e6,
            independent_reference_residual=state["verification"][
                "reference_element_relative_residual"
            ],
            best_tested={
                k: dict(
                    config=r["config"],
                    iterations=r["iterations"],
                    setup_s=r["setup_s"],
                    solve_s=r["solve_s"],
                    total_s=r["total_s"],
                )
                for k, r in state["finalists"].items()
            },
        )

    report["solutions"] = dict(
        original=summary(old, a, oldresult), ftetwild=summary(new, b, newresult)
    )
    report["common_surface_comparison"] = dict(
        method="Closest-point interpolation of new nodal displacement and recovered stress onto original surface vertices; original surface lumped-area weights",
        displacement_relative_l2=relative(u_new, u_old, weights),
        stress_relative_l2=relative(s_new[:, None], s_old[:, None], weights),
        stress_relative_l2_more_than_5mm_from_fixed=relative(
            s_new[away, None], s_old[away, None], weights[away]
        ),
        max_projection_distance_mm=float(np.sqrt(dist.max()) * 1000),
        caveat="Two P1 discretizations of the same tagged surface, with changed node locations and equal-weight load sampling; not a converged stress ground truth.",
    )
    Path("results/simjeb-mesh-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    # Plot the poor-quality tail and a common-surface stress comparison.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.21, top=0.78, wspace=0.32)
    for mesh, name, color in [(old, "Original SimJEB", "#D17C34"), (new, "fTetWild", "#09897D")]:
        _, ratio, angle = quality(mesh["vertices"], mesh["tets"])
        for ax, values in zip(axes[:2], [angle, ratio]):
            values = np.sort(values)
            ax.plot(
                values, 100 * np.arange(1, len(values) + 1) / len(values), color=color, label=name
            )
            ax.set_yscale("log")
            ax.set_ylim(0.001, 100)
            ax.grid(alpha=0.2)
    axes[0].set_xlabel("Minimum dihedral angle (degrees)")
    axes[0].set_ylabel("Cumulative tetrahedra (%)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Mean-ratio quality (1 = equilateral)")
    axes[1].set_ylabel("Cumulative tetrahedra (%)")
    sel = np.random.default_rng(7).choice(len(surface), min(6000, len(surface)), replace=False)
    axes[2].scatter(s_old[sel] / 1e6, s_new[sel] / 1e6, s=3, alpha=0.25, color="#09897D")
    limit = max(s_old.max(), s_new.max()) / 1e6
    axes[2].plot([0, limit], [0, limit], color="#8794A4", ls="--")
    axes[2].set_xlabel("Original recovered stress (MPa)")
    axes[2].set_ylabel("fTetWild recovered stress (MPa)")
    axes[2].grid(alpha=0.2)
    fig.text(
        0.07,
        0.94,
        "SimJEB #225: original mesh versus fTetWild",
        fontsize=22,
        weight="bold",
        color="#263448",
    )
    fig.text(
        0.07,
        0.875,
        f"{len(old['tets']):,} versus {len(new['tets']):,} tetrahedra · same surface regions, material, pin resultant and reference moment",
        color="#657083",
    )
    fig.text(
        0.07,
        0.09,
        "Quality curves expose the poor-element tail. Stress comparison uses closest-point interpolation onto the original surface.",
        color="#657083",
    )
    fig.text(
        0.07,
        0.045,
        "Both references use pure Warp solves. A smoother field on one mesh is not, by itself, proof of stress convergence.",
        color="#657083",
    )
    fig.savefig("assets/simjeb-mesh-quality.png", dpi=190, facecolor="white")
    print(json.dumps(report["solutions"], indent=2))
    print(report["common_surface_comparison"])


if __name__ == "__main__":
    main()

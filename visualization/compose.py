"""Annotate the raw-data dragon and four solves with a common scalar legend."""

import argparse
import json
from pathlib import Path

import numpy as np
from colormap import striped_okloop
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/visualization"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/dragon-biharmonic-comparison.png")
    )
    args = parser.parse_args()
    meta = json.loads((args.data / "solve.json").read_text())
    render = json.loads((args.data / "render.json").read_text())
    scene = Image.open(args.data / "five_dragons.png").convert("RGBA")
    data = np.load(args.data / "fields.npz")
    reference_name = meta["display_order"][0]
    reference_stats = meta["results"][reference_name]
    data_std = meta["target_statistics"]["mass_weighted_std"]
    comparison = dict(
        reference_field=reference_name,
        mass_weighted_rms_change=reference_stats["data_fit_rms"],
        rms_change_over_data_std=reference_stats["data_fit_rms"] / data_std,
        max_abs_change=float(np.max(np.abs(data[reference_name] - data["target"]))),
        data_std=data_std,
        smoothed_std=reference_stats["mass_weighted_std"],
        std_reduction_fraction=1 - reference_stats["mass_weighted_std"] / data_std,
        bending_reduction_fraction=1
        - reference_stats["bending_energy"] / meta["target_bending_energy"],
    )
    width, ph = scene.size
    header, footer = 260, 285
    height = ph + header + footer
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(scene, (0, header), scene)
    draw = ImageDraw.Draw(canvas)
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def text(x, y, label, size=34, heavy=False, color="#273448"):
        draw.text(
            (x, y),
            label,
            font=ImageFont.truetype(bold if heavy else regular, size),
            fill=color,
            anchor="mt",
        )

    text(width / 2, 24, "Raw data and biharmonic smoothing", 66, True)
    text(
        width / 2,
        111,
        f"Data weight α = {meta['data_weight']:g}  ·  Smoothing weight = 1  ·  Initial guess = {'data f' if meta['initial_guess'] == 'data' else 'zero'}  ·  One shared scalar scale",
        31,
        color="#657083",
    )
    for ob in render["objects"]:
        name = ob["name"]
        cx = ob["label_x_fraction"] * width
        y = header + ph - 25
        if name == "target":
            text(cx, 185, "Raw data f", 44, True)
            text(cx, 239, "Unsmoothed input", 37)
            text(cx, y, "No solve", 36, True)
            text(
                cx,
                y + 49,
                f"Bending energy  {meta['target_bending_energy']:.2f}",
                29,
                color="#657083",
            )
            text(cx, y + 88, f"Mass-weighted std  {data_std:.3f}", 27, color="#657083")
            continue
        r = meta["results"][name]
        label = "Warp CG + FSAI" if name.startswith("fsai") else "Warp CR + Jacobi"
        if name.startswith("fsai") and meta.get("tuned"):
            label = "Warp CG + tuned FSAI"
        text(cx, 185, label, 44, True)
        multiplier = r["iteration_multiplier"]
        budget_label = "k" if multiplier == 1 else f"{multiplier}k"
        status = " · converged" if name.startswith("fsai") else ""
        text(cx, 239, f"{budget_label} = {r['actual_iterations']:,}{status}", 37)
        text(cx, y, f"{r['solve_s']:.2f} s solve", 36, True)
        text(cx, y + 49, f"Relative residual  {r['relative_residual']:.2e}", 29, color="#657083")
        text(cx, y + 88, f"Setup  {r['setup_s']:.3f} s", 27, color="#657083")
    _, palette = striped_okloop()
    legend_w = 2200
    legend_x = (width - legend_w) // 2
    legend_y = height - 137
    for i, c in enumerate(palette):
        x0 = legend_x + round(i * legend_w / 26)
        x1 = legend_x + round((i + 1) * legend_w / 26)
        draw.rectangle((x0, legend_y, x1, legend_y + 33), fill=tuple(np.rint(c * 255).astype(int)))
    lo, hi = render["scalar_range"]
    for t in [0, 0.25, 0.5, 0.75, 1]:
        x = legend_x + t * legend_w
        draw.line((x, legend_y + 35, x, legend_y + 43), fill="#8B94A1", width=2)
        text(x, legend_y + 48, f"{lo + t * (hi - lo):.3g}", 25, color="#657083")
    text(
        width / 2,
        height - 42,
        f"Same scale for f and u · k from FSAI stopping tolerance {meta['convergence']['fsai_rtol']:.0e} · RMS change {comparison['mass_weighted_rms_change']:.3f} ({100 * comparison['rms_change_over_data_std']:.1f}% of data std) · std reduction {100 * comparison['std_reduction_fraction']:.1f}%",
        27,
        color="#657083",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    meta["render"] = render
    meta["solver_display_order"] = meta["display_order"]
    meta["display_order"] = render["display_order"]
    meta["solver_scalar_range"] = meta["scalar_range"]
    meta["scalar_range"] = render["scalar_range"]
    meta["raw_data_comparison"] = comparison
    meta["figure_size"] = list(canvas.size)
    meta["note"] = (
        "One scene render; labels and legend added afterward. Single scalar RHS timings, excluding setup."
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output, canvas.size)


if __name__ == "__main__":
    main()

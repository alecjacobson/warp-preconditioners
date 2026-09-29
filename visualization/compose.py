"""Annotate one four-dragon scene render with solver labels and a common legend."""

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
    if (args.data / "weight_sweep.json").exists():
        meta["weight_sweep"] = json.loads((args.data / "weight_sweep.json").read_text())
    render = json.loads((args.data / "render.json").read_text())
    scene = Image.open(args.data / "four_dragons.png").convert("RGBA")
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

    text(width / 2, 24, "Biharmonic smoothing · fixed iteration budgets", 66, True)
    text(
        width / 2,
        111,
        f"Data weight α = {meta['data_weight']:g}  ·  Smoothing weight = 1  ·  Zero initial guesses  ·  One shared scalar scale",
        31,
        color="#657083",
    )
    for ob in render["objects"]:
        name = ob["name"]
        r = meta["results"][name]
        cx = ob["label_x_fraction"] * width
        label = "Warp CG + FSAI" if name.startswith("fsai") else "Warp CR + Jacobi"
        text(cx, 185, label, 44, True)
        text(cx, 239, f"k = {r['actual_iterations']:,}", 37)
        y = header + ph - 25
        text(cx, y, f"{r['solve_s']:.2f} s solve", 36, True)
        text(cx, y + 49, f"Relative residual  {r['relative_residual']:.2e}", 29, color="#657083")
        text(cx, y + 88, f"Backward error  {r['backward_error']:.2e}", 27, color="#657083")
    _, palette = striped_okloop()
    legend_w = 2200
    legend_x = (width - legend_w) // 2
    legend_y = height - 137
    for i, c in enumerate(palette):
        x0 = legend_x + round(i * legend_w / 26)
        x1 = legend_x + round((i + 1) * legend_w / 26)
        draw.rectangle((x0, legend_y, x1, legend_y + 33), fill=tuple(np.rint(c * 255).astype(int)))
    lo, hi = meta["scalar_range"]
    for t in [0, 0.25, 0.5, 0.75, 1]:
        x = legend_x + t * legend_w
        draw.line((x, legend_y + 35, x, legend_y + 43), fill="#8B94A1", width=2)
        text(x, legend_y + 48, f"{lo + t * (hi - lo):.1f}", 25, color="#657083")
    text(
        width / 2,
        height - 42,
        "Computed height u · original mesh units · finite-budget iterates, not converged references",
        27,
        color="#657083",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    meta["render"] = render
    meta["figure_size"] = list(canvas.size)
    meta["note"] = (
        "One scene render; labels and legend added afterward. Single scalar RHS timings, excluding setup."
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output, canvas.size)


if __name__ == "__main__":
    main()

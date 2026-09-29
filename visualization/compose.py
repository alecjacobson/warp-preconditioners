"""Compose the scientific comparison and common legend on a white canvas."""

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
    left = Image.open(args.data / "jacobi_cr.png").convert("RGBA")
    right = Image.open(args.data / "fsai_cg.png").convert("RGBA")
    assert left.size == right.size
    pw, ph = left.size
    width = 2 * pw + 120
    height = ph + 490
    canvas = Image.new("RGB", (width, height), "white")
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

    text(width / 2, 36, "Biharmonic field on the dragon", 62, True)
    text(
        width / 2,
        118,
        "Same mesh. Same scalar field problem. Same 26 isointervals.",
        30,
        color="#657083",
    )
    centers = [30 + pw / 2, 90 + 1.5 * pw]
    text(centers[0], 196, "Without FSAI", 43, True)
    text(centers[1], 196, "With FSAI", 43, True)
    text(centers[0], 251, "Warp CR + Jacobi", 30, color="#657083")
    text(centers[1], 251, "Warp CG + adaptive FSAI", 30, color="#657083")
    # The renders include transparent Cycles shadow catchers. Alpha-compositing
    # over white preserves contact shadows while leaving the background white.
    image_y = 287
    canvas.paste(left, (30, image_y), left)
    canvas.paste(right, (90 + pw, image_y), right)
    # Camera has generous lower whitespace, used for metrics beneath the feet.
    metrics_y = image_y + ph - 58
    for cx, name in zip(centers, ["jacobi_cr", "fsai_cg"]):
        r = meta["results"][name]
        text(
            cx,
            metrics_y,
            f"{r['selected_iterations']:,} iterations  ·  {r['solve_s']:.2f} s solve",
            32,
            True,
        )
        text(cx, metrics_y + 47, f"Backward error  {r['backward_error']:.2e}", 27, color="#657083")
    _, palette = striped_okloop()
    legend_w = 1560
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
        text(x, legend_y + 48, f"{lo + t * (hi - lo):.1f}", 23, color="#657083")
    text(
        width / 2,
        height - 42,
        "Biharmonic height u  ·  shared scale in original mesh units",
        25,
        color="#657083",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    # Keep small, text-only provenance alongside the public PNG.
    for entry in meta["results"].values():
        entry.pop("history", None)
    meta["render"] = json.loads((args.data / "render.json").read_text())
    meta["figure_size"] = list(canvas.size)
    meta["note"] = (
        "Single scalar RHS timings from this reproduction; not the three-RHS benchmark totals."
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output, canvas.size)


if __name__ == "__main__":
    main()

"""Annotate head/tail Dirichlet interpolation and its solver comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from colormap import striped_okloop
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dirichlet"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/dragon-dirichlet-comparison.png")
    )
    args = parser.parse_args()
    meta = json.loads((args.data / "solve.json").read_text())
    render = json.loads((args.data / "render.json").read_text())
    scene = Image.open(args.data / "five_dragons.png").convert("RGBA")
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

    text(width / 2, 24, "Biharmonic interpolation: tail −1, head +1", 66, True)
    text(
        width / 2,
        111,
        "Minimize ½ uᵀ L M⁻¹ L u  ·  No data term  ·  Free variables start at zero  ·  Shared scale for all solutions",
        31,
        color="#657083",
    )
    for ob in render["objects"]:
        name = ob["name"]
        cx = ob["label_x_fraction"] * width
        y = header + ph - 25
        if name == "constraints":
            text(cx, 185, "Prescribed regions", 44, True)
            text(cx, 239, "Blue tail −1 / red head +1", 35)
            text(cx, y, "Gray = unconstrained", 36, True)
            text(
                cx,
                y + 49,
                f"{meta['constraints']['tail']['vertices']:,} tail + {meta['constraints']['head']['vertices']:,} head vertices",
                28,
                color="#657083",
            )
            text(cx, y + 88, "Colors mark constraints, not a solution", 26, color="#657083")
            continue
        r = meta["results"][name]
        fsai = name.startswith("fsai")
        text(cx, 185, "Warp CG + FSAI" if fsai else "Warp CR + Jacobi", 44, True)
        multiplier = r["iteration_multiplier"]
        label = "k" if multiplier == 1 else f"{multiplier}k"
        text(
            cx, 239, f"{label} = {r['actual_iterations']:,}" + (" · converged" if fsai else ""), 37
        )
        text(cx, y, f"{r['solve_s']:.2f} s solve", 36, True)
        text(cx, y + 49, f"Relative residual  {r['relative_residual']:.2e}", 29, color="#657083")
        metric = (
            f"Setup {r['setup_s']:.2f} s · width 48 · κ = 0.003"
            if fsai and meta.get("tuned")
            else f"Energy  {r['bending_energy']:.5g}"
            if fsai
            else f"Field error vs FSAI  {100 * r['mass_relative_error_to_fsai']:.3g}%"
        )
        text(cx, y + 88, metric, 27, color="#657083")
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
        (
            f"Both methods: factored L M⁻¹ L · FSAI: float32 factors / float64 arithmetic · k from tolerance {meta['convergence']['fsai_rtol']:.0e} · Independently recomputed residuals · Exact constraints"
            if meta.get("tuned")
            else f"k from FSAI recursive stopping tolerance {meta['convergence']['fsai_rtol']:.0e} · Residuals recomputed independently · Constraints exact in every field · No clipping of overshoot"
        ),
        27,
        color="#657083",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    for filename, key in [
        ("cholesky.json", "cholesky_audit"),
        ("refinement.json", "refinement_audit"),
    ]:
        audit_path = args.data / filename
        if audit_path.exists():
            meta[key] = json.loads(audit_path.read_text())
    meta["render"] = render
    meta["figure_size"] = list(canvas.size)
    meta["note"] = (
        "One scene; labels and legend added afterward. Single RHS timings exclude setup. Region diagram uses a categorical palette."
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output, canvas.size)


if __name__ == "__main__":
    main()

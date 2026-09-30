"""Label the simultaneous elasticity rendering with measured times and errors."""

import json
from pathlib import Path

import numpy as np
from colormap import striped_okloop
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path("data/elasticity")
    meta = json.loads(Path("results/elasticity.json").read_text())
    render = json.loads((root / "render.json").read_text())
    scene = Image.open(root / "elasticity-render.png").convert("RGBA")
    width, ph = scene.size
    header, footer = 255, 270
    canvas = Image.new("RGB", (width, ph + header + footer), "white")
    canvas.paste(scene, (0, header), scene)
    draw = ImageDraw.Draw(canvas)

    def text(x, y, label, size=32, bold=False, color="#273448"):
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size
        )
        draw.text((x, y), label, font=font, fill=color, anchor="mt")

    text(width / 2, 20, "Linear elasticity: head and tail clamped, gravity ↓", 60, True)
    text(
        width / 2,
        104,
        f"Frozen near {meta['snapshot_budget_s']:.3f} s setup + solve  ·  NVIDIA L40  ·  Actual displacement, no exaggeration  ·  Dark gray = fixed",
        31,
        color="#657083",
    )
    names = dict(
        reference="Verified reference", jacobi="Scalar Jacobi", block="Block Jacobi", fsai="FSAI"
    )
    for ob in render["objects"]:
        name = ob["name"]
        cx = ob["label_x_fraction"] * width
        title = names[name]
        if name != "reference":
            title += " + " + meta["finalists"][name]["config"]["solver"].upper()
        text(cx, 184, title, 41, True)
        y = header + ph - 12
        if name == "reference":
            text(cx, 240, "Pure Warp · tightly converged", 29)
            text(cx, y, "Reference displacement", 33, True)
            text(cx, y + 48, "Independent element-force verification", 24, color="#657083")
            text(cx, y + 87, "Colors show recovered von Mises stress", 24, color="#657083")
        else:
            row = meta["finalists"][name]
            snap = row["snapshot"]
            text(cx, 240, f"{snap['iterations']:,} iterations", 30)
            text(cx, y, f"{snap['total_s']:.3f} s setup + solve", 33, True)
            text(
                cx,
                y + 48,
                f"Displacement error  {100 * snap['mass_error']:.3g}%",
                27,
                color="#657083",
            )
            text(
                cx, y + 87, f"Energy error  {100 * snap['energy_error']:.3g}%", 26, color="#657083"
            )
    _, palette = striped_okloop()
    lw = 2100
    lx = (width - lw) // 2
    ly = canvas.height - 128
    for i, c in enumerate(palette):
        draw.rectangle(
            (lx + round(i * lw / 26), ly, lx + round((i + 1) * lw / 26), ly + 28),
            fill=tuple(np.rint(c * 255).astype(int)),
        )
    for t in [0, 0.25, 0.5, 0.75, 1]:
        text(
            lx + t * lw,
            ly + 38,
            f"{t * render['scalar_range_pa'][1] / 1000:.2f}",
            24,
            color="#657083",
        )
    text(
        width / 2,
        canvas.height - 36,
        "Von Mises stress (kPa), shared linear scale  ·  225,770 tetrahedra  ·  152,058 free DOFs  ·  Target: relative displacement and energy errors ≤ 10⁻⁴",
        25,
        color="#657083",
    )
    out = Path("assets/dragon-elasticity-comparison.png")
    canvas.save(out, optimize=True)
    figure_meta = {
        k: meta[k] for k in ["mesh", "gpu", "warp", "snapshot_budget_s", "winner", "stress"]
    }
    figure_meta["results_source"] = "results/elasticity.json"
    figure_meta["render"] = render
    figure_meta["finalists"] = {
        name: {k: value for k, value in row.items() if k not in ["samples", "timed_prefixes"]}
        for name, row in meta["finalists"].items()
    }
    out.with_suffix(".json").write_text(json.dumps(figure_meta, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

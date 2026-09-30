"""Label the converged stress comparison and the transferred boundary node sets."""

import json
from pathlib import Path

import numpy as np
from colormap import striped_okloop
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path("data/simjeb-ftetwild")
    meta = json.loads((root / "render-comparison.json").read_text())
    report = json.loads(Path("results/simjeb-mesh-comparison.json").read_text())
    width, row_height, header, gap, footer = 3200, 1350, 230, 110, 180
    image = Image.new("RGB", (width, header + 3 * row_height + 2 * gap + footer), "white")
    draw = ImageDraw.Draw(image)

    def text(x, y, s, size=28, bold=False, color="#273448"):
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size
        )
        draw.text((x, y), s, font=font, fill=color, anchor="mt")

    text(width / 2, 20, "SimJEB #225: does a better tetrahedral mesh change the stress?", 48, True)
    text(
        width / 2,
        90,
        "Converged pure Warp references · actual displacement (1×) · shared stress scale",
        29,
        color="#657083",
    )
    centers = [width * (0.5 + (i - 0.5) * 2.35 / 4.9) for i in range(2)]
    for i, key in enumerate(["original", "ftetwild"]):
        q, s = report[key], report["solutions"][key]
        text(centers[i], 150, ["Original SimJEB / HyperMesh", "fTetWild remesh"][i], 38, True)
        text(
            centers[i],
            202,
            f"{q['tetrahedra']:,} tets · worst angle {q['min_dihedral_degrees']['0']:.2f}° · max |u| {s['max_displacement_mm']:.4f} mm",
            27,
        )
    for j, view in enumerate(["top", "underside", "boundary"]):
        y = header + j * (row_height + gap)
        scene = Image.open(root / f"comparison-{view}.png").convert("RGBA")
        image.paste(scene, (0, y), scene)
        if j == 0:
            text(
                width / 2,
                y + row_height + 12,
                "Underside: bolt-hole stress concentrations",
                35,
                True,
            )
            text(
                width / 2,
                y + row_height + 61,
                "Rigid display rotation only; the stress values and color scale are unchanged",
                27,
                color="#657083",
            )
        if j == 1:
            text(
                width / 2,
                y + row_height + 12,
                "Transferred boundary conditions: all fixed and loaded nodes",
                35,
                True,
            )
            text(
                width / 2,
                y + row_height + 61,
                "Orange: ux = uy = uz = 0 · Blue: pin load nodes · Arrow: total +Z force, 35,585.77 N",
                27,
                color="#657083",
            )
        if j == 2:
            for i, counts in enumerate(meta["boundary_nodes"]):
                text(
                    centers[i],
                    y + row_height - 55,
                    f"{counts['fixed']} fixed nodes · {counts['loaded']} loaded nodes",
                    30,
                    True,
                )
    bar_y = image.height - 125
    left, length = 600, 2000
    for i, color in enumerate(striped_okloop()[1]):
        draw.rectangle(
            (left + round(i * length / 26), bar_y, left + round((i + 1) * length / 26), bar_y + 25),
            fill=tuple(np.rint(color * 255).astype(int)),
        )
    for t in [0, 0.25, 0.5, 0.75, 1]:
        text(left + t * length, bar_y + 32, f"{t * meta['scalar_range_pa'][1] / 1e6:.1f}", 25)
    text(
        width / 2,
        image.height - 42,
        "Recovered von Mises stress (MPa) · 26 crisp bands · white background",
        27,
        color="#657083",
    )
    output = Path("assets/simjeb-mesh-comparison.png")
    image.save(output, optimize=True)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

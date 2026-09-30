"""Label the exact boundary-condition rendering with source-deck values."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path("data/simjeb")
    meta = json.loads((root / "boundary-render.json").read_text())
    scene = Image.open(root / "boundary-render.png").convert("RGBA")
    width, height = scene.size
    header = 220
    footer = 630
    canvas = Image.new("RGB", (width, height + header + footer), "white")
    canvas.paste(scene, (0, header), scene)
    draw = ImageDraw.Draw(canvas)

    def text(x, y, s, size=34, bold=False, color="#263448", anchor="mt"):
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size
        )
        draw.text((x, y), s, font=font, fill=color, anchor=anchor)

    text(width / 2, 24, "SimJEB #225 — exact fixed values and applied loads", 68, True)
    text(
        width / 2,
        114,
        "Undeformed input mesh  ·  SUBCASE 1 / SPC 1 / LOAD 2  ·  Colors identify boundary conditions, not stress",
        34,
        color="#657083",
    )
    titles = [
        "Top / pin side",
        "Underside / four supports",
        "See-through / coupling and nodal loads",
    ]
    for panel in meta["panels"]:
        i = panel["index"]
        cx = (0.5 + (i - 1) * 2.9 / 8.9) * width
        text(cx, 184, titles[i], 41, True)
        for key, point in panel["anchors"].items():
            x, y = point[0] * width, header + (1 - point[1]) * height
            if key.startswith("B"):
                radius = 29
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill="white",
                    outline="#DF6529",
                    width=4,
                )
                text(x, y, key, 26, True, anchor="mm")
            elif key == "force_tip":
                text(x + 28, y - 45, "35,585.77 N  (+Z)", 34, True, color="#1764AE", anchor="lm")
            elif key == "pin" and i == 2:
                text(x - 28, y + 38, "GRID 44126", 27, color="#1764AE", anchor="rt")
        for axis, point in panel["axes"].items():
            text(
                point[0] * width + 8,
                header + (1 - point[1]) * height - 10,
                axis,
                25,
                True,
                anchor="mm",
            )
    y = header + height + 16
    draw.line((120, y - 20, width - 120, y - 20), fill="#DEE3EA", width=2)
    text(150, y, "ORANGE  ·  Prescribed displacement", 41, True, color="#B64B13", anchor="lt")
    text(150, y + 69, "u_x = u_y = u_z = 0 at all 428 marked bolt-hole nodes", 37, True, anchor="lt")
    text(150, y + 130, "B1: 106 nodes    B2: 109    B3: 108    B4: 105", 32, anchor="lt")
    text(150, y + 183, "RBE2 centers: GRID 44122, 44123, 44124, 44125", 31, anchor="lt")
    text(
        150,
        y + 231,
        "Each center fixes translation + rotation; solid tet nodes have 3 translations.",
        29,
        anchor="lt",
    )
    text(2570, y, "BLUE  ·  Applied pin load", 41, True, color="#1764AE", anchor="lt")
    text(2570, y + 69, "F = (0, 0, 35,585.77) N at GRID 44126", 37, True, anchor="lt")
    text(2570, y + 130, "RBE3 distributes force to all 753 blue interface nodes.", 32, anchor="lt")
    text(
        2570,
        y + 183,
        "Their displacements remain unknown. Small arrows: ≈47.259 N each.",
        29,
        anchor="lt",
    )
    text(2570, y + 231, "Load center: (−21.028, −74.9383, 44.62994) mm", 31, anchor="lt")
    text(
        width / 2,
        y + 328,
        "Gray: unconstrained material. All other exposed faces have zero prescribed traction.",
        35,
        True,
    )
    text(
        width / 2,
        y + 388,
        "All 1,181 boundary nodes are marked; the see-through panel reveals nodes hidden by the solid surface.",
        30,
        color="#657083",
    )
    text(
        width / 2,
        y + 438,
        "Coupling lines are subsampled. Small nodal arrows share one scale; large arrows show the resultant at a separate scale.",
        28,
        color="#657083",
    )
    text(
        width / 2,
        y + 488,
        "Fixed regions are the four bolt-hole walls. This case has no gravity, contact, or prescribed nonzero displacement.",
        29,
        color="#657083",
    )
    text(
        width / 2,
        y + 543,
        "Node masks checked exactly against both 225.fem and the matrix/RHS input. Geometry and labels come directly from the source deck.",
        27,
        color="#657083",
    )
    out = Path("assets/simjeb-boundary-conditions.png")
    canvas.save(out, optimize=True)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

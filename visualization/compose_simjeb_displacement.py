"""Label actual displacement snapshots with load, accuracy and separate row scales."""

import json
from pathlib import Path

import numpy as np
from colormap import striped_okloop
from PIL import Image, ImageDraw, ImageFont


def main():
    state = json.loads(Path("results/simjeb-load-scaling.json").read_text())
    render = json.loads(
        Path("data/simjeb-ftetwild/load-scaling/render-displacement.json").read_text()
    )
    width, ph, header, footer = 4800, 1250, 275, 290
    stride = header + ph + footer
    canvas = Image.new("RGB", (width, stride * 2 + 90), "white")
    draw = ImageDraw.Draw(canvas)

    def text(x, y, s, size=30, bold=False, color="#273448"):
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size
        )
        draw.text((x, y), s, font=font, fill=color, anchor="mt")

    labels = dict(
        reference="Converged reference",
        jacobi="Scalar Jacobi + CG",
        block="Block Jacobi + CG",
        fsai="FSAI + CG",
    )
    for i, panel in enumerate(render["cases"]):
        case = next(c for c in state["cases"] if c["scale"] == panel["scale"])
        origin = i * stride
        scene = Image.open(panel["image"]).convert("RGBA")
        canvas.paste(scene, (0, origin + header), scene)
        force = np.linalg.norm(case["resultant_n"]) / 1000
        text(
            width / 2,
            origin + 18,
            f"SimJEB displacement · {case['scale']:g}× load ({force:.2f} kN)",
            58,
            True,
        )
        text(
            width / 2,
            origin + 102,
            f"Actual displacement (1× display) · Gray outline: unloaded shape · Snapshots near {case['snapshot_budget_s']:.4f} s · Color scale is specific to this row",
            31,
            color="#657083",
        )
        for obj in panel["objects"]:
            name = obj["name"]
            cx = width * obj["label_x_fraction"]
            text(cx, origin + 183, labels[name], 39, True)
            if name == "reference":
                text(
                    cx,
                    origin + 235,
                    f"Maximum displacement {case['reference_max_displacement_mm']:.4f} mm",
                    29,
                )
                text(cx, origin + header + ph, "Pure Warp · independently verified", 30, True)
            else:
                row = case["methods"][name]
                snap = row["snapshot"]
                passed = max(snap["mass_error"], snap["energy_error"]) <= state["target"]
                text(cx, origin + 235, f"Shown at {snap['iterations']:,} iterations", 29)
                y = origin + header + ph
                text(cx, y, f"Time to target: {row['total_s']:.4f} s", 32, True)
                text(
                    cx,
                    y + 45,
                    f"Snapshot {snap['total_s']:.4f} s · "
                    + ("target met" if passed else "target not reached"),
                    26,
                    color="#087E8B" if passed else "#9B5625",
                )
                text(
                    cx,
                    y + 84,
                    f"Displacement error {100 * snap['mass_error']:.3g}% · Energy {100 * snap['energy_error']:.3g}%",
                    25,
                    color="#657083",
                )
        by = origin + stride - 133
        length, left = 2200, 1300
        for j, color in enumerate(striped_okloop()[1]):
            draw.rectangle(
                (left + round(j * length / 26), by, left + round((j + 1) * length / 26), by + 26),
                fill=tuple(np.rint(color * 255).astype(int)),
            )
        for t in [0, 0.25, 0.5, 0.75, 1]:
            text(left + t * length, by + 35, f"{t * panel['scalar_range_m'][1] * 1000:.3f}", 25)
        text(
            width / 2,
            origin + stride - 40,
            "Displacement magnitude |u| (mm) · Shared scale across the four methods in this row",
            29,
            color="#657083",
        )
    text(
        width / 2,
        canvas.height - 70,
        "Fixed small-strain linear model at both loads · Relative displacement and energy target: each ≤ 0.01%",
        32,
        color="#657083",
    )
    output = Path("assets/simjeb-displacement.png")
    canvas.save(output, optimize=True)
    output.with_suffix(".json").write_text(
        json.dumps(dict(results_source="results/simjeb-load-scaling.json", render=render), indent=2)
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()

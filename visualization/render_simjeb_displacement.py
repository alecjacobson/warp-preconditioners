"""Render actual GPU load-scaling snapshots, colored by displacement magnitude."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colormap import striped_okloop
from render_dragon import area, look_at
from render_simjeb_boundary import material, mesh_object


def outline(vertices, faces, direction, radius, mat):
    p = vertices[faces]
    front = np.einsum("ij,j->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), direction) > 0
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    signs = np.bincount(inv, weights=np.repeat(front.astype(float) * 2 - 1, 3))
    edges = edges[(counts == 1) | (signs == 0)]
    curve = bpy.data.curves.new("Unloaded silhouette", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 0
    for edge in edges:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        for point, coord in zip(spline.points, vertices[edge]):
            point.co = (*coord, 1)
    ob = bpy.data.objects.new("Unloaded silhouette", curve)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)


def main():
    meta = json.loads(Path("results/simjeb-load-scaling.json").read_text())
    outdir = Path("data/simjeb-ftetwild/load-scaling")
    cases = [meta["cases"][0], meta["cases"][-1]]
    order = ["reference", "jacobi", "block", "fsai"]
    az, el = math.radians(-130), math.radians(32)
    right = Vector((-math.sin(az), math.cos(az), 0))
    direction = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    manifests = []
    for case in cases:
        fields = np.load(case["fields_file"])
        v, f = fields["vertices"], fields["faces"]
        center = (v.min(0) + v.max(0)) / 2
        center[2] = v[:, 2].min()
        scale = 1.90 / np.ptp(v @ np.array(right))
        maximum = max(float(np.linalg.norm(fields[k], axis=1).max()) for k in order)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "OPTIX"
        scene.cycles.device = "GPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 6
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 8
        scene.render.resolution_x = 4800
        scene.render.resolution_y = 1250
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 0.45
        mat = material("Physical displacement magnitude", (0.3, 0.3, 0.3))
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "displacement_m"
        normalize = nodes.new("ShaderNodeMath")
        normalize.operation = "DIVIDE"
        normalize.inputs[1].default_value = maximum
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
        for i, c in enumerate(striped_okloop()[0]):
            element = (
                ramp.color_ramp.elements[0] if i == 0 else ramp.color_ramp.elements.new(i / 26)
            )
            element.position = i / 26
            element.color = (*c, 1)
        links.new(attr.outputs["Fac"], normalize.inputs[0])
        links.new(normalize.outputs[0], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
        gray = material("Unloaded outline gray", (0.12, 0.15, 0.19))
        objects = []
        for i, name in enumerate(order):
            displacement = fields[name]
            digest = hashlib.sha256(displacement.tobytes()).hexdigest()
            expected = (
                case["reference_sha256"]
                if name == "reference"
                else case["methods"][name]["snapshot"]["field_sha256"]
            )
            assert digest == expected and np.all(displacement[fields["fixed"]] == 0)
            shift = np.array((i - 1.5) * 2.08 * right)
            positions = (v + displacement - center) * scale + shift
            obj = mesh_object(name, positions, f, [mat])
            obj.data.polygons.foreach_set("use_smooth", np.ones(len(f), dtype=bool))
            magnitude = np.linalg.norm(displacement, axis=1).astype(np.float32)
            attribute = obj.data.attributes.new("displacement_m", "FLOAT", "POINT")
            attribute.data.foreach_set("value", magnitude)
            coords = np.empty(positions.size, dtype=np.float32)
            obj.data.vertices.foreach_get("co", coords)
            np.testing.assert_array_equal(coords, positions.astype(np.float32).ravel())
            outline((v - center) * scale + shift, f, np.array(direction), 0.0015, gray)
            objects.append(
                dict(name=name, field_sha256=digest, label_x_fraction=0.5 + (i - 1.5) * 2.08 / 8.84)
            )
        area("soft key", (-3, -6, 9), 1000, 9)
        area("fill", (5, -3, 6), 380, 8)
        area("rim", (1, 5, 8), 800, 9)
        camdata = bpy.data.cameras.new("Shared camera")
        camera = bpy.data.objects.new("camera", camdata)
        bpy.context.collection.objects.link(camera)
        target = Vector((0, 0, 0.40))
        camera.location = target + 5 * direction
        look_at(camera, target)
        camdata.type = "ORTHO"
        camdata.ortho_scale = 8.84
        scene.camera = camera
        output = outdir / f"displacement-{case['scale']:g}.png"
        scene.render.filepath = str(output.resolve())
        bpy.ops.render.render(write_still=True)
        manifests.append(
            dict(
                scale=case["scale"], objects=objects, scalar_range_m=[0, maximum], image=str(output)
            )
        )
    (outdir / "render-displacement.json").write_text(
        json.dumps(
            dict(
                cases=manifests,
                amplification=1,
                outline="Unloaded surface silhouette in gray",
                colors="Displacement magnitude, 26 striped OKLab intervals; shared within each row, separate physical ranges across rows",
                renderer=bpy.app.version_string,
                device="OptiX",
                samples=64,
                model="Small-strain linear elasticity even at 10x load; not a finite-deformation or plasticity simulation",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

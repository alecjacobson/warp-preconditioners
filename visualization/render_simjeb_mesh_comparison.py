"""Render both converged meshes with shared stress scale and inspectable BC transfer."""

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
from render_simjeb_boundary import arrow, dots, material, mesh_object


def main():
    root = Path("data/simjeb-ftetwild")
    meshes = [np.load("data/simjeb/mesh.npz"), np.load(root / "mesh.npz")]
    fields = [np.load("data/simjeb/comparison.npz"), np.load(root / "comparison.npz")]
    states = [
        json.loads(Path(p).read_text())
        for p in ["results/elasticity.json", "results/simjeb-ftetwild.json"]
    ]
    bc = np.load("data/simjeb/boundary-conditions.npz")
    groups = [bc["fixed_group"], meshes[1]["fixed_group"]]
    loads = [bc["loaded"], meshes[1]["loaded"]]
    v = meshes[0]["vertices"]
    center = (v.min(0) + v.max(0)) / 2
    az, el = math.radians(-130), math.radians(32)
    right = Vector((-math.sin(az), math.cos(az), 0))
    scale = 1.9 / np.ptp(v @ np.array(right))
    maximum = max(float(f["von_mises_reference"].max()) for f in fields)
    provenance = []
    for mesh, field, state in zip(meshes, fields, states):
        np.testing.assert_array_equal(field["vertices"], mesh["vertices"])
        np.testing.assert_array_equal(field["faces"], mesh["faces"])
        assert np.all(field["reference"][mesh["fixed"]] == 0)
        digest = hashlib.sha256(field["von_mises_reference"].tobytes()).hexdigest()
        assert digest == state["stress"]["fields"]["reference"]["nodal_sha256"]
        provenance.append(
            dict(
                mesh_sha256=state["mesh"]["mesh_sha256"],
                stress_sha256=digest,
                displacement_sha256=hashlib.sha256(field["reference"].tobytes()).hexdigest(),
            )
        )
    for view in ["top", "underside", "boundary"]:
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
        scene.cycles.transparent_max_bounces = 32
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 8
        scene.render.resolution_x = 3200
        scene.render.resolution_y = 1350
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 0.5
        stressmat = material("Recovered von Mises stress", (0.3, 0.3, 0.3))
        nodes, links = stressmat.node_tree.nodes, stressmat.node_tree.links
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "stress"
        normalize = nodes.new("ShaderNodeMath")
        normalize.operation = "DIVIDE"
        normalize.inputs[1].default_value = maximum
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
        for j, color in enumerate(striped_okloop()[0]):
            element = (
                ramp.color_ramp.elements[0] if j == 0 else ramp.color_ramp.elements.new(j / 26)
            )
            element.position = j / 26
            element.color = (*color, 1)
        links.new(attr.outputs["Fac"], normalize.inputs[0])
        links.new(normalize.outputs[0], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
        ghost = material("Transparent undeformed surface", (0.24, 0.27, 0.32), 0.09)
        orange = material("Fixed u=0 nodes", (0.92, 0.23, 0.035))
        blue = material("Loaded nodes", (0.018, 0.30, 0.76))
        for i, (mesh, field) in enumerate(zip(meshes, fields)):
            rotation = np.diag([1, -1, -1]) if view == "underside" else np.eye(3)
            shift = np.array((i - 0.5) * 2.35 * right)
            vertices = (
                mesh["vertices"] if view == "boundary" else mesh["vertices"] + field["reference"]
            )
            positions = ((vertices - center) * scale) @ rotation.T + shift
            ob = mesh_object(
                f"mesh {i}", positions, mesh["faces"], [ghost if view == "boundary" else stressmat]
            )
            ob.data.polygons.foreach_set("use_smooth", np.ones(len(mesh["faces"]), dtype=bool))
            if view == "boundary":
                dots(f"all fixed nodes {i}", positions[groups[i] > 0], 0.0075, orange)
                dots(f"all loaded nodes {i}", positions[loads[i]], 0.0075, blue)
                ref = (bc["load_center"] - center) * scale + shift
                arrow("Total pin force +Z", ref, ref + np.array([0, 0, 0.75]), 0.014, blue)
            else:
                attribute = ob.data.attributes.new("stress", "FLOAT", "POINT")
                attribute.data.foreach_set("value", field["von_mises_reference"].astype(np.float32))
        area("key", (-3, -6, 8), 900, 8)
        area("fill", (5, -2, 6), 450, 7)
        area("rim", (0, 5, 7), 650, 7)
        camdata = bpy.data.cameras.new("shared orthographic camera")
        camera = bpy.data.objects.new("camera", camdata)
        bpy.context.collection.objects.link(camera)
        target = Vector((0, 0, 0.08 if view == "boundary" else 0))
        direction = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        camera.location = target + 6 * direction
        look_at(camera, target)
        camdata.type = "ORTHO"
        camdata.ortho_scale = 4.9
        scene.camera = camera
        scene.render.filepath = str((root / f"comparison-{view}.png").resolve())
        bpy.ops.render.render(write_still=True)
    (root / "render-comparison.json").write_text(
        json.dumps(
            dict(
                provenance=provenance,
                scalar_range_pa=[0, maximum],
                amplification=1,
                stress_recovery="Volume-average element stress tensors, then von Mises; no additional smoothing",
                views=[
                    "top",
                    "underside (rigid display rotation)",
                    "undeformed transparent boundary nodes",
                ],
                renderer=bpy.app.version_string,
                device="OptiX",
                samples=64,
                boundary_nodes=[
                    dict(fixed=int((group > 0).sum()), loaded=int(loaded.sum()))
                    for group, loaded in zip(groups, loads)
                ],
                colormap="isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

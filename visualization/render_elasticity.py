"""Render actual frozen elasticity iterates, sharing camera, light and colormap.

blender -b --factory-startup --python visualization/render_elasticity.py
"""

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


def main():
    root = Path("data/elasticity").resolve()
    fields = np.load(root / "comparison.npz")
    meta = json.loads(Path("results/elasticity.json").read_text())
    order = ["reference", "jacobi", "block", "fsai"]
    amplification = 1.0
    v, f = fields["vertices"], fields["faces"]
    center = (v.min(0) + v.max(0)) / 2
    center[2] = v[:, 2].min()
    scale = 2 / np.ptp(v[:, 0])
    scalar_max = max(float(fields["von_mises_" + name].max()) for name in order)
    palette, _ = striped_okloop()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.resolution_x = 4800
    scene.render.resolution_y = 1150
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
    mat = bpy.data.materials.new("Recovered von Mises stress, shared 26 bands")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.83
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "von_mises_pa"
    normalize = nodes.new("ShaderNodeMapRange")
    normalize.inputs["From Min"].default_value = 0
    normalize.inputs["From Max"].default_value = scalar_max
    normalize.clamp = True
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
    for i, c in enumerate(palette):
        el = ramp.color_ramp.elements[0] if i == 0 else ramp.color_ramp.elements.new(i / 26)
        el.position = i / 26
        el.color = (*c, 1)
    links.new(attr.outputs["Fac"], normalize.inputs["Value"])
    links.new(normalize.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    clamp = bpy.data.materials.new("Clamped head and tail")
    clamp.use_nodes = True
    clamp.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.12,
        0.14,
        0.17,
        1,
    )
    clamp.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
    az, el = math.radians(-70), math.radians(19)
    right = Vector((-math.sin(az), math.cos(az), 0))
    objects = []
    ground_z = (
        min(float(((v + amplification * fields[n] - center) * scale)[:, 2].min()) for n in order)
        - 0.015
    )
    for i, name in enumerate(order):
        displacement = fields[name]
        if name != "reference":
            assert (
                hashlib.sha256(displacement.tobytes()).hexdigest()
                == meta["finalists"][name]["snapshot"]["field_sha256"]
            )
        assert np.all(displacement[fields["fixed"]] == 0)
        positions = (v + amplification * displacement - center) * scale
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(v))
        mesh.vertices.foreach_set("co", positions.ravel())
        mesh.loops.add(f.size)
        mesh.loops.foreach_set("vertex_index", f.ravel())
        mesh.polygons.add(len(f))
        mesh.polygons.foreach_set("loop_start", np.arange(len(f), dtype=np.int32) * 3)
        mesh.polygons.foreach_set("loop_total", np.full(len(f), 3, dtype=np.int32))
        mesh.polygons.foreach_set("use_smooth", np.ones(len(f), dtype=bool))
        mesh.materials.append(mat)
        mesh.materials.append(clamp)
        mesh.polygons.foreach_set(
            "material_index", np.all(fields["fixed"][f], axis=1).astype(np.int32)
        )
        mesh.update()
        stress = fields["von_mises_" + name]
        assert (
            hashlib.sha256(stress.tobytes()).hexdigest()
            == meta["stress"]["fields"][name]["nodal_sha256"]
        )
        u = stress.astype(np.float32)
        attr = mesh.attributes.new("von_mises_pa", "FLOAT", "POINT")
        attr.data.foreach_set("value", u)
        check = np.empty_like(u)
        attr.data.foreach_get("value", check)
        np.testing.assert_array_equal(check, u)
        coords = np.empty(3 * len(v), dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        np.testing.assert_array_equal(coords, positions.astype(np.float32).ravel())
        ob = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(ob)
        ob.location = (i - 1.5) * 2.08 * right
        objects.append(
            dict(
                name=name,
                label_x_fraction=0.5 + (i - 1.5) * 2.08 / 8.84,
                field_sha256=hashlib.sha256(displacement.tobytes()).hexdigest(),
                geometry_and_attribute_verified=True,
            )
        )
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, ground_z))
    bpy.context.object.is_shadow_catcher = True
    area("soft key", (-3, -6, 9), 1000, 9)
    area("fill", (5, -3, 6), 380, 8)
    area("rim", (1, 5, 8), 800, 9)
    camera_data = bpy.data.cameras.new("orthographic comparison")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)
    target = Vector((0, 0, 0.36))
    direction = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    camera.location = target + 5 * direction
    look_at(camera, target)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 8.84
    scene.camera = camera
    scene.render.filepath = str(root / "elasticity-render.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "elasticity.blend"))
    bpy.ops.render.render(write_still=True)
    (root / "render.json").write_text(
        json.dumps(
            dict(
                objects=objects,
                amplification=amplification,
                scalar_range_pa=[0, scalar_max],
                renderer=bpy.app.version_string,
                device="OptiX",
                samples=96,
                geometry="TetGen boundary plus actual saved physical displacement; no exaggeration",
                colormap="isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))",
                clamps="Dark gray surface patches",
                image_size=[4800, 1150],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

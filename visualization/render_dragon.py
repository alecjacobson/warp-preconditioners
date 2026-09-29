"""Render the computed scalar fields with Blender Cycles, never RGB interpolation.

Run: blender -b --factory-startup --python visualization/render_dragon.py -- ...
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colormap import striped_okloop


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area(name, pos, power, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.shape = "DISK"
    light.size = size
    ob = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(ob)
    ob.location = pos
    look_at(ob, (0, 0, 0.4))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/visualization"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1240)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--azimuth", type=float, default=-70)
    parser.add_argument("--elevation", type=float, default=19)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    args.data = args.data.resolve()
    data = np.load(args.data / "fields.npz")
    meta = json.loads((args.data / "solve.json").read_text())
    linear, srgb = striped_okloop()
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 4
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 0.45
    v = data["vertices"].copy()
    center = (v.min(0) + v.max(0)) / 2
    center[2] = v[:, 2].min()
    scale = 2.0 / np.ptp(v[:, 0])
    v = (v - center) * scale
    f = data["faces"].astype(np.int32)
    mesh = bpy.data.meshes.new("original dragon mesh")
    mesh.vertices.add(len(v))
    mesh.vertices.foreach_set("co", v.ravel())
    mesh.loops.add(f.size)
    mesh.loops.foreach_set("vertex_index", f.ravel())
    mesh.polygons.add(len(f))
    mesh.polygons.foreach_set("loop_start", np.arange(len(f), dtype=np.int32) * 3)
    mesh.polygons.foreach_set("loop_total", np.full(len(f), 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(f), dtype=bool))
    mesh.update()
    dragon = bpy.data.objects.new("dragon", mesh)
    bpy.context.collection.objects.link(dragon)
    attribute = mesh.attributes.new("biharmonic_u", "FLOAT", "POINT")
    material = bpy.data.materials.new("26 crisp OKLab isointervals")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.83
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "biharmonic_u"
    normalize = nodes.new("ShaderNodeMapRange")
    normalize.inputs["From Min"].default_value = meta["scalar_range"][0]
    normalize.inputs["From Max"].default_value = meta["scalar_range"][1]
    normalize.clamp = True
    links.new(attr.outputs["Fac"], normalize.inputs["Value"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[-1])
    for i, c in enumerate(linear):
        element = ramp.color_ramp.elements[0] if i == 0 else ramp.color_ramp.elements.new(i / 26)
        element.position = i / 26
        element.color = (*c, 1.0)
    links.new(normalize.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(material)
    # A shadow catcher gives contact shadows on an exactly white background.
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.005))
    ground = bpy.context.object
    ground.name = "white ground shadow catcher"
    ground.is_shadow_catcher = True
    floor_mat = bpy.data.materials.new("white diffuse ground")
    floor_mat.diffuse_color = (1, 1, 1, 1)
    floor_mat.use_nodes = True
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 1, 1, 1)
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1
    ground.data.materials.append(floor_mat)
    area("large soft key", (-3, -4, 6), 440, 4)
    area("front fill", (4, -2, 3), 170, 3.5)
    area("soft rim", (1, 4, 5), 360, 3)
    camera_data = bpy.data.cameras.new("orthographic comparison")
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)
    az, el = math.radians(args.azimuth), math.radians(args.elevation)
    direction = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    target = Vector((0, 0, 0.43))
    camera.location = target + 5 * direction
    look_at(camera, target)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.42
    scene.camera = camera
    scene.render.use_persistent_data = True
    names = ["fsai_cg"] if args.preview else ["jacobi_cr", "fsai_cg"]
    for name in names:
        attribute.data.foreach_set("value", data[name].astype(np.float32))
        mesh.update()
        scene.render.filepath = str(args.data / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    render_meta = dict(
        renderer=bpy.app.version_string,
        engine="Cycles",
        device="OptiX",
        samples=args.samples,
        camera=dict(
            azimuth=args.azimuth,
            elevation=args.elevation,
            orthographic_scale=camera_data.ortho_scale,
        ),
        panel_size=[args.width, args.height],
        colormap="isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))",
        scalar_range=meta["scalar_range"],
        bands=26,
        interpolation="scalar interpolated on triangles, then constant color ramp per shading sample",
        geometry="original mesh; uniform scale and translation for scene placement only",
        display_transform="Standard sRGB",
        palette_srgb=srgb.tolist(),
    )
    (args.data / "render.json").write_text(json.dumps(render_meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

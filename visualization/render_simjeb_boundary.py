"""Render exact undeformed FEM boundary node sets, including a see-through view."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_dragon import area, look_at


def material(name, color, alpha=1):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = 0.72
    p.inputs["Specular IOR Level"].default_value = 0.12
    if alpha < 1:
        transparent = m.node_tree.nodes.new("ShaderNodeBsdfTransparent")
        mix = m.node_tree.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = alpha
        m.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
        m.node_tree.links.new(p.outputs[0], mix.inputs[2])
        m.node_tree.links.new(
            mix.outputs[0], m.node_tree.nodes.get("Material Output").inputs["Surface"]
        )
    return m


def mesh_object(name, vertices, faces, mats, indices=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    for mat in mats:
        mesh.materials.append(mat)
    if indices is not None:
        mesh.polygons.foreach_set("material_index", indices.astype(np.int32))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def dots(name, points, radius, mat):
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * radius
    triangles = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    )
    vertices = (points[:, None, :] + octa).reshape(-1, 3)
    faces = (triangles[None, :, :] + 6 * np.arange(len(points))[:, None, None]).reshape(-1, 3)
    return mesh_object(name, vertices, faces, [mat])


def line(name, a, b, radius, mat):
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.resolution_u = 1
    c.bevel_depth = radius
    c.bevel_resolution = 0
    poly = c.splines.new("POLY")
    poly.points.add(1)
    poly.points[0].co = (*a, 1)
    poly.points[1].co = (*b, 1)
    ob = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)


def arrow(name, a, b, radius, mat):
    a, b = Vector(a), Vector(b)
    d = b - a
    head = min(0.10, d.length * 0.32)
    tip = b - d.normalized() * head
    line(name, a, tip, radius, mat)
    bpy.ops.mesh.primitive_cone_add(
        vertices=12, radius1=radius * 3.1, radius2=0, depth=head, location=(tip + b) / 2
    )
    ob = bpy.context.object
    ob.name = name + " tip"
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.data.materials.append(mat)


def main():
    root = Path("data/simjeb")
    data = np.load(root / "boundary-conditions.npz")
    meta = json.loads((root / "boundary-conditions.json").read_text())
    assert (
        hashlib.sha256((root / "boundary-conditions.npz").read_bytes()).hexdigest()
        == meta["boundary_npz_sha256"]
    )
    v, f = data["vertices"], data["faces"]
    groups, loaded = data["fixed_group"], data["loaded"]
    center = (v.min(0) + v.max(0)) / 2
    az, el = math.radians(-130), math.radians(32)
    right = Vector((-math.sin(az), math.cos(az), 0))
    scale = 2.05 / np.ptp(v @ np.array(right))
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
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.render.resolution_x = 4800
    scene.render.resolution_y = 1700
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
    gray = material("Unconstrained solid", (0.24, 0.27, 0.32))
    ghost = material("See-through geometry", (0.24, 0.27, 0.32), 0.09)
    orange = material("Prescribed displacement zero", (0.92, 0.23, 0.035))
    blue = material("Loaded RBE3 nodes", (0.018, 0.30, 0.76))
    faintorange = material("RBE2 coupling", (0.7, 0.15, 0.018))
    faintblue = material("RBE3 interpolation", (0.025, 0.25, 0.65))
    dark = material("Reference GRID", (0.04, 0.055, 0.075))
    indices = np.zeros(len(f), dtype=np.int32)
    indices[np.all(groups[f] > 0, axis=1)] = 1
    indices[np.all(loaded[f], axis=1)] = 2
    panels = []
    for panel in range(3):
        R = np.array(Matrix.Rotation(math.pi if panel == 1 else 0, 3, "X"))
        shift = np.array((panel - 1) * 2.9 * right)

        def transform(p):
            return ((p - center) * scale) @ R.T + shift

        pos = transform(v)
        obj = mesh_object(
            f"boundary panel {panel}",
            pos,
            f,
            [ghost if panel == 2 else gray, orange, blue],
            None if panel == 2 else indices,
        )
        obj.data.polygons.foreach_set("use_smooth", np.ones(len(f), dtype=bool))
        dots("428 exact fixed nodes", pos[groups > 0], 0.0075, orange)
        dots("753 exact loaded nodes", pos[loaded], 0.0075, blue)
        centers = transform(data["bolt_centers"])
        ref = transform(data["load_center"])
        anchors = {}
        for k, c in enumerate(centers):
            dots(f"B{k + 1} center", np.array([c]), 0.025, dark)
            anchors[f"B{k + 1}"] = c.tolist()
            if panel == 2:
                for p in pos[np.flatnonzero(groups == k + 1)[::8]]:
                    line("RBE2 selected spokes", c, p, 0.0018, faintorange)
        dots("pin reference center", np.array([ref]), 0.027, blue)
        anchors["pin"] = ref.tolist()
        force_dir = np.array([0, 0, 1.0]) @ R.T
        tip = ref + 0.88 * force_dir
        if panel != 1:
            arrow("Resultant 35585.77 N", ref, tip, 0.015, blue)
            anchors["force_tip"] = tip.tolist()
        if panel == 2:
            for p in pos[np.flatnonzero(loaded)[::16]]:
                line("RBE3 selected spokes", ref, p, 0.0018, faintblue)
            # Actual nodal vectors, same scale for all 48 displayed arrows.
            for j in np.flatnonzero(loaded)[::16]:
                end = pos[j] + (data["forces"][j] @ R.T) * 0.0038
                arrow("actual nodal force", pos[j], end, 0.0025, blue)
        origin = shift + np.array([0, 0, -0.82]) - 0.68 * np.array(right)
        axes = {}
        for axis in range(3):
            end = origin + (0.20 * np.eye(3)[axis]) @ R.T
            arrow("XYZ"[axis], origin, end, 0.004, dark)
            axes["XYZ"[axis]] = end.tolist()
        panels.append(
            dict(
                index=panel,
                anchors=anchors,
                axes=axes,
                view=["top / pin side", "underside", "see-through coupling"][panel],
                all_fixed_nodes_drawn=int((groups > 0).sum()),
                all_loaded_nodes_drawn=int(loaded.sum()),
                sampled_force_arrows=48 if panel == 2 else 0,
            )
        )
    area("key", (-3, -6, 8), 900, 8)
    area("fill", (5, -2, 6), 450, 7)
    area("rim", (0, 5, 7), 650, 7)
    camdata = bpy.data.cameras.new("boundary orthographic")
    camera = bpy.data.objects.new("camera", camdata)
    bpy.context.collection.objects.link(camera)
    target = Vector((0, 0, 0.09))
    direction = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    camera.location = target + 6 * direction
    look_at(camera, target)
    camdata.type = "ORTHO"
    camdata.ortho_scale = 8.9
    scene.camera = camera
    bpy.context.view_layer.update()
    for panel in panels:
        for group in ["anchors", "axes"]:
            panel[group] = {
                key: list(world_to_camera_view(scene, camera, Vector(p)))[:2]
                for key, p in panel[group].items()
            }
    scene.render.filepath = str((root / "boundary-render.png").resolve())
    bpy.ops.render.render(write_still=True)
    (root / "boundary-render.json").write_text(
        json.dumps(
            dict(
                panels=panels,
                source=meta,
                dimensions=[4800, 1700],
                fixed_surface_faces=int((indices == 1).sum()),
                loaded_surface_faces=int((indices == 2).sum()),
                line_sampling="Every eighth fixed node for RBE2 spokes; every sixteenth loaded node for RBE3 spokes and nodal force glyphs. All nodes are marked.",
                arrow_scales="Large arrows depict total load direction; small arrows in panel 3 are actual nodal forces at one common scale, separate from the resultant arrow scale.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

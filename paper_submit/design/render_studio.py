"""Fresh, auditable Cycles studio renders of frozen publication evidence.

Run in Blender's Python with -- --phase preview|final|verify.  The verify
phase independently reopens every saved .blend and compares every local
vertex and every pose with the source evidence (absolute tolerance 1e-7).
No training code, dataset, checkpoint, or source evidence is modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "paper_submit/audit/evidence"
OUT = ROOT / "paper_submit/design/renders"
AUDIT = ROOT / "paper_submit/audit/fresh_blender_provenance.json"
ORIGINAL_RENDERER = ROOT / "src/knolling_dataset/blender_worker.py"
PALETTE = {
    "red": (0.62, 0.025, 0.025, 1.0),
    "green": (0.025, 0.42, 0.045, 1.0),
    "blue": (0.025, 0.07, 0.62, 1.0),
}
ROUGHNESS = 0.65
SURFACE = (0.80, 0.80, 0.80, 1.0)
STUDIO_CAMERA = (0.30, 0.04, 0.95)
LOOK_AT = (0.30, 0.20, 0.0)
STUDIO_SCALE = 0.66
OVERHEAD_CAMERA = (0.30, 0.20, 1.0)
OVERHEAD_SCALE = 0.64
LIGHTS = (
    {"name": "Key", "position": (0.02, -0.10, 0.65), "energy_w": 5.0, "size_m": 0.35},
    {"name": "Fill", "position": (0.65, 0.45, 0.70), "energy_w": 1.2, "size_m": 0.70},
)
WORLD_STRENGTH = 0.30
SEED = 20261115
EPSILON = 1e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def frozen_inputs():
    names = ("full8b_choice.json", "full67_choice.json", "v2_smoke_source_records.json", "v2_validation_smoke.json")
    values = {name: json.loads((EVIDENCE / name).read_text()) for name in names}
    snapshots = {name: canonical(value) for name, value in values.items()}
    file_hashes = {name: sha(EVIDENCE / name) for name in names}
    return values, snapshots, file_hashes


def check_original_materials():
    tree = ast.parse(ORIGINAL_RENDERER.read_text())
    palette = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PALETTE" for t in node.targets))
    assert palette == PALETTE
    text = ORIGINAL_RENDERER.read_text()
    assert 'bsdf.inputs["Roughness"].default_value = .65' in text
    assert 'scene.view_settings.view_transform = "Standard"' in text
    assert 'scene.view_settings.look = "None"' in text


def jobs(inputs):
    result = []
    for group in ("full8b", "full67"):
        name = group + "_choice.json"
        record = inputs[name]
        for stage in ("before", "after"):
            field = "source_pose" if stage == "before" else "target_pose"
            result.append({"name": group + "_" + stage, "group": group, "sample_id": record["sample_id"],
                           "objects": record["scene"]["objects"],
                           "poses": {o["object_id"]: o[field] for o in record["scene"]["objects"]},
                           "sources": [name], "pose_origin": "scene.objects[]." + field, "overhead": False})
    predictions = inputs["v2_validation_smoke.json"]
    source_records = inputs["v2_smoke_source_records.json"]
    full = next(p for p in predictions if p["mode"] == "full")
    source = source_records[full["sample_id"]]["source"]
    result.append({"name": "smoke_before", "group": "smoke", "sample_id": full["sample_id"],
                   "objects": source["objects"], "poses": {o["object_id"]: o["source_pose"] for o in source["objects"]},
                   "sources": ["v2_smoke_source_records.json"], "pose_origin": "source.objects[].source_pose", "overhead": False})
    for prediction in predictions:
        source = source_records[prediction["sample_id"]]["source"]
        poses = {p["object_id"]: {k: p[k] for k in ("x_m", "y_m", "theta_rad")} for p in prediction["objects"]}
        assert set(poses) == {o["object_id"] for o in source["objects"]}
        result.append({"name": "smoke_" + prediction["mode"], "group": "smoke", "sample_id": prediction["sample_id"],
                       "objects": source["objects"], "poses": poses,
                       "sources": ["v2_smoke_source_records.json", "v2_validation_smoke.json"],
                       "pose_origin": "v2_validation_smoke.objects[].{x_m,y_m,theta_rad}; includes measured no-op drift",
                       "overhead": False})
    for mode in ("full", "partial"):
        original = next(job for job in result if job["name"] == "smoke_" + mode)
        result.append({**original, "name": original["name"] + "_overhead", "group": "smoke_overhead", "overhead": True})
    return result


def material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = ROUGHNESS
    return mat


def solid(raw, pose):
    points = raw["asset"]["footprint"]
    n = len(points)
    vertices = [(x, y, z) for z in (0.0, raw["asset"]["h_m"]) for x, y in points]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    mesh = bpy.data.meshes.new(raw["object_id"] + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(raw["object_id"], mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = (pose["x_m"], pose["y_m"], 0.0)
    obj.rotation_euler = (0.0, 0.0, pose["theta_rad"])
    obj.data.materials.append(material(raw["object_id"] + "_material", PALETTE[raw["color"]]))
    obj["frozen_object_id"] = raw["object_id"]
    obj["frozen_asset_id"] = raw["asset"]["asset_id"]
    return obj


def aim(obj, target=LOOK_AT):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def make_scene(job, preview):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = False
    scene.cycles.seed = SEED
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1280
    scene.render.resolution_percentage = 50 if preview else 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("Studio World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = WORLD_STRENGTH
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    camera.location = OVERHEAD_CAMERA if job["overhead"] else STUDIO_CAMERA
    aim(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = OVERHEAD_SCALE if job["overhead"] else STUDIO_SCALE
    camera.data.clip_start, camera.data.clip_end = 0.01, 10.0
    scene.camera = camera
    # One seamless surface.  No tray, bevel, rim, texture, or geometry additions.
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.30, 0.20, -0.00001))
    surface = bpy.context.object
    surface.name = "Seamless Surface"
    surface.data.materials.append(material("Neutral Surface", SURFACE))
    for spec in LIGHTS:
        lamp = bpy.data.objects.new(spec["name"], bpy.data.lights.new(spec["name"], "AREA"))
        bpy.context.collection.objects.link(lamp)
        lamp.location = spec["position"]
        aim(lamp)
        lamp.data.energy = spec["energy_w"]
        lamp.data.shape = "DISK"
        lamp.data.size = spec["size_m"]
        lamp.data.color = (1.0, 1.0, 1.0)
    for raw in job["objects"]:
        solid(raw, job["poses"][raw["object_id"]])
    scene["frozen_render_name"] = job["name"]
    scene["frozen_sample_id"] = job["sample_id"]
    bpy.context.view_layer.update()
    return scene


def conditions(scene):
    camera = scene.camera
    return {
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "resolution_percentage": scene.render.resolution_percentage,
        "camera": {"position_m": list(camera.location), "rotation_euler_rad": list(camera.rotation_euler),
                   "type": camera.data.type, "ortho_scale_m": camera.data.ortho_scale,
                   "clip_start_m": camera.data.clip_start, "clip_end_m": camera.data.clip_end,
                   "matrix_world": [list(row) for row in camera.matrix_world]},
        "lights": [{"name": name, "position_m": list(bpy.data.objects[name].location),
                    "rotation_euler_rad": list(bpy.data.objects[name].rotation_euler),
                    "energy_w": bpy.data.objects[name].data.energy, "shape": bpy.data.objects[name].data.shape,
                    "size_m": bpy.data.objects[name].data.size, "color": list(bpy.data.objects[name].data.color)}
                   for name in ("Key", "Fill")],
        "surface": {"location_m": list(bpy.data.objects["Seamless Surface"].location),
                    "rgba": list(bpy.data.materials["Neutral Surface"].diffuse_color), "roughness": ROUGHNESS},
        "world_strength": scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value,
        "world_rgba": list(scene.world.node_tree.nodes["Background"].inputs["Color"].default_value),
        "color_management": {"view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
                             "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma},
        "cycles": {"engine": scene.render.engine, "device": scene.cycles.device, "threads": scene.render.threads,
                   "samples": scene.cycles.samples, "denoising": scene.cycles.use_denoising,
                   "denoiser": scene.cycles.denoiser, "seed": scene.cycles.seed,
                   "adaptive_threshold": scene.cycles.adaptive_threshold,
                   "denoising_use_gpu": getattr(scene.cycles, "denoising_use_gpu", False)},
    }


def projection(scene, job):
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100
    def pixel(x, y, z=0):
        projected = world_to_camera_view(scene, scene.camera, Vector((x, y, z)))
        return [projected.x * width, (1.0 - projected.y) * height]
    return {"pixel_origin": "top_left", "resolution": [width, height],
            "workspace_corners_px": {str((x, y)): pixel(x, y) for x, y in ((0, 0), (.6, 0), (.6, .4), (0, .4))},
            "object_centers_base_px": {key: pixel(pose["x_m"], pose["y_m"]) for key, pose in job["poses"].items()},
            "object_centers_top_px": {raw["object_id"]: pixel(job["poses"][raw["object_id"]]["x_m"],
                                                              job["poses"][raw["object_id"]]["y_m"], raw["asset"]["h_m"])
                                       for raw in job["objects"]}}


def verify_geometry(scene, job):
    per_object = []
    for raw in job["objects"]:
        obj = bpy.data.objects[raw["object_id"]]
        pose = job["poses"][raw["object_id"]]
        expected_location = [pose["x_m"], pose["y_m"], 0.0]
        expected_rotation = [0.0, 0.0, pose["theta_rad"]]
        vertices = [(x, y, z) for z in (0.0, raw["asset"]["h_m"]) for x, y in raw["asset"]["footprint"]]
        assert len(obj.data.vertices) == len(vertices)
        max_vertex_error = max(abs(float(vertex.co[axis]) - expected[axis])
                               for vertex, expected in zip(obj.data.vertices, vertices) for axis in range(3))
        max_location_error = max(abs(float(actual) - expected) for actual, expected in zip(obj.location, expected_location))
        max_rotation_error = max(abs(float(actual) - expected) for actual, expected in zip(obj.rotation_euler, expected_rotation))
        max_scale_error = max(abs(float(actual) - 1.0) for actual in obj.scale)
        base_color = obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
        max_color_error = max(abs(float(actual) - expected) for actual, expected in zip(base_color, PALETTE[raw["color"]]))
        roughness_error = abs(obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value - ROUGHNESS)
        assert len(obj.modifiers) == 0
        assert all(not face.use_smooth for face in obj.data.polygons)
        assert max(max_vertex_error, max_location_error, max_rotation_error, max_scale_error, max_color_error, roughness_error) <= EPSILON
        per_object.append({"object_id": raw["object_id"], "asset_id": raw["asset"]["asset_id"],
                           "vertices_compared": len(vertices), "max_vertex_abs_error_m": max_vertex_error,
                           "max_location_abs_error_m": max_location_error, "max_rotation_abs_error_rad": max_rotation_error,
                           "max_scale_abs_error": max_scale_error, "max_base_color_error": max_color_error,
                           "roughness_error": roughness_error, "modifier_count": len(obj.modifiers), "pass": True})
    assert len([obj for obj in scene.objects if obj.type == "MESH"]) == len(job["objects"]) + 1
    return {"absolute_tolerance": EPSILON, "comparison": "every vertex and all location/rotation/scale components against frozen evidence",
            "objects": per_object, "pass": True}


def render(job, preview, source_hashes):
    scene = make_scene(job, preview)
    stem = "preview_" + job["name"] if preview else job["name"]
    png, blend = OUT / (stem + ".png"), OUT / (stem + ".blend")
    scene.render.filepath = str(png)
    metadata = {"name": job["name"], "group": job["group"], "sample_id": job["sample_id"],
                "source_files": {str(EVIDENCE / name): source_hashes[name] for name in job["sources"]},
                "pose_origin": job["pose_origin"], "conditions": conditions(scene), "projection": projection(scene, job),
                "pre_save_geometry_verification": verify_geometry(scene, job)}
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    bpy.ops.render.render(write_still=True)
    metadata.update({"image_path": str(png), "image_sha256": sha(png), "blend_path": str(blend), "blend_sha256": sha(blend)})
    (OUT / (stem + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print("STUDIO_RENDER_DONE " + stem, flush=True)


def verify(all_jobs, source_hashes):
    records, groups = [], {}
    for job in all_jobs:
        path = OUT / (job["name"] + ".blend")
        bpy.ops.wm.open_mainfile(filepath=str(path))
        scene = bpy.context.scene
        record = json.loads((OUT / (job["name"] + ".json")).read_text())
        assert record["blend_sha256"] == sha(path)
        assert record["image_sha256"] == sha(record["image_path"])
        record["independent_reopen_verification"] = verify_geometry(scene, job)
        saved_conditions = conditions(scene)
        assert canonical(record["conditions"]) == canonical(saved_conditions)
        signature = hashlib.sha256(canonical(saved_conditions).encode()).hexdigest()
        if job["group"] in groups:
            assert groups[job["group"]] == signature
        groups[job["group"]] = signature
        record["conditions_sha256"] = signature
        record["same_conditions_within_group"] = True
        records.append(record)
    provenance = {
        "description": "Fresh studio setup authored from frozen JSON and original project renderer only; no earlier publication renderer/compositor reused.",
        "blender_version": bpy.app.version_string, "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha(__file__), "reference_renderer_path": str(ORIGINAL_RENDERER),
        "reference_renderer_sha256": sha(ORIGINAL_RENDERER), "reference_verified": {"palette": PALETTE, "roughness": ROUGHNESS, "view_transform": "Standard", "look": "None"},
        "source_file_hashes": {str(EVIDENCE / name): digest for name, digest in source_hashes.items()},
        "frozen_inputs_unchanged": True, "all_renders_cpu_only": True, "threads": 8,
        "natural_geometry": "Exact original footprint vertices extruded from z=0 to original h_m; flat faces; no modifiers; unit scale.",
        "surface": "Neutral seamless plane, no raised tray; plane z=-0.00001 m.",
        "conditions_by_group": groups, "renders": records, "pass": True,
    }
    AUDIT.write_text(json.dumps(provenance, indent=2) + "\n")
    print("STUDIO_INDEPENDENT_VERIFICATION_DONE " + str(AUDIT), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preview", "final", "verify"))
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    check_original_materials()
    inputs, snapshots, source_hashes = frozen_inputs()
    all_jobs = jobs(inputs)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "verify":
        verify(all_jobs, source_hashes)
    else:
        selected = all_jobs[:2] if args.phase == "preview" else all_jobs
        if args.only:
            selected = [job for job in selected if job["name"] in args.only]
        assert selected
        for job in selected:
            render(job, args.phase == "preview", source_hashes)
    assert {name: canonical(value) for name, value in inputs.items()} == snapshots
    assert {name: sha(EVIDENCE / name) for name in inputs} == source_hashes


if __name__ == "__main__":
    main()

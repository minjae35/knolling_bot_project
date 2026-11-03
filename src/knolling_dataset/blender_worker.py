"""Runs ONLY in Blender's Python. Input is a JSON batch, never Python snippets."""
import json
import math
from pathlib import Path
import random
import sys
import bpy

PALETTE = {"red":(.62,.025,.025,1),"green":(.025,.42,.045,1),"blue":(.025,.07,.62,1)}


def material(name,color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = .65
    return mat


def solid(raw,pose,index):
    asset = raw["asset"]
    points = asset["footprint"]
    n = len(points)
    verts = [(x,y,z) for z in (0,asset["h_m"]) for x,y in points]
    faces = [tuple(reversed(range(n))),tuple(range(n,2*n))]
    faces += [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
    mesh = bpy.data.meshes.new(raw["object_id"])
    mesh.from_pydata(verts,[],faces)
    mesh.update()
    obj = bpy.data.objects.new(raw["object_id"],mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = (pose["x_m"],pose["y_m"],0)
    obj.rotation_euler = (0,0,pose["theta_rad"])
    obj.data.materials.append(material(raw["object_id"]+"_mat",PALETTE[raw["color"]]))
    obj.color = (index/255,0,0,1)
    return obj


def render(job):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    rng = random.Random(job["render_seed"])
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x,scene.render.resolution_y = 576,384
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.cycles.seed = job["render_seed"] % 2147483647
    scene.cycles.use_animated_seed = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1,1,1,1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .35
    scene.world = world
    camera = bpy.data.objects.new("Camera",bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    camera.location = (.3,.2,1)
    camera.rotation_euler = (0,0,0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = .6
    camera.data.clip_start,camera.data.clip_end = .01,10
    scene.camera = camera
    bpy.ops.mesh.primitive_plane_add(size=2,location=(.3,.2,-.001))
    table = bpy.context.object
    table.name = "Table"
    gray = rng.uniform(.35,.55)
    table.data.materials.append(material("table",(gray,gray,gray,1)))
    table.color = (0,0,0,1)
    lamp = bpy.data.objects.new("Area",bpy.data.lights.new("Area","AREA"))
    bpy.context.collection.objects.link(lamp)
    lamp.location = (rng.uniform(.15,.45),rng.uniform(.1,.3),.65)
    lamp.data.energy,lamp.data.shape,lamp.data.size = rng.uniform(3,5),"DISK",.5
    for i,raw in enumerate(job["scene"]["objects"],1):
        solid(raw,raw[job["pose_field"]],i)
    out = Path(job["output_dir"])
    out.mkdir(parents=True,exist_ok=True)
    scene.render.filepath = str(out/"rgb.png")
    bpy.ops.render.render(write_still=True)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_specular_highlight = False
    scene.display.shading.show_object_outline = False
    scene.display.shading.background_type = "WORLD"
    scene.world.color = (0,0,0)
    scene.display.render_aa = "OFF"
    scene.view_settings.view_transform = "Raw"
    scene.render.filepath = str(out/"mask_rgb.png")
    bpy.ops.render.render(write_still=True)
    metadata = {"blender_version":bpy.app.version_string,"rgb_engine":"CYCLES","samples":32,
                "mask_engine":"BLENDER_WORKBENCH","mask_antialiasing":False,
                "resolution":[576,384],"camera_type":"ORTHO","ortho_scale_m":.6,
                "camera_position_m":[.3,.2,1],"render_seed":job["render_seed"],
                "instance_ids":{str(i):o["object_id"] for i,o in enumerate(job["scene"]["objects"],1)}}
    (out/"render.json").write_text(json.dumps(metadata,indent=2))
    print("KNOLLING_RENDER_DONE "+job["key"],flush=True)


if __name__ == "__main__":
    batch = json.loads(Path(sys.argv[sys.argv.index("--")+1]).read_text())
    for job in batch:
        render(job)

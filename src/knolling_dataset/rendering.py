"""M6 host adapter and an independent raster/geometry consistency oracle."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from shapely import contains_xy
from .schema import IMAGE_WIDTH, IMAGE_HEIGHT, from_pixel, to_pixel, canonical_json, scene_dict
from .validator import polygon


def blender_executable():
    path = os.environ.get("BLENDER_EXECUTABLE") or shutil.which("blender")
    if not path:
        raise RuntimeError("Blender not found; set BLENDER_EXECUTABLE")
    return path


def make_job(scene,pose_field,output_dir,key,render_seed=None):
    if pose_field not in ("source_pose","target_pose"):
        raise ValueError("invalid render pose field")
    if any(getattr(o,pose_field) is None for o in scene.objects):
        raise ValueError("cannot render missing poses")
    return {"key":key,"scene":scene_dict(scene),"pose_field":pose_field,
            "output_dir":str(Path(output_dir).resolve()),"render_seed":scene.seed if render_seed is None else render_seed}


def render_batch(jobs,work_dir):
    if not jobs:
        return
    if len({j["output_dir"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate render output directory")
    work = Path(work_dir)
    work.mkdir(parents=True,exist_ok=True)
    batch_path = work/"render_jobs.json"
    batch_path.write_text(canonical_json(jobs))
    with (work/"blender.log").open("w") as log:
        subprocess.run([blender_executable(),"--background","--factory-startup","--threads","4",
                        "--python-exit-code","1","--python",str(Path(__file__).with_name("blender_worker.py")),
                        "--",str(batch_path.resolve())],stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=120+len(jobs)*90)
    for job in jobs:
        out = Path(job["output_dir"])
        raw = np.asarray(Image.open(out/"mask_rgb.png").convert("RGB"))
        if np.any(raw[:,:,1:]) or not set(np.unique(raw[:,:,0])).issubset(set(range(len(job["scene"]["objects"])+1))):
            raise RuntimeError("mask render did not preserve exact integer IDs")
        Image.fromarray(raw[:,:,0].astype(np.uint16)).save(out/"instance_mask.png")
        (out/"mask_rgb.png").unlink()


def validate_render(scene,pose_field,directory):
    directory = Path(directory)
    rgb = np.asarray(Image.open(directory/"rgb.png").convert("RGB"))
    mask = np.asarray(Image.open(directory/"instance_mask.png"))
    errors,objects = [],[]
    if rgb.shape != (IMAGE_HEIGHT,IMAGE_WIDTH,3) or mask.shape != (IMAGE_HEIGHT,IMAGE_WIDTH):
        return {"valid":False,"errors":["image_dimensions"],"objects":[]}
    if set(np.unique(mask)) != set(range(len(scene.objects)+1)):
        errors.append("instance_ids")
    clipped_fraction = float(np.mean(np.all(rgb >= 250,axis=2)))
    if clipped_fraction > .01:
        errors.append("overexposed_render")
    yy,xx = np.mgrid[:IMAGE_HEIGHT,:IMAGE_WIDTH]
    wx = (xx+.5)/IMAGE_WIDTH*.6
    wy = (1-(yy+.5)/IMAGE_HEIGHT)*.4
    for i,obj in enumerate(scene.objects,1):
        pose = getattr(obj,pose_field)
        p = polygon(obj,pose,envelope=False)
        expected = contains_xy(p,wx,wy)
        actual = mask == i
        union = np.count_nonzero(expected|actual)
        iou = np.count_nonzero(expected&actual)/union if union else 0
        pixels = np.argwhere(actual)
        center = pixels.mean(axis=0)[::-1] if len(pixels) else np.array([1e9,1e9])
        centroid_error = float(np.linalg.norm(center-np.asarray(to_pixel(pose.x_m,pose.y_m))))
        # Exclude a 2 mm silhouette band for color/lighting checks.
        interior = contains_xy(p.buffer(-.002),wx,wy)&actual
        channel = {"red":0,"green":1,"blue":2}[obj.color]
        color_accuracy = float(np.mean(np.argmax(rgb[interior],axis=1) == channel)) if interior.any() else 0
        if iou < .98:
            errors.append(f"mask_geometry:{obj.object_id}")
        if centroid_error > 1:
            errors.append(f"centroid_projection:{obj.object_id}")
        if color_accuracy < .99:
            errors.append(f"rgb_mask_color:{obj.object_id}")
        ys,xs = np.where(actual)
        objects.append({"object_id":obj.object_id,"instance_id":i,"mask_iou":iou,
                        "centroid_error_px":centroid_error,"rgb_color_accuracy":color_accuracy,
                        "visible_pixels":len(pixels),"bbox_xyxy_px":[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(pixels) else None})
    return {"valid":not errors,"errors":errors,"objects":objects,"clipped_fraction":clipped_fraction}

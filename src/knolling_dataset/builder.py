"""M7: deterministic annotations, real renders and audited sample manifests."""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
from .assets import build_registry, registry_dict
from .instructions import generate_instruction
from .scenes import generate_scene, SamplingFailure
from .schema import (SPLITS, canonical_json, digest, scene_dict, scene_from_dict,
                     instruction_from_dict, to_normalized, to_pixel)
from .teacher import solve
from .validator import validate_layout, validate_source, polygon
from .rendering import make_job, render_batch, validate_render


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plan(count,seed):
    if count < 10 or count > 100 or count%10:
        raise ValueError("Phase 1 pilot count must be a multiple of 10, from 10 to 100")
    rng = random.Random(seed)
    modes = ["full"]*(count//2)+["partial"]*(3*count//10)+["no_op"]*(count//5)
    splits = ["train"]*(7*count//10)+sum(([s]*(count//10) for s in SPLITS[1:]),[])
    rng.shuffle(modes)
    rng.shuffle(splits)
    return list(zip(modes,splits))


def _record(source,ins,result,sample_id,role,noop_type=None):
    source_validation = validate_source(source,result.mode)
    validation = validate_layout(source,result.target,ins,result.mode)
    if not source_validation.valid or not validation.valid:
        raise RuntimeError(f"invalid accepted sample: {source_validation.errors}, {validation.errors}")
    return {"schema_version":"1.2","sample_id":sample_id,"scene_family_id":source.family_id,
            "split":source.split,"mode":result.mode,"role":role,"no_op_type":noop_type,
            "status":result.status,"instruction":asdict(ins),"source":scene_dict(source),
            "target":scene_dict(result.target),"rows":result.rows,"relations":result.relations,
            "validation":validation.as_dict(),"source_validation":source_validation.as_dict(),
            "teacher_candidates_checked":result.candidates_checked}


def prepare_annotations(count=100,seed=1202):
    registry = build_registry()
    records,rejections = [],[]
    primary_attempts,counter_attempts,noop_number = 0,0,0
    for index,(desired,split) in enumerate(_plan(count,seed)):
        family = "sf_"+digest(["pilot-v1.2",seed,index])[:20]
        noop_type = ("repeat" if noop_number < round(count*.14) else "keep_all") if desired == "no_op" else None
        if desired == "no_op":
            noop_number += 1
        for attempt in range(100):
            primary_attempts += 1
            scene_seed = seed*100000+index*100+attempt
            try:
                source = generate_scene(registry[split],scene_seed,split,2+index%7,
                                        desired,family_id=family)
                ins_mode = "keep_all" if noop_type == "keep_all" else "partial" if desired == "partial" else "full"
                ins = generate_instruction(source,scene_seed+1,ins_mode)
            except (SamplingFailure,ValueError) as exc:
                rejections.append({"family_id":family,"attempt":attempt,"stage":"source_instruction","reason":str(exc)})
                continue
            result = solve(source,ins)
            if result.status != "ok":
                rejections.append({"family_id":family,"attempt":attempt,"stage":"teacher","reason":result.reason,"status":result.status})
                continue
            if noop_type == "repeat":
                source = replace(source,objects=tuple(replace(o,source_pose=o.target_pose,target_pose=None,
                                 locked=False,move_label=False) for o in result.target.objects))
                result = solve(source,ins)
            if result.mode != desired:
                rejections.append({"family_id":family,"attempt":attempt,"stage":"mode_balance","reason":f"expected_{desired}_got_{result.mode}"})
                continue
            records.append(_record(source,ins,result,f"scene_{index:04d}_primary","primary",noop_type))
            break
        else:
            raise RuntimeError(f"100 retries exhausted for {family}; generation stopped")
    # Same RGB, meaningfully different instructions/targets; same family split.
    required = round(count*.30)
    for primary in list(records):
        if primary["mode"] != "full" or required == 0:
            continue
        source = scene_from_dict(primary["source"])
        original = instruction_from_dict(primary["instruction"])
        counter_attempts += 1
        ins = generate_instruction(source,source.seed+700000,"full",language=original.language,
              group_by=original.group_by,sort_direction=original.sort_direction,
              axis_alignment="vertical" if original.axis_alignment == "horizontal" else "horizontal",
              gap_m=original.gap_m,region="full")
        result = solve(source,ins)
        if result.status != "ok" or result.mode == "no_op":
            rejections.append({"family_id":source.family_id,"stage":"counterfactual","reason":result.reason or "no_op","status":result.status})
            continue
        records.append(_record(source,ins,result,primary["sample_id"].replace("primary","counterfactual"),"counterfactual"))
        required -= 1
    if required:
        raise RuntimeError("not enough distinct valid counterfactual instruction families")
    stats = {"seed":seed,"requested_scenes":count,"accepted_scenes":count,"accepted_samples":len(records),
             "primary_attempts":primary_attempts,"primary_rejections":primary_attempts-count,
             "primary_acceptance_rate":count/primary_attempts,
             "counterfactual_attempts":counter_attempts,"counterfactual_accepted":round(count*.30),
             "rejection_categories":dict(Counter(r["reason"] for r in rejections)),"rejections":rejections,
             "primary_mode_counts":dict(Counter(r["mode"] for r in records if r["role"] == "primary")),
             "sample_mode_counts":dict(Counter(r["mode"] for r in records)),
             "primary_split_counts":dict(Counter(r["split"] for r in records if r["role"] == "primary")),
             "sample_split_counts":dict(Counter(r["split"] for r in records)),
             "language_counts":dict(Counter(r["instruction"]["language"] for r in records)),
             "object_count_histogram":dict(Counter(len(r["source"]["objects"]) for r in records if r["role"] == "primary")),
             "no_op_types":dict(Counter(r["no_op_type"] for r in records if r["no_op_type"]))}
    return registry,records,stats


def _object_annotations(record,source_render,target_render):
    source = scene_from_dict(record["source"])
    target = scene_from_dict(record["target"])
    output = []
    for i,(src,dst) in enumerate(zip(source.objects,target.objects)):
        bounds = polygon(src,src.source_pose).bounds
        output.append({"object_id":src.object_id,"instance_id":i+1,"asset_id":src.asset.asset_id,
                       "asset_family_id":src.asset.family_id,"color":src.color,"shape":src.asset.shape,
                       "dimensions_m":[src.asset.a_m,src.asset.b_m,src.asset.h_m],
                       "source_center_uv":to_normalized(src.source_pose.x_m,src.source_pose.y_m),
                       "target_center_uv":to_normalized(dst.target_pose.x_m,dst.target_pose.y_m),
                       "source_center_px":to_pixel(src.source_pose.x_m,src.source_pose.y_m),
                       "target_center_px":to_pixel(dst.target_pose.x_m,dst.target_pose.y_m),
                       "source_envelope_xyxy_uv":[bounds[0]/.6,bounds[1]/.4,bounds[2]/.6,bounds[3]/.4],
                       "source_mask_bbox_xyxy_px":source_render["objects"][i]["bbox_xyxy_px"],
                       "target_mask_bbox_xyxy_px":target_render["objects"][i]["bbox_xyxy_px"],
                       "locked":dst.locked,"move_label":dst.move_label})
    return output


def build_dataset(output,count=100,seed=1202):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory is not empty; choose a new directory")
    registry,records,stats = prepare_annotations(count,seed)
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"asset_registry.json",registry_dict(registry))
    write_json(root/"generation.json",stats)
    jobs,rendered_sources = [],set()
    for record in records:
        source,target = scene_from_dict(record["source"]),scene_from_dict(record["target"])
        source_dir = Path("scenes")/source.family_id/"source"
        target_dir = source_dir if record["mode"] == "no_op" else Path("samples")/record["sample_id"]/"target"
        if source.family_id not in rendered_sources:
            jobs.append(make_job(source,"source_pose",root/source_dir,source.family_id))
            rendered_sources.add(source.family_id)
        if record["mode"] != "no_op":
            jobs.append(make_job(target,"target_pose",root/target_dir,record["sample_id"]))
        record["source_render_dir"],record["target_render_dir"] = str(source_dir),str(target_dir)
        write_json(root/"samples"/record["sample_id"]/"annotation.json",record)
    print(f"Prepared {count} scene families / {len(records)} samples; rendering {len(jobs)} views",flush=True)
    render_batch(jobs,root/"work")
    manifest,source_reports = [],{}
    for record in records:
        source,target = scene_from_dict(record["source"]),scene_from_dict(record["target"])
        if source.family_id not in source_reports:
            source_reports[source.family_id] = validate_render(source,"source_pose",root/record["source_render_dir"])
        src_report = source_reports[source.family_id]
        dst_report = validate_render(target,"target_pose",root/record["target_render_dir"])
        if not src_report["valid"] or not dst_report["valid"]:
            write_json(root/"FAILED.json",{"sample_id":record["sample_id"],"source":src_report,"target":dst_report})
            raise RuntimeError(f"render validation failed: {record['sample_id']}")
        record["render_validation"] = {"source":src_report,"target":dst_report}
        record["objects"] = _object_annotations(record,src_report,dst_report)
        hashes = {}
        for role in ("source","target"):
            for name in ("rgb.png","instance_mask.png","render.json"):
                path = Path(record[role+"_render_dir"])/name
                hashes[str(path)] = file_hash(root/path)
        record["file_hashes"] = hashes
        annotation = Path("samples")/record["sample_id"]/"annotation.json"
        write_json(root/annotation,record)
        manifest.append({"sample_id":record["sample_id"],"scene_family_id":source.family_id,"split":source.split,
                         "mode":record["mode"],"role":record["role"],"annotation":str(annotation),
                         "annotation_sha256":file_hash(root/annotation)})
    (root/"manifest.jsonl").write_text("".join(canonical_json(m)+"\n" for m in manifest))
    try:
        revision = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        revision = "unknown"
    write_json(root/"provenance.json",{"schema_version":"1.2","code_revision":revision,"python":platform.python_version(),
              "seed":seed,"scene_count":count,"sample_count":len(records),"rendered_views":len(jobs),
              "coordinate_units":"meters, radians; uv dimensionless; pixel centers are integers",
              "camera":{"width_m":.6,"height_m":.4,"image_width":576,"image_height":384},
              "rgb_color_space":"sRGB","mask_dtype":"uint16","background_id":0})
    from .qc import write_qc
    write_qc(root,records,stats)
    return records,stats

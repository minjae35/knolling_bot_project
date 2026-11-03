"""M8: independently recompute lineage, geometry and file integrity checks."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .assets import geometry_fingerprint,asset_split
from .schema import scene_from_dict,instruction_from_dict,digest,SPLITS
from .validator import validate_layout,validate_source,inferred_rows,polygon
from .rendering import validate_render
from .builder import file_hash,write_json


def _physical_scene_fingerprint(scene):
    # Exclude all IDs, split labels, seeds and colors' asset metadata aliases.
    objects = [{"shape":o.asset.shape,"footprint":o.asset.footprint,
                "h_m":o.asset.h_m,"color":o.color,"pose":[o.source_pose.x_m,o.source_pose.y_m,o.source_pose.theta_rad]}
               for o in scene.objects]
    return digest(sorted(objects,key=digest))


def audit_records(records):
    errors=[]
    families,assets,geometry_splits,physical_splits = (defaultdict(set) for _ in range(4))
    family_sources = defaultdict(set)
    samples = set()
    def fail(sid,code):
        errors.append(f"{sid}:{code}")
    for record in records:
        sid = record["sample_id"]
        if sid in samples:
            fail(sid,"duplicate_sample_id")
        samples.add(sid)
        source,target = scene_from_dict(record["source"]),scene_from_dict(record["target"])
        ins = instruction_from_dict(record["instruction"])
        split,family = record["split"],record["scene_family_id"]
        if (source.split,source.family_id) != (split,family):
            fail(sid,"lineage_metadata")
        families[family].add(split)
        fingerprint = _physical_scene_fingerprint(source)
        family_sources[family].add(fingerprint)
        physical_splits[fingerprint].add(split)
        for obj in source.objects:
            asset = obj.asset
            fingerprint = geometry_fingerprint(asset)
            assets[asset.family_id].add(split)
            geometry_splits[fingerprint].add(split)
            if asset.family_id != "af_"+fingerprint[:20]:
                fail(sid,"asset_family_fingerprint")
            if asset_split(asset.family_id) != split:
                fail(sid,"asset_pool_split")
        for code in validate_source(source,record["mode"]).errors:
            fail(sid,"source_"+code)
        for code in validate_layout(source,target,ins,record["mode"]).errors:
            fail(sid,code)
        if record["mode"] == "no_op":
            if record["rows"] or record["relations"]:
                fail(sid,"noop_relations")
        else:
            rows = inferred_rows([o for o in target.objects if o.object_id in ins.selected_ids])
            row_ids = [[o.object_id for o in row] for row in rows]
            if row_ids != record["rows"]:
                fail(sid,"row_labels")
            h = [{"ids":[a,b],"gap_m":ins.gap_m} for row in row_ids for a,b in zip(row,row[1:])]
            v = [{"upper_ids":a,"lower_ids":b,"gap_m":ins.gap_m*(2 if ins.group_by == "color" and up[0].color != down[0].color else 1)}
                 for a,b,up,down in zip(row_ids,row_ids[1:],rows,rows[1:])]
            if record["relations"] != {"horizontal_neighbors":h,"vertical_row_neighbors":v}:
                fail(sid,"relation_labels")
        if "objects" in record:
            by_source = {o.object_id:o for o in source.objects}
            by_target = {o.object_id:o for o in target.objects}
            labels = record["objects"]
            if len(labels) != len(source.objects) or {o["object_id"] for o in labels} != by_source.keys():
                fail(sid,"annotation_object_preservation")
            else:
                for obj in labels:
                    src,dst = by_source[obj["object_id"]],by_target[obj["object_id"]]
                    if obj["asset_id"] != src.asset.asset_id or obj["locked"] != dst.locked or obj["move_label"] != dst.move_label:
                        fail(sid,"annotation_identity")
                    expected = {"asset_family_id":src.asset.family_id,"color":src.color,"shape":src.asset.shape,
                                "dimensions_m":[src.asset.a_m,src.asset.b_m,src.asset.h_m]}
                    if any(obj[k] != v for k,v in expected.items()):
                        fail(sid,"annotation_asset_metadata")
                    bounds = polygon(src,src.source_pose).bounds
                    if not np.allclose(obj["source_envelope_xyxy_uv"],
                                       [bounds[0]/.6,bounds[1]/.4,bounds[2]/.6,bounds[3]/.4],atol=1e-12,rtol=0):
                        fail(sid,"source_box_units")
                    for role,p in (("source",src.source_pose),("target",dst.target_pose)):
                        expected_uv = [p.x_m/.6,p.y_m/.4]
                        expected_px = [576*p.x_m/.6-.5,384*(1-p.y_m/.4)-.5]
                        if not np.allclose(obj[role+"_center_uv"],expected_uv,atol=1e-12,rtol=0):
                            fail(sid,"coordinate_units")
                        if not np.allclose(obj[role+"_center_px"],expected_px,atol=1e-9,rtol=0):
                            fail(sid,"pixel_projection")
    for label,mapping in (("scene_family_leakage",families),("asset_family_leakage",assets),
                          ("geometry_alias_leakage",geometry_splits),("duplicate_scene_leakage",physical_splits)):
        errors += [f"{key}:{label}" for key,splits in mapping.items() if len(splits) != 1]
    errors += [f"{key}:family_source_drift" for key,hashes in family_sources.items() if len(hashes) != 1]
    return {"valid":not errors,"errors":errors,"scene_families":len(families),"samples":len(records),
            "asset_families":len(assets),"split_sample_counts":dict(Counter(r["split"] for r in records))}


def _safe_path(root,relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("dataset path escapes root")
    return path


def audit_dataset(root,verify_renders=True):
    root = Path(root)
    complete = root/"COMPLETE.json"
    if complete.exists():
        complete.unlink()  # A failed or interrupted re-audit cannot leave stale success.
    manifest = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    records,integrity_errors,checked_files,render_cache = [],[],set(),{}
    for entry in manifest:
        path = _safe_path(root,entry["annotation"])
        if file_hash(path) != entry["annotation_sha256"]:
            integrity_errors.append(entry["sample_id"]+":annotation_hash")
        record = json.loads(path.read_text())
        records.append(record)
        if any(entry[k] != record[k] for k in ("sample_id","scene_family_id","split","mode","role")):
            integrity_errors.append(entry["sample_id"]+":manifest_metadata")
        for relative,expected in record["file_hashes"].items():
            if file_hash(_safe_path(root,relative)) != expected:
                integrity_errors.append(entry["sample_id"]+":file_hash:"+relative)
            checked_files.add(relative)
        for role in ("source","target"):
            scene = scene_from_dict(record[role])
            directory = _safe_path(root,record[role+"_render_dir"])
            metadata = json.loads((directory/"render.json").read_text())
            expected_ids = {str(i):o.object_id for i,o in enumerate(scene.objects,1)}
            if metadata["instance_ids"] != expected_ids or metadata["resolution"] != [576,384] or metadata["ortho_scale_m"] != .6:
                integrity_errors.append(entry["sample_id"]+":render_metadata")
            if verify_renders:
                key = (str(directory),digest(record[role]),role)
                if key not in render_cache:
                    render_cache[key] = validate_render(scene,role+"_pose",directory)
                result = render_cache[key]
                if not result["valid"]:
                    integrity_errors.extend(entry["sample_id"]+":"+c for c in result["errors"])
                byid = {o["object_id"]:o for o in record["objects"]}
                for obj in result["objects"]:
                    annotation = byid[obj["object_id"]]
                    if annotation["instance_id"] != obj["instance_id"] or annotation[role+"_mask_bbox_xyxy_px"] != obj["bbox_xyxy_px"]:
                        integrity_errors.append(entry["sample_id"]+":mask_annotation")
    result = audit_records(records)
    provenance = json.loads((root/"provenance.json").read_text())
    if (result["samples"],result["scene_families"]) != (provenance["sample_count"],provenance["scene_count"]):
        integrity_errors.append("manifest_count")
    result["errors"].extend(integrity_errors)
    result["valid"] = not result["errors"]
    result["verified_file_count"] = len(checked_files)+len(manifest)
    result["renders_revalidated"] = verify_renders
    write_json(root/"audit.json",result)
    # Explicit split manifests are derivative indexes, never independent split decisions.
    for split in SPLITS:
        entries = [m for m in manifest if m["split"] == split]
        (root/f"{split}.jsonl").write_text("".join(json.dumps(m,sort_keys=True)+"\n" for m in entries))
    if result["valid"] and verify_renders:
        write_json(complete,{"manifest_sha256":file_hash(root/"manifest.jsonl"),"scenes":result["scene_families"],
                            "samples":result["samples"],"human_review":"pending"})
    elif complete.exists():
        complete.unlink()
    return result

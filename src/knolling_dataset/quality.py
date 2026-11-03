"""Dataset diagnostics: validity is separate from training coverage and aesthetics."""
from collections import Counter,defaultdict
from dataclasses import asdict,replace
import math
from .schema import scene_from_dict,instruction_from_dict,digest,wrap_angle,angle_error,REGIONS
from .validator import polygon,validate_layout
from .teacher_balanced import movement_cost,observed_rows


def target_hash(scene, *, canonical=True):
    values={}
    for o in scene.objects:
        p=o.target_pose
        values[o.object_id]=[round(p.x_m,12),round(p.y_m,12),round(wrap_angle(p.theta_rad),12)] if canonical else asdict(p)
    return digest(values)


def target_changed(before,after):
    a={o.object_id:o.target_pose for o in before.objects}
    b={o.object_id:o.target_pose for o in after.objects}
    return a.keys()!=b.keys() or any(math.hypot(a[k].x_m-b[k].x_m,a[k].y_m-b[k].y_m)>1e-12 or
                                   angle_error(a[k].theta_rad,b[k].theta_rad)>1e-12 for k in a)


def layout_metrics(source,target,ins):
    selected=[o for o in target.objects if o.object_id in ins.selected_ids]
    if not selected:
        return {"J_move":0.0,"J_shape":None}
    boxes=[polygon(o,o.target_pose).bounds for o in selected]
    width=max(b[2] for b in boxes)-min(b[0] for b in boxes)
    height=max(b[3] for b in boxes)-min(b[1] for b in boxes)
    l,b,r,t=REGIONS[ins.region]
    return {"J_move":movement_cost(source,ins,{o.object_id:o.target_pose for o in selected}),
            "J_shape":max(width/(r-l),height/(t-b))}


def object_movements(source,target):
    by_id={o.object_id:o for o in target.objects}
    return [{"object_id":o.object_id,"locked":by_id[o.object_id].locked,
             "translation_mm":1000*math.hypot(by_id[o.object_id].target_pose.x_m-o.source_pose.x_m,
                                              by_id[o.object_id].target_pose.y_m-o.source_pose.y_m),
             "rotation_deg":math.degrees(angle_error(by_id[o.object_id].target_pose.theta_rad,o.source_pose.theta_rad)),
             "move_label":by_id[o.object_id].move_label} for o in source.objects]


def partial_quality(before,after):
    reports=[]
    by_id={r["sample_id"]:r for r in after}
    for old in before:
        if old["mode"]!="partial": continue
        new=by_id.get(old["sample_id"])
        source=scene_from_dict(old["source"]); ins=instruction_from_dict(old["instruction"])
        targets={"legacy":scene_from_dict(old["target"])}
        if new: targets["v1_3"]=scene_from_dict(new["target"])
        count=len(ins.selected_ids)
        all_ids=tuple(o.object_id for o in source.objects)
        organized=any(observed_rows(source,replace(ins,selected_ids=all_ids,locked_ids=(),group_by=g,sort_direction=s))
                      for g in ("none","color") for s in ("asc","desc"))
        report={"sample_id":old["sample_id"],"split":old["split"],"selected_ids":ins.selected_ids,"locked_ids":ins.locked_ids,
                "selected_count":count,"locked_count":len(ins.locked_ids),"source_origin":"legacy_randomized_scatter",
                "whole_source_structurally_organized":bool(organized),"selected_existing_rows":bool(observed_rows(source,ins)),
                "coverage_class":"single_object_reposition_or_reorient" if count==1 else "multi_object_local_arrangement",
                "supervises_selected_object_relations":count>=2,"human_quality_judgment":None}
        for label,target in targets.items():
            movements=object_movements(source,target)
            report[label]={"moved_count":sum(o["move_label"] for o in movements),"object_movements":movements,
                           "physical_moved_count":sum(o["translation_mm"]>1e-6 or o["rotation_deg"]>1e-6 for o in movements),
                           "instruction_satisfied":validate_layout(source,target,ins,"partial").valid,
                           **layout_metrics(source,target,ins)}
        reports.append(report)
    summary={"partial_samples":len(reports),"selected_count_histogram":dict(sorted(Counter(r["selected_count"] for r in reports).items())),
             "single_object_examples":sum(r["selected_count"]==1 for r in reports),
             "multi_object_examples":sum(r["selected_count"]>=2 for r in reports),
             "organized_source_examples":sum(r["whole_source_structurally_organized"] for r in reports),
             "known_organized_source_episodes":0,"multi_turn_episodes":0,"distribution_modified":False,
             "recommendations":["Retain single-object examples as grounding/keep supervision, not pairwise knolling evidence.",
                                "Keep the current 130-sample distribution unchanged.",
                                "For a separately approved Interactive extension, add 8–12 single-step examples starting organized, selecting 2–4 objects while locking others; preserve lineage splits.",
                                "Do not require multi-turn episodes for the first neural overfit pilot or claim they were validated here."]}
    return reports,summary


def counterfactual_audit(records):
    families=defaultdict(list)
    for r in records: families[r["scene_family_id"]].append(r)
    pairs=[]
    semantic_fields=("selected_ids","locked_ids","group_by","sort_direction","axis_alignment","gap_m","region","keep_all")
    for family,members in families.items():
        primary=next((r for r in members if r["role"]=="primary"),None)
        if primary is None: continue
        for other in members:
            if other["role"]!="counterfactual": continue
            a,b=primary,other
            source=scene_from_dict(a["source"])
            ia,ib=instruction_from_dict(a["instruction"]),instruction_from_dict(b["instruction"])
            ab=validate_layout(source,scene_from_dict(a["target"]),ib,a["mode"])
            ba=validate_layout(source,scene_from_dict(b["target"]),ia,b["mode"])
            pairs.append({"pair_id":family,"sample_ids":[a["sample_id"],b["sample_id"]],
                          "changed_semantic_fields":[k for k in semantic_fields if getattr(ia,k)!=getattr(ib,k)],
                          "a_target_under_b":ab.as_dict(),"b_target_under_a":ba.as_dict(),
                          "diagnostic_pair":not ab.valid and not ba.valid})
    return pairs

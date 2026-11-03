"""English v1 contract: deterministic, observable grounding, no translation API."""
from dataclasses import asdict,replace
from .schema import check_instruction,scene_from_dict,instruction_from_dict

LANGUAGE_POLICY="english_only_v1"
INSTRUCTION_CONTRACT="english_rows_v1"


def semantic_fields(ins):
    return {k:v for k,v in asdict(ins).items() if k not in ("text","language")}


def observable_subject(scene,ins):
    selected=set(ins.selected_ids)
    if selected=={o.object_id for o in scene.objects}:
        return "all objects"
    for color in ("red","green","blue"):
        color_ids={o.object_id for o in scene.objects if o.color==color}
        if selected and selected==color_ids:
            return f"only the {color} objects"
    by_x=sorted(scene.objects,key=lambda o:o.source_pose.x_m)
    if (selected=={by_x[0].object_id} and len(by_x)>1 and
        by_x[1].source_pose.x_m-by_x[0].source_pose.x_m>=.003):
        return "only the leftmost object"
    raise ValueError("selected IDs have no unambiguous supported English referent")


def english_instruction(scene,ins):
    """Change text/language only, deriving the referent from the SOURCE scene."""
    check_instruction(scene,ins)
    if ins.expected_status!="ok" or ins.preference_text:
        raise ValueError("English positive v1 contract requires ok status and no free-form preference")
    if ins.keep_all:
        return replace(ins,language="en",text="Keep every object at its current position and orientation.")
    subject=observable_subject(scene,ins)
    area={"full":"full workspace","left":"left region","right":"right region"}[ins.region]
    order={"asc":"smallest first","desc":"largest first"}[ins.sort_direction]
    mm=round(ins.gap_m*1000)
    text=f"Arrange {subject} in rows in the {area}; long axes {ins.axis_alignment}. "
    if ins.group_by=="color":
        text+="Use single-color rows, with each color's rows together. Within each color, "
    else:
        text+="Across the selected objects, "
    text+=f"order by bounding-rectangle area, {order}, left to right then top to bottom. "
    text+=f"Leave {mm} mm horizontal gaps between neighboring bounding rectangles in a row. "
    text+=(f"Use vertical gaps of {mm} mm between consecutive row bands within a color and {2*mm} mm between colors. "
           if ins.group_by=="color" else f"Use {mm} mm vertical gaps between consecutive row bands. ")
    text+="A row band spans the full height of that row's bounding rectangles."
    if ins.locked_ids:
        text+=" Keep all other objects fixed in position and orientation."
    return replace(ins,language="en",text=text)


def audit_english_records(records):
    """Validate the bounded English grammar, not an unreliable language guess."""
    errors=[]
    for r in records:
        sid=r["sample_id"]
        try:
            ins=instruction_from_dict(r["instruction"])
            if ins.language!="en" or not ins.text.isascii():
                errors.append(sid+":non_english_instruction")
            expected=english_instruction(scene_from_dict(r["source"]),ins)
            if ins.text!=expected.text:
                errors.append(sid+":english_text_semantics_mismatch")
        except (KeyError,ValueError) as exc:
            errors.append(sid+":english_contract:"+str(exc))
    return {"valid":not errors,"errors":errors,"samples":len(records),"language_policy":LANGUAGE_POLICY}

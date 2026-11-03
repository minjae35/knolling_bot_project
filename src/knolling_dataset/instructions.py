"""M3: a bounded bilingual grammar with explicit, observable referents.

Semantic IDs are supervision only: text never refers to hidden object IDs.
Area always means the major/minor bounding rectangle, not mesh surface area.
"""
import random
from dataclasses import replace
from .schema import Instruction, check_instruction

KO_COLOR = {"red":"빨간색", "green":"초록색", "blue":"파란색"}


def generate_instruction(scene, seed, mode="full", *, language=None, group_by=None,
                         sort_direction=None, axis_alignment=None, gap_m=None, region=None,
                         contract="legacy_v1_2"):
    if contract not in ("legacy_v1_2","row_edges_v1"):
        raise ValueError("unknown instruction contract")
    if mode not in ("full","partial","keep_all"):
        raise ValueError("unknown instruction mode")
    rng = random.Random(seed)
    language = language or rng.choice(("ko","en"))
    ids = tuple(o.object_id for o in scene.objects)
    if mode == "keep_all":
        text = "모든 물체의 현재 위치와 방향을 그대로 유지하세요." if language == "ko" else "Keep every object at its current position and orientation."
        return Instruction(text,language,(),ids,keep_all=True)
    selected,locked,ref = ids,(),"all"
    if mode == "partial":
        colors = sorted({o.color for o in scene.objects})
        if len(colors) > 1:
            ref = rng.choice(colors)
            selected = tuple(o.object_id for o in scene.objects if o.color == ref)
        else:
            by_x = sorted(scene.objects,key=lambda o:o.source_pose.x_m)
            if by_x[1].source_pose.x_m-by_x[0].source_pose.x_m < .003:
                raise ValueError("ambiguous leftmost referent: resample scene")
            selected,ref = (by_x[0].object_id,),"leftmost"
        locked = tuple(oid for oid in ids if oid not in selected)
    group_by = group_by or rng.choice(("none","color"))
    sort_direction = sort_direction or rng.choice(("asc","desc"))
    axis_alignment = axis_alignment or rng.choice(("horizontal","vertical"))
    gap_m = gap_m or rng.choice((.015,.020,.030))
    region = region or rng.choice(("full","left","right"))
    mm = round(gap_m*1000)
    if language == "ko":
        subject = "모든 물체를" if ref == "all" else "가장 왼쪽 물체만" if ref == "leftmost" else f"{KO_COLOR[ref]} 물체만"
        area = {"full":"작업대 전체 영역", "left":"왼쪽 영역", "right":"오른쪽 영역"}[region]
        axis = "수평" if axis_alignment == "horizontal" else "수직"
        order = "작은 것부터 큰 것" if sort_direction == "asc" else "큰 것부터 작은 것"
        text = f"{subject} {area}에 행으로 정돈하세요. 긴 축은 {axis}으로 맞추세요. "
        text += "색상별로 연속된 별도 행에 모으고, 각 색상 안에서 " if group_by == "color" else "전체를 "
        text += f"외접 직사각형 면적이 {order} 순으로 왼쪽에서 오른쪽, 위에서 아래로 배치하세요. "
        text += f"같은 행의 물체 가장자리 간격과 연속 행의 수직 가장자리 간격은 {mm}mm로 하세요."
        if group_by == "color":
            text += f" 서로 다른 색상 행 사이의 수직 간격은 {2*mm}mm로 하세요."
        if locked:
            text += " 나머지 물체의 위치와 방향은 그대로 유지하세요."
    else:
        subject = "all objects" if ref == "all" else "only the leftmost object" if ref == "leftmost" else f"only the {ref} objects"
        area = {"full":"full workspace", "left":"left region", "right":"right region"}[region]
        text = f"Arrange {subject} in rows in the {area}, with long axes {axis_alignment}. "
        text += "Use separate contiguous rows per color; within each color, " if group_by == "color" else "Across all objects, "
        text += f"sort by bounding rectangle area in {'ascending' if sort_direction == 'asc' else 'descending'} order, left to right then top to bottom. "
        text += f"Use {mm} mm horizontal edge gaps and vertical edge gaps between consecutive rows."
        if group_by == "color":
            text += f" Use {2*mm} mm vertical gaps between different colors."
        if locked:
            text += " Keep all remaining objects at their current positions and orientations."
    result = Instruction(text,language,selected,locked,group_by,sort_direction,axis_alignment,gap_m,region)
    check_instruction(scene,result)
    return clarify_instruction(result) if contract == "row_edges_v1" else result


def clarify_instruction(ins):
    """Clarify the bounded grammar, preserving every semantic field and numeral.

No RNG or referent selection is rerun. Refuse unrecognized templates instead of
silently replacing user language whose semantic correspondence is unknown.
"""
    if ins.keep_all or ins.expected_status != "ok":
        return ins
    mm=round(ins.gap_m*1000)
    if ins.language=="ko":
        old=f"같은 행의 물체 가장자리 간격과 연속 행의 수직 가장자리 간격은 {mm}mm로 하세요."
        new=(f"같은 행에서 이웃한 물체의 외접 직사각형 사이 수평 간격은 {mm}mm로 하세요. "
             "각 행의 모든 외접 직사각형을 포함하는 수평 띠를 '행 띠'라고 합니다. "
             f"연속한 행 띠 사이의 수직 간격은 {mm}mm로 하세요. "
             "이는 서로 다른 행의 모든 개별 물체 쌍 사이 간격을 뜻하지 않습니다.")
        old_group=f"서로 다른 색상 행 사이의 수직 간격은 {2*mm}mm로 하세요."
        new_group=f"단, 서로 다른 색상 행 띠 사이의 수직 간격은 {2*mm}mm로 하세요."
    else:
        old=f"Use {mm} mm horizontal edge gaps and vertical edge gaps between consecutive rows."
        new=(f"Use {mm} mm horizontal gaps between neighboring object envelopes in a row. "
             "A row band is the horizontal strip enclosing all object envelopes in that row. "
             f"Use {mm} mm vertical gaps between consecutive row bands, not between every cross-row object pair.")
        old_group=f"Use {2*mm} mm vertical gaps between different colors."
        new_group=f"For different-color row bands, use {2*mm} mm vertical gaps instead."
    if new in ins.text:
        return ins
    if ins.text.count(old)!=1 or (ins.group_by=="color" and ins.text.count(old_group)!=1):
        raise ValueError("unrecognized legacy row-gap template")
    text=ins.text.replace(old,new)
    if ins.group_by=="color":
        text=text.replace(old_group,new_group)
    return replace(ins,text=text)


def negative_instruction(scene, status):
    """Labeled interface fixtures, not synthetic geometric infeasibility claims."""
    if status == "unsupported":
        text,reason = "Stack every object into a tower.","stacking_out_of_scope"
    elif status == "ambiguous":
        text,reason = "Move that object to the left.","missing_referent"
    else:
        raise ValueError("only linguistically provable negative statuses supported")
    return Instruction(text,"en",tuple(o.object_id for o in scene.objects),(),expected_status=status,reason=reason)

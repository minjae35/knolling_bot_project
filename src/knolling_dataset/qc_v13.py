"""Complete, offline before/after review; all human ratings remain pending."""
import csv
import html
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw
from .qc import overlay,_font
from .schema import scene_from_dict,to_pixel
from .quality import object_movements


def movement_overlay(record,root):
    image=overlay(record,"source",root)
    draw=ImageDraw.Draw(image)
    targets={o.object_id:o for o in scene_from_dict(record["target"]).objects}
    for obj in scene_from_dict(record["source"]).objects:
        dst=targets[obj.object_id]
        start=to_pixel(obj.source_pose.x_m,obj.source_pose.y_m)
        end=to_pixel(dst.target_pose.x_m,dst.target_pose.y_m)
        if dst.locked or math.dist(start,end)<.01:
            continue
        draw.line([start,end],fill="#ffbd13",width=3)
        angle=math.atan2(end[1]-start[1],end[0]-start[0])
        head=[end]+[(end[0]-9*math.cos(angle+d),end[1]-9*math.sin(angle+d)) for d in (-.45,.45)]
        draw.polygon(head,fill="#ffbd13")
        draw.ellipse((end[0]-4,end[1]-4,end[0]+4,end[1]+4),outline="black",width=1)
    return image


def write_csv(path,rows,fields):
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_migration_qc(root,records,old_records,migrations,partials,partial_summary):
    root=Path(root); qc=root/"qc"; qc.mkdir(exist_ok=True)
    old_by={r["sample_id"]:r for r in old_records}
    migration_by={r["sample_id"]:r for r in migrations}
    partial_by={r["sample_id"]:r for r in partials}
    cards=[]; comparison=[]; movements=[]
    esc=lambda value:html.escape(str(value),quote=True)
    for r in records:
        sid=r["sample_id"]; old=old_by[sid]; row=migration_by[sid]
        legacy=dict(old,target_render_dir=row["legacy_target_render_dir"])
        views=[("source","Source",r,r["source_render_dir"]),
               ("legacy","v1.2 target",legacy,row["legacy_target_render_dir"]),
               ("target","v1.3 target",r,r["target_render_dir"])]
        figures=[]
        for role,title,record,directory in views:
            annotated=f"{sid}_{role}.png"
            overlay(record,"source" if role=="source" else "target",root).save(qc/annotated)
            raw=f"../{directory}/rgb.png"
            figures.append(f'<figure><img loading="lazy" class="toggle" src="{esc(annotated)}" data-annotated="{esc(annotated)}" data-raw="{esc(raw)}"><figcaption>{title} · <a href="{esc(raw)}">raw RGB</a></figcaption></figure>')
        movement_overlay(r,root).save(qc/f"{sid}_movement.png")
        object_rows=object_movements(scene_from_dict(r["source"]),scene_from_dict(r["target"]))
        for obj in object_rows:
            movements.append(dict(sample_id=sid,**obj))
        selected=len(r["instruction"]["selected_ids"]); locked=len(r["instruction"]["locked_ids"])
        # Keep-all has an empty selected set; repeat no-op locks all in its target.
        locked_target=sum(o["locked"] for o in r["target"]["objects"])
        moved=sum(o["move_label"] for o in object_rows)
        physical=sum(o["translation_mm"]>1e-6 or o["rotation_deg"]>1e-6 for o in object_rows)
        cmp={"sample_id":sid,"mode":r["mode"],"split":r["split"],"role":r["role"],"change_kind":row["change_kind"],
             "selected_count":selected,"locked_instruction_count":locked,"locked_target_count":locked_target,
             "physical_moved_count":physical,"move_label_count":moved,"instruction_satisfied":r["validation"]["valid"],
             "old_row_sizes":"/".join(str(len(x)) for x in row["old_rows"]),
             "new_row_sizes":"/".join(str(len(x)) for x in row["new_rows"]),
             "old_J_shape":row["old_metrics"]["J_shape"],"new_J_shape":row["new_metrics"]["J_shape"],
             "old_J_move_m2":row["old_metrics"]["J_move"],"new_J_move_m2":row["new_metrics"]["J_move"],
             "target_render_action":row["target_render_action"]}
        comparison.append(cmp)
        table="".join(f'<tr><td>{esc(o["object_id"])}</td><td>{o["locked"]}</td><td>{o["translation_mm"]:.3f}</td><td>{o["rotation_deg"]:.3f}</td><td>{o["move_label"]}</td></tr>' for o in object_rows)
        quality=partial_by.get(sid)
        note=(f'Partial coverage: {quality["coverage_class"]}. Organized source evidence: {quality["whole_source_structurally_organized"]}. '
              'This is a structural diagnostic, not a human quality rating.') if quality else ''
        cards.append(f'''<article id="{esc(sid)}" data-mode="{esc(r['mode'])}" data-split="{esc(r['split'])}" data-change="{esc(row['change_kind'])}">
<h2>{esc(sid)} · {esc(r['mode'])} · {esc(r['split'])} · {esc(r['role'])}</h2>
<p class="instruction" lang="{esc(r['instruction']['language'])}">{esc(r['instruction']['text'])}</p>
<details><summary>Original instruction (same semantic request)</summary><p>{esc(old['instruction']['text'])}</p></details>
<p>{esc(row['change_kind'])} · selected {selected} · locked instruction/target {locked}/{locked_target} · moved physically {physical} · move-label positive {moved}. Automated instruction check: {r['validation']['valid']}.</p>
<div class="views">{''.join(figures)}</div>
<details open><summary>Source → v1.3 movements and identity</summary><div class="movement">
<img loading="lazy" src="{esc(sid)}_movement.png" alt="Yellow source-to-target translation arrows; locked objects have no arrows">
<table><thead><tr><th>ID</th><th>Locked</th><th>Δ mm</th><th>Δ ° (mod π)</th><th>Move label</th></tr></thead><tbody>{table}</tbody></table></div></details>
<p>Rows: {esc(cmp['old_row_sizes'])} → {esc(cmp['new_row_sizes'])}; J_shape: {esc(cmp['old_J_shape'])} → {esc(cmp['new_J_shape'])}; J_move (m²): {cmp['old_J_move_m2']:.8f} → {cmp['new_J_move_m2']:.8f}.</p>
<p>{esc(note)}</p><p><strong>Human review: pending.</strong> Rate instruction satisfaction, tidiness, partial movement naturalness, visual legibility and overall quality (1–5); mark acceptable separately. Keep-all requires preservation, not tidiness improvement.</p>
<p><a href="../samples/{esc(sid)}/annotation.json">Annotation and automated checks</a> · <a href="../parent_annotations/{esc(sid)}.json">Original annotation</a></p></article>''')
    write_csv(qc/"target_comparison.csv",comparison,list(comparison[0]))
    write_csv(qc/"object_movements.csv",movements,list(movements[0]))
    flat=[]
    for p in partials:
        flat.append({"sample_id":p["sample_id"],"split":p["split"],"selected_count":p["selected_count"],"locked_count":p["locked_count"],
                     "selected_ids":"/".join(p["selected_ids"]),"locked_ids":"/".join(p["locked_ids"]),
                     "legacy_moved":p["legacy"]["physical_moved_count"],"v1_3_moved":p["v1_3"]["physical_moved_count"],
                     "legacy_instruction_satisfied":p["legacy"]["instruction_satisfied"],"v1_3_instruction_satisfied":p["v1_3"]["instruction_satisfied"],
                     "coverage_class":p["coverage_class"],"organized_source":p["whole_source_structurally_organized"],
                     "old_J_move_m2":p["legacy"]["J_move"],"new_J_move_m2":p["v1_3"]["J_move"]})
    if flat: write_csv(qc/"partial_quality.csv",flat,list(flat[0]))
    report=["# Partial refinement audit", "", "Automated geometry and coverage diagnostics; human visual judgments remain pending.",
            "", "The move label uses >3 mm or >3°; physical movement counts use >1e-6 mm or >1e-6° numerical tolerance.",
            "", "| Sample | Selected | Locked | Moved v1.2 → v1.3 | Instruction v1.2 / v1.3 | Coverage |", "|---|---:|---:|---:|---|---|"]
    for p in flat:
        report.append(f"| {p['sample_id']} | {p['selected_count']} | {p['locked_count']} | {p['legacy_moved']} → {p['v1_3_moved']} | {p['legacy_instruction_satisfied']} / {p['v1_3_instruction_satisfied']} | {p['coverage_class']} |")
    report += ["",f"Single-object: {partial_summary['single_object_examples']}; multi-object: {partial_summary['multi_object_examples']}; structurally organized sources: {partial_summary['organized_source_examples']}; known organized-source episodes: 0.",
               "", "Multi-object local arrangement is represented, but this does not establish interactive refinement of previously organized scenes. Locked scattered objects are required to remain scattered. A single selected object cannot supervise pairwise ordering or spacing.",
               "", "## Minimal recommendation (not applied)", ""]+["- "+r for r in partial_summary["recommendations"]]
    (qc/"partial_quality.md").write_text("\n".join(report)+"\n")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>v1.3 complete migration review</title>
<style>body{font:16px system-ui,sans-serif;background:#edf1f5;color:#132333;max-width:1800px;margin:auto;padding:24px}article{background:white;border:1px solid #bdcad5;border-radius:10px;margin:24px 0;padding:20px}h2{font-size:19px}.instruction{font-size:18px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;display:block}figcaption{margin:8px 0}.movement{display:flex;gap:20px;align-items:start}.movement img{max-width:576px;width:50%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccd6de;text-align:left}select{padding:6px;font:inherit}a{color:#14578c}details{margin:12px 0}nav{position:sticky;top:0;background:#edf1f5;padding:12px;z-index:1}@media(max-width:800px){.views{grid-template-columns:1fr}.movement{display:block}.movement img{width:100%}}</style>
<h1>Knolling v1.3 · Complete manual review</h1>
<p>COUNT samples from the original fixed source scenes. <strong>Human review is pending for every sample.</strong> Geometry validity and lower J_shape do not establish aesthetic quality or useful training coverage.</p>
<p>Yellow arrows show translation; the table includes π-periodic rotations. [L] marks locked objects. Physical movement and the >3 mm / >3° move label are reported separately. Row spacing means separation between the upper/lower envelope edges of whole row bands, not every cross-row object pair.</p>
<p><a href="target_comparison.csv">All target comparisons</a> · <a href="object_movements.csv">All object movements</a> · <a href="partial_quality.md">30 Partial cases and recommendations</a> · <a href="partial_quality.csv">Partial CSV</a> · <a href="../human_review.jsonl">Pending human review records</a> · <a href="../audit.json">Automated audit</a> · <a href="summary.json">Image QC metrics</a></p>
<p>Record reviewer, review time, individual 1–5 ratings, acceptable and notes in human_review.jsonl after inspection. No samples are exported for training in this phase. Primary quality gate: at least 90/100 with applicable overall quality ≥4/5; inspect all 30 additional samples too. Any ≤2/5 problem needs a recorded resolution. Keep-all does not require a tidiness improvement.</p>
<nav><label><input id="raw" type="checkbox"> Unannotated RGB</label> <label>Mode <select id="mode"><option value="">All</option><option>full</option><option>partial</option><option>no_op</option></select></label> <label>Split <select id="split"><option value="">All</option><option>train</option><option>validation</option><option>development</option><option>test</option></select></label> <label>Change <select id="change"><option value="">All</option><option>target_changed</option><option>text_only</option><option>unchanged</option></select></label> <span id="shown"></span></nav>
CARDS
<script>const filters=['mode','split','change'].map(id=>document.getElementById(id));function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=filters.some(f=>f.value&&a.dataset[f.id]!==f.value);if(!a.hidden)n++});document.getElementById('shown').textContent=n+' samples shown'}filters.forEach(f=>f.onchange=filter);document.getElementById('raw').onchange=e=>document.querySelectorAll('img.toggle').forEach(i=>i.src=e.target.checked?i.dataset.raw:i.dataset.annotated);filter();</script></html>'''
    (qc/"index.html").write_text(page.replace("COUNT",str(len(records))).replace("CARDS","".join(cards)))
    # A navigation aid only; this does not substitute for all-sample review.
    representatives=[]
    for mode in ("full","partial","no_op"):
        group=sorted((r for r in records if r["role"]=="primary" and r["mode"]==mode),key=lambda r:len(r["instruction"]["selected_ids"]))
        if group: representatives.extend([group[0],group[len(group)//2],group[-1]])
    sheet=Image.new("RGB",(1152,292*len(representatives)),"white"); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(representatives):
        sid=r["sample_id"]; draw.text((8,i*292+4),f"{sid} | {r['mode']} | source / legacy / v1.3",fill="black",font=_font(16))
        for j,role in enumerate(("source","legacy","target")):
            with Image.open(qc/f"{sid}_{role}.png") as im: sheet.paste(im.resize((384,256)),(j*384,i*292+30))
    sheet.save(qc/"migration_contact_sheet.png")
    return {"review_cards":len(cards),"movement_rows":len(movements),"human_review_complete":False}

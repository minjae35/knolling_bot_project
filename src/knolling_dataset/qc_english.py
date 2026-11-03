"""English-only manual QC; historical bilingual labels are not model inputs."""
from collections import Counter
import html
import json
from pathlib import Path
from .qc import write_qc
from .qc_v13 import movement_overlay,write_csv
from .quality import object_movements
from .schema import scene_from_dict


def write_english_qc(root,records,stats):
    root=Path(root); qc=root/"qc"
    summary=write_qc(root,records,stats)
    summary.update(language_policy="english_only_v1",language_counts=dict(Counter(r["instruction"]["language"] for r in records)))
    (qc/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    esc=lambda s:html.escape(str(s),quote=True)
    cards=[]; instructions=[]; movements=[]
    for r in records:
        sid=r["sample_id"]; ins=r["instruction"]
        instructions.append({"sample_id":sid,"split":r["split"],"mode":r["mode"],"language":ins["language"],"instruction":ins["text"]})
        items=object_movements(scene_from_dict(r["source"]),scene_from_dict(r["target"]))
        movements.extend(dict(sample_id=sid,**o) for o in items)
        movement_overlay(r,root).save(qc/f"{sid}_movement.png")
        figures=[]
        for role,title in (("source","Source"),("target","Target")):
            raw=f"../{r[role+'_render_dir']}/rgb.png"; annotated=f"{sid}_{role}.png"
            figures.append(f'<figure><img class="toggle" loading="lazy" src="{esc(annotated)}" data-raw="{esc(raw)}" data-annotated="{esc(annotated)}"><figcaption>{title} · <a href="{esc(raw)}">Raw RGB</a></figcaption></figure>')
        table=''.join(f'<tr><td>{esc(o["object_id"])}</td><td>{o["locked"]}</td><td>{o["translation_mm"]:.3f}</td><td>{o["rotation_deg"]:.3f}</td><td>{o["move_label"]}</td></tr>' for o in items)
        cards.append(f'''<article id="{esc(sid)}" data-mode="{esc(r['mode'])}" data-split="{esc(r['split'])}">
<h2>{esc(sid)} · {esc(r['mode'])} · {esc(r['split'])} · {esc(r['role'])}</h2>
<p class="instruction" lang="en">{esc(ins['text'])}</p>
<p>Selected: {esc(', '.join(ins['selected_ids']) or 'none')}; instruction locks: {esc(', '.join(ins['locked_ids']) or 'none')}. Automated geometry/instruction check: {r['validation']['valid']}.</p>
<div class="views">{''.join(figures)}</div>
<details><summary>Object movements (source to target, not changes made by language migration)</summary><div class="views"><img loading="lazy" src="{esc(sid)}_movement.png" alt="Source-to-target translation arrows"><table><tr><th>ID</th><th>Locked</th><th>Translation mm</th><th>Rotation degrees</th><th>Move label</th></tr>{table}</table></div></details>
<p>Human review: pending. Check English clarity, instruction satisfaction, selected-object tidiness, preservation of locked objects and movement naturalness. Keep-all requires preservation, not tidiness improvement.</p>
<a href="../samples/{esc(sid)}/annotation.json">English annotation, identity and geometry checks</a></article>''')
    write_csv(qc/"english_instructions.csv",instructions,list(instructions[0]))
    write_csv(qc/"object_movements.csv",movements,list(movements[0]))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>English-only Knolling v1 QC</title>
<style>body{font:16px system-ui,sans-serif;background:#eef2f5;color:#132333;max-width:1400px;margin:auto;padding:24px}article{background:white;padding:20px;margin:24px 0;border:1px solid #c5d0da;border-radius:10px}h2{font-size:18px}.instruction{font-size:18px}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;display:block}figcaption{margin:8px 0}table{border-collapse:collapse;height:fit-content}td,th{border:1px solid #ccd6de;padding:6px;text-align:left}nav{position:sticky;top:0;background:#eef2f5;padding:12px;z-index:1}select{font:inherit;padding:6px}a{color:#14578c}@media(max-width:750px){.views{grid-template-columns:1fr}}</style>
<h1>Knolling v1 · English-only manual review</h1>
<p>COUNT active samples. Every instruction is English. Human visual and language review remains pending; automated semantic equality does not supply human approval.</p>
<p>Gap values refer to bounding rectangles. A row band spans the full vertical extent of all bounding rectangles in a row; its gap is vertical edge-to-edge distance. Partial constraints apply to selected objects, while locked context stays fixed. Yellow arrows show translation; rotations in the table use 180-degree periodicity.</p>
<p><a href="english_instructions.csv">All English instructions</a> · <a href="object_movements.csv">All object movements</a> · <a href="summary.json">QC metrics</a> · <a href="../audit.json">Automatic audit</a> · <a href="../human_review.jsonl">Pending review records</a></p>
<p>Training inputs come only from the active manifest and its split indexes. Historical parent annotations are archival evidence and must never be globbed into training. Record reviewer/time, instruction satisfaction, tidiness, Partial movement naturalness, legibility, overall score (1–5), acceptable and notes. For the 100-scene pilot, review all 130 samples and apply the existing quality gate; no training export is approved here.</p>
<nav><label><input type="checkbox" id="raw"> Unannotated RGB</label> <label>Mode <select id="mode"><option value="">All</option><option>full</option><option>partial</option><option>no_op</option></select></label> <label>Split <select id="split"><option value="">All</option><option>train</option><option>validation</option><option>development</option><option>test</option></select></label> <span id="shown"></span></nav>
CARDS
<script>const mode=document.getElementById('mode'),split=document.getElementById('split');function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!!((mode.value&&a.dataset.mode!==mode.value)||(split.value&&a.dataset.split!==split.value));if(!a.hidden)n++});document.getElementById('shown').textContent=n+' samples shown'}mode.onchange=split.onchange=filter;document.getElementById('raw').onchange=e=>document.querySelectorAll('img.toggle').forEach(i=>i.src=e.target.checked?i.dataset.raw:i.dataset.annotated);filter();</script></html>'''
    (qc/"index.html").write_text(page.replace("COUNT",str(len(records))).replace("CARDS",''.join(cards)))
    return summary

"""Local, offline visual QC. Passing automation is not human sign-off."""
from collections import Counter
import html
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .schema import scene_from_dict,to_pixel
from .validator import polygon


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def overlay(record,role,root):
    scene = scene_from_dict(record["source"] if role == "source" else record["target"])
    image = Image.open(root/record[role+"_render_dir"]/"rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    locked = {o["object_id"] for o in record["target"]["objects"] if o["locked"]}
    for obj in scene.objects:
        pose = obj.source_pose if role == "source" else obj.target_pose
        points = [to_pixel(x,y) for x,y in polygon(obj,pose).exterior.coords]
        draw.line(points,fill=(10,220,230),width=1)
        x,y = to_pixel(pose.x_m,pose.y_m)
        end = to_pixel(pose.x_m+obj.asset.a_m*.30*math.cos(pose.theta_rad),
                       pose.y_m+obj.asset.a_m*.30*math.sin(pose.theta_rad))
        draw.line([(x,y),end],fill="black",width=2)
        label = obj.object_id+(" [L]" if obj.object_id in locked else "")
        draw.text((x,y-9),label,font=_font(13),anchor="mm",fill="white",stroke_width=2,stroke_fill="black")
    return image


def write_qc(root,records,stats):
    root = Path(root)
    qc = root/"qc"
    qc.mkdir(exist_ok=True)
    primary = [r for r in records if r["role"] == "primary"]
    assets = [o["asset"] for r in primary for o in r["source"]["objects"]]
    checks = [v for r in records for role in ("source","target") for v in r["render_validation"][role]["objects"]]
    summary = {"all_sample_geometry_valid":all(r["validation"]["valid"] for r in records),
               "all_render_checks_valid":all(r["render_validation"][p]["valid"] for r in records for p in ("source","target")),
               "min_mask_iou":min(v["mask_iou"] for v in checks),
               "max_centroid_error_px":max(v["centroid_error_px"] for v in checks),
               "min_rgb_color_accuracy":min(v["rgb_color_accuracy"] for v in checks),
               "shape_counts":dict(Counter(a["shape"] for a in assets)),
               "color_counts":dict(Counter(o["color"] for r in primary for o in r["source"]["objects"])),
               "unique_asset_families":len({a["family_id"] for a in assets}),
               "dimension_ranges_m":{key:[min(a[key] for a in assets),max(a[key] for a in assets)] for key in ("a_m","b_m","h_m")},
               "instruction_axes":dict(Counter(r["instruction"]["axis_alignment"] for r in records if not r["instruction"]["keep_all"])),
               "instruction_groups":dict(Counter(r["instruction"]["group_by"] for r in records if not r["instruction"]["keep_all"])),
               "instruction_regions":dict(Counter(r["instruction"]["region"] for r in records if not r["instruction"]["keep_all"])),
               "human_review_status":"pending_user_review","automated_report_is_human_approval":False}
    (qc/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    cards = []
    for record in records:
        sid = record["sample_id"]
        for role in ("source","target"):
            overlay(record,role,root).save(qc/f"{sid}_{role}.png")
        title = f"{sid} | {record['mode']} | {record['split']} | {len(record['source']['objects'])} objects"
        cards.append(f'''<article data-mode="{record['mode']}" data-split="{record['split']}">
<h2>{html.escape(title)}</h2><p lang="{record['instruction']['language']}">{html.escape(record['instruction']['text'])}</p>
<div class="pair"><figure><img loading="lazy" src="{sid}_source.png"><figcaption>Input · Source IDs</figcaption></figure>
<figure><img loading="lazy" src="{sid}_target.png"><figcaption>Target · Same IDs · [L] = locked</figcaption></figure></div>
<p><a href="../samples/{sid}/annotation.json">Complete annotation / geometry / render checks</a> ·
<a href="../{record['source_render_dir']}/instance_mask.png">Source uint16 mask</a> ·
<a href="../{record['target_render_dir']}/instance_mask.png">Target uint16 mask</a></p></article>''')
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Knolling pilot QC</title>
<style>body{{font:16px system-ui,sans-serif;background:#edf1f5;color:#132333;max-width:1240px;margin:auto;padding:24px}}
article{{background:white;border:1px solid #cad5df;border-radius:10px;margin:24px 0;padding:20px}}h2{{font-size:17px}}
.pair{{display:flex;gap:16px}}figure{{margin:0;flex:1}}img{{width:100%;display:block}}figcaption{{margin-top:8px}}
pre{{white-space:pre-wrap}}select{{font:inherit;padding:6px}}a{{color:#14578c}}@media(max-width:700px){{.pair{{display:block}}}}</style>
<h1>Knolling Phase 1 · Pilot quality control</h1>
<p>{stats['accepted_scenes']} distinct scene families, {len(records)} samples. Same-image counterfactual instructions share their family and split.</p>
<p><strong>Human review is pending.</strong> Automated checks verify geometry, identity, instruction predicates,
and RGB/mask projections. Please inspect visual legibility, natural-language intent and useful layout diversity.
Masks contain integer IDs 1–8 on background 0 and may appear black in ordinary viewers.</p>
<p><a href="contact_sheet.png">Representative contact sheet</a> · <a href="summary.json">QC metrics</a> ·
<a href="../generation.json">Attempts, rejections and diversity</a> · <a href="../audit.json">Independent dataset audit</a></p>
<label>Mode <select id="mode"><option value="">All</option><option>full</option><option>partial</option><option>no_op</option></select></label>
<label>Split <select id="split"><option value="">All</option><option>train</option><option>validation</option><option>development</option><option>test</option></select></label>
{''.join(cards)}<script>function filter(){{document.querySelectorAll('article').forEach(a=>{{a.hidden=(mode.value&&a.dataset.mode!==mode.value)||(split.value&&a.dataset.split!==split.value)}})}}
const mode=document.getElementById('mode'),split=document.getElementById('split');mode.onchange=split.onchange=filter;</script></html>'''
    (qc/"index.html").write_text(report)
    representatives = []
    for mode in ("full","partial","no_op"):
        group = sorted((r for r in primary if r["mode"] == mode),key=lambda r:len(r["source"]["objects"]))
        representatives.extend([group[0],group[len(group)//2],group[-1]])
    sheet = Image.new("RGB",(768,len(representatives)*292),"white")
    draw = ImageDraw.Draw(sheet)
    for i,r in enumerate(representatives):
        y = i*292
        draw.text((8,y+4),f"{r['sample_id']} | {r['mode']} | {len(r['source']['objects'])} objects | {r['split']}",fill="black",font=_font(16))
        for j,role in enumerate(("source","target")):
            tile = Image.open(qc/f"{r['sample_id']}_{role}.png").resize((384,256))
            sheet.paste(tile,(j*384,y+30))
    sheet.save(qc/"contact_sheet.png")
    return summary

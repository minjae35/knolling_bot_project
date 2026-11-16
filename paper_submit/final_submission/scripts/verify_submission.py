"""Check manuscript completeness and frozen evidence, without model inference."""
from pathlib import Path
import hashlib
import json
import math
import re
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT / 'knolling_bot_v2_paper.pdf'
doc = pymupdf.open(pdf)
pages = [p.get_text() for p in doc]
assert len(pages) == 12, f'Expected 12 pages, found {len(pages)}'
assert 'Conclusion' in pages[4] or 'CONCLUSION' in pages[4]
assert 'References' in pages[5] or 'REFERENCES' in pages[5]
assert 'IMPLEMENTATION DETAILS' in pages[6] or 'Implementation Details' in pages[6]
assert all('References' not in x and 'REFERENCES' not in x for x in pages[:5])
text = '\n'.join(pages)
for required in ['Minjae Bang', 'Columbia University', '540/640', '489/640', '84.4', '76.4']:
    assert required in text, required
assert '??' not in text
assert 'Published as a conference paper' not in text
assert 'Under review as a conference paper' not in text
for p in doc:
    assert not p.get_text(clip=pymupdf.Rect(0,0,p.rect.width,65)).strip(), 'Unexpected running header'
    for word in p.get_text('words'):
        x0,y0,x1,y1=word[:4]
        assert x0 >= 0 and y0 >= 0 and x1 <= p.rect.width+.1 and y1 <= p.rect.height+.1

tex = (ROOT/'main.tex').read_text() + (ROOT/'appendix.tex').read_text()
bib = (ROOT/'references.bib').read_text()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cited = set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in group.split(','))
assert cited <= keys
assert len(cited) == 9
assert tex.count('\\begin{figure}') == 7
assert tex.count('\\begin{table}') == 6
stats = json.loads((ROOT/'audit/verified_results.json').read_text())
assert stats['v2']['overall']['success']==540
assert stats['v1']['overall']['success']==489
assert stats['paired']['gained']==51 and stats['paired']['lost']==0
assert stats['v2']['unique_spacing_failure_scenes']==71
for name,sha in stats['sources'].items():
    assert hashlib.sha256((ROOT/'audit/evidence'/name).read_bytes()).hexdigest()==sha
for entry in json.loads((ROOT/'template/upstream.json').read_text())['files']:
    assert hashlib.sha256((ROOT/'template'/entry['file']).read_bytes()).hexdigest()==entry['sha256']
manifest=ROOT/'audit/evidence_manifest.json'
if manifest.exists():
    for name,sha in json.loads(manifest.read_text())['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, name
measurements=json.loads((ROOT/'audit/visual_measurements.json').read_text())
for name,sha in measurements['provenance']['inputs'].items():
    assert hashlib.sha256((ROOT/'audit/evidence'/name).read_bytes()).hexdigest()==sha
assert 17.65 < measurements['full']['vertical_gaps'][1]['actual_mm'] < 17.67
assert 9.895 < measurements['partial']['minimum_clearance_mm'] < 9.900
figure_names=['architecture_main','qualitative_full_annotated','qualitative_modes','holdout_diagnostics','selector_training','motion_correspondence','failure_explanation']
for name in figure_names:
    for ext in ['pdf','svg','png']:
        assert (ROOT/'figures'/f'{name}.{ext}').is_file()
    figure=pymupdf.open(ROOT/'figures'/f'{name}.pdf')
    assert abs(figure[0].rect.width-396)<.1
    for word in figure[0].get_text('words'):
        assert word[0]>=-.1 and word[1]>=-.1 and word[2]<=396.1 and word[3]<=figure[0].rect.height+.1, (name,word)
motion=json.loads((ROOT/'audit/motion_annotations.json').read_text())
failure=json.loads((ROOT/'audit/failure_annotations.json').read_text())
for hashes in [motion['sources'],failure['provenance']['input_sha256']]:
    for path,sha in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha, path
saved=json.loads((ROOT/'audit/evidence/full8b_choice.json').read_text())
by_id={x['object_id']:x for x in saved['scene']['objects']}
assert len(motion['objects'])==8
assert motion['saved_blender_camera_basis_max_error']<1e-7
for item in motion['objects']:
    obj=by_id[item['object_id']]
    a,b=obj['source_pose'],obj['target_pose']
    assert item['source']==a and item['target']==b
    distance=1000*math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m'])
    angle=math.degrees((b['theta_rad']-a['theta_rad']+math.pi/2)%math.pi-math.pi/2)
    assert abs(item['translation_mm']-distance)<1e-9
    assert abs(item['delta_theta_deg']-angle)<1e-9
assert abs(failure['full']['excess_over_upper_limit_mm']-(measurements['full']['vertical_gaps'][1]['actual_mm']-17))<1e-9
assert abs(failure['partial']['deficit_um']-1000*(9.9-measurements['partial']['minimum_clearance_mm']))<1e-8
render_manifest=json.loads((ROOT/'audit/blender_provenance.json').read_text())
pose_audit=json.loads((ROOT/'audit/blender_pose_checks.json').read_text())
assert pose_audit['status']=='PASS' and pose_audit['maximum_pose_error']<=1e-7
assert len(render_manifest['renders'])==len(pose_audit['renders'])==8
for name,sha in render_manifest['input_sha256'].items():
    assert hashlib.sha256((ROOT/'audit/evidence'/name).read_bytes()).hexdigest()==sha
for name,record in render_manifest['renders'].items():
    assert hashlib.sha256((ROOT/'renders'/record['image']).read_bytes()).hexdigest()==record['sha256']
    assert hashlib.sha256((ROOT/'renders'/(name+'.blend')).read_bytes()).hexdigest()==pose_audit['renders'][name]['blend_sha256']
    assert record['resolution']==[1920,1280]
conditions=['camera_location','camera_euler','ortho_scale_m','resolution','samples','world_strength','view_transform','look','lights','object_materials']
for group in [['full_before','full_after'],['smoke_before','smoke_full','smoke_partial','smoke_no_op'],['smoke_full_overhead','smoke_partial_overhead']]:
    base=pose_audit['renders'][group[0]]
    for name in group[1:]:
        for key in conditions:assert pose_audit['renders'][name][key]==base[key],(name,key)
teaser=json.loads((ROOT/'audit/teaser6_provenance.json').read_text())
assert teaser['status']=='PASS' and teaser['maximum_absolute_pose_or_vertex_error']<1e-7
assert hashlib.sha256((ROOT/teaser['source_path']).read_bytes()).hexdigest()==teaser['source_sha256']
for name,record in teaser['renders'].items():
    assert hashlib.sha256((ROOT/'renders'/(name+'.png')).read_bytes()).hexdigest()==record['image_sha256']
    assert hashlib.sha256((ROOT/'renders'/(name+'.blend')).read_bytes()).hexdigest()==record['blend_sha256']
    assert record['resolution']==[1920,1280]
architecture=json.loads((ROOT/'audit/final_architecture.json').read_text())
assert architecture['width_pt']==396 and architecture['minimum_label_pt']>=7
for i,page in enumerate(doc):
    assert len(page.get_text().strip())>800, ('Unexpected sparse/overflow page',i+1)
log=ROOT/'build/main.log'
if log.exists():
    contents=log.read_text()
    assert 'Overfull' not in contents
    assert 'undefined' not in contents.lower()
    assert 'LaTeX Warning' not in contents
report={'status':'PASS','total_pages':12,'main_pages':[1,5],'references_pages':[6,6],
        'appendix_pages':[7,12],'figures':7,'tables':6,'cited_references':9,
        'checks':['page boundaries','author and primary counts','no unresolved references',
                  'no conference acceptance claim','PDF text within page bounds',
                  'frozen score hashes','official template hashes','evidence manifest',
                  'no LaTeX warnings or overfull boxes','qualitative measurement input hashes',
                  'geometric annotations and all vector/preview exports',
                  'eight Blender render/source hashes','saved Blender mesh and pose audit',
                  'matched before/after camera, light and material settings',
                  'no running headers','all figure text within bounds',
                  'eight source-to-target distances and signed rotations',
                  'failure margins and motion/failure source hashes',
                  'six-object teaser Blender/image/source hashes',
                  'architecture minimum label size and width',
                  'no sparse overflow pages'],
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(ROOT/'audit/submission_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

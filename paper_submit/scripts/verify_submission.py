"""Check manuscript completeness and frozen evidence, without model inference."""
from pathlib import Path
import hashlib
import json
import math
import re
import pymupdf
from preserve_figure6 import verify as verify_figure6
from verify_reference_analysis import verify as verify_reference_analysis

ROOT = Path(__file__).resolve().parents[1]
figure6_check = verify_figure6()
reference_analysis_check = verify_reference_analysis()
pdf = ROOT / 'knolling_bot_v2_paper.pdf'
doc = pymupdf.open(pdf)
pages = [p.get_text() for p in doc]
assert len(pages) == 14, f'Expected 14 pages, found {len(pages)}'
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
assert tex.count('\\begin{figure}') == 8
assert tex.count('\\begin{table}') == 7
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
figure_names=['architecture_main','qualitative_full_annotated','qualitative_modes','holdout_diagnostics','selector_training','motion_correspondence','failure_explanation','layout_constraints']
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
motion_plot=json.loads((ROOT/'audit/motion_visualization.json').read_text())
assert sorted(r['object_id'] for r in motion_plot['object_rows'])==sorted(by_id)
for path,sha in motion_plot['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
for row in motion_plot['object_rows']:
    obj=by_id[row['object_id']];a,b=obj['source_pose'],obj['target_pose']
    assert row['source_pose']==a and row['predicted_target_pose']==b
    dx,dy=1000*(b['x_m']-a['x_m']),1000*(b['y_m']-a['y_m'])
    distance=math.hypot(dx,dy)
    angle=math.degrees((b['theta_rad']-a['theta_rad']+math.pi/2)%math.pi-math.pi/2)
    for actual,expected in [(row['dx_mm'],dx),(row['dy_mm'],dy),
        (row['translation_mm'],distance),(row['delta_theta_deg'],angle)]:
        assert abs(actual-expected)<1e-9
    assert row['display_distance_mm']==f'{distance:.1f}'
    assert row['display_delta_theta_deg']==f'{angle:+.1f}'
    start,end=row['direction_arrow_start_bp'],row['direction_arrow_end_bp']
    assert abs(end[0]-start[0]-8*dx/distance)<1e-9
    assert abs(end[1]-start[1]+8*dy/distance)<1e-9
    for axis,value,point in [('distance',distance,row['distance_point_bp']),
        ('rotation',angle,row['rotation_point_bp'])]:
        limits=motion_plot[axis+'_axis'];lo,hi=limits['range_bp']
        v0,v1=limits['range_mm' if axis=='distance' else 'range_deg']
        assert abs(point[0]-(lo+(value-v0)/(v1-v0)*(hi-lo)))<1e-9
        assert abs(point[1]-row['row_y_bp'])<1e-9
for ext,sha in motion_plot['exports_sha256'].items():
    assert hashlib.sha256((ROOT/'figures'/f'motion_correspondence.{ext}').read_bytes()).hexdigest()==sha
assert 'Figure 4:' in pages[4] and 'Figure 8:' in pages[12]
assert 'Table 7:' in pages[12]
layout=json.loads((ROOT/'audit/layout_constraints.json').read_text())
assert len(layout['table7_rows'])==8
for path,sha in layout['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
appendix=(ROOT/'appendix.tex').read_text()
for row in layout['table7_rows']:
    obj=by_id[row['object_id']];a,b=obj['source_pose'],obj['target_pose']
    assert row['source_m_rad']==a and row['predicted_m_rad']==b
    distance=1000*math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m'])
    angle=math.degrees((b['theta_rad']-a['theta_rad']+math.pi/2)%math.pi-math.pi/2)
    values=[1000*a['x_m'],1000*a['y_m'],math.degrees(a['theta_rad']),
            1000*b['x_m'],1000*b['y_m'],math.degrees(b['theta_rad']),distance,angle]
    expected=[f'{v:.2f}' for v in values]
    assert row['display_values_2dp']==expected
    assert ' & '.join([str(row['id'])]+expected) in appendix
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
details=json.loads((ROOT/'audit/detail_render_provenance.json').read_text())
assert set(details['renders'])=={'detail_full_workspace','detail_row_gap','detail_clearance'}
for name,record in details['renders'].items():
    assert record['noncamera_scene_unchanged'] and record['saved_file_reopened_and_checked']
    assert hashlib.sha256((ROOT/record['source_blend']).read_bytes()).hexdigest()==record['source_sha256']
    assert hashlib.sha256((ROOT/'renders'/record['image']).read_bytes()).hexdigest()==record['image_sha256']
    assert hashlib.sha256((ROOT/'renders'/(name+'.blend')).read_bytes()).hexdigest()==record['blend_sha256']
    image=pymupdf.Pixmap(ROOT/'renders'/record['image'])
    assert [image.width,image.height]==record['resolution']
    x0,x1,y0,y1=record['world_extent_m']
    assert abs((x1-x0)/(y1-y0)-image.width/image.height)<1e-10
    assert record['camera_euler']==[0,0,0]
    assert abs(record['ortho_scale_m']-(x1-x0))<1e-7
for i,page in enumerate(doc):
    assert len(page.get_text().strip())>800, ('Unexpected sparse/overflow page',i+1)
log=ROOT/'build/main.log'
if log.exists():
    contents=log.read_text()
    assert 'Overfull' not in contents
    assert 'undefined' not in contents.lower()
    assert 'LaTeX Warning' not in contents
report={'status':'PASS','total_pages':14,'main_pages':[1,5],'references_pages':[6,6],
        'appendix_pages':[7,14],'figures':8,'tables':7,'cited_references':9,
        'checks':['page boundaries','author and primary counts','no unresolved references',
                  'no conference acceptance claim','PDF text within page bounds',
                  'frozen score hashes','official template hashes','evidence manifest',
                  'no LaTeX warnings or overfull boxes','qualitative measurement input hashes',
                  'geometric annotations and all vector/preview exports',
                  'eight Blender render/source hashes','saved Blender mesh and pose audit',
                  'matched before/after camera, light and material settings',
                  'no running headers','all figure text within bounds',
                  'eight source-to-target distances and signed rotations',
                  'eight plotted world-direction arrows, displacement/rotation axes and rounded labels',
                  'Table 7 full source/predicted coordinates, unsnapped angles and differences',
                  'five main pages; retained Figure 5–7 numbers; appended Figure 8 and pose Table 7',
                  'failure margins and motion/failure source hashes',
                  'six-object teaser Blender/image/source hashes',
                  'three dedicated overhead render hashes, extents, resolution, and saved-scene preservation',
                  'architecture minimum label size and width',
                  'no sparse overflow pages',
                  'user-approved Figure 6 exports, scenes, and LaTeX unchanged',
                  'independent command-wise paired family-bootstrap intervals',
                  'all nine emitted error-count comparisons',
                  'instruction geometry reconstructed from eight saved target poses',
                  '56 footprint/envelope distance pairs and nearest-point checks'],
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(ROOT/'audit/submission_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

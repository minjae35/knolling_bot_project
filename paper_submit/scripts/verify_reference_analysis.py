"""Independently check the derived figure analyses against immutable saved records.

No figure-generator helpers, model imports, inference, or training are used.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import random
import numpy as np
from shapely.geometry import Polygon, Point

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT/name).read_text())

def close(a,b,tol=1e-8):
    assert abs(a-b)<tol,(a,b)

def geometry(asset,pose,envelope=False):
    if envelope:
        w,h=asset['a_m']/2,asset['b_m']/2
        points=[(-w,-h),(w,-h),(w,h),(-w,h)]
    else:
        points=asset['footprint']
    c,s=math.cos(pose['theta_rad']),math.sin(pose['theta_rad'])
    return Polygon([(pose['x_m']+c*x-s*y,pose['y_m']+s*x+c*y) for x,y in points])

def verify():
    mode_audit=read('audit/paired_mode_analysis.json')
    releases={}
    for version in ('v1','v2'):
        path=ROOT/'audit/evidence'/f'{version}_holdout.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==mode_audit['source_sha256'][path.name]
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        releases[version]={r['sample_id']:r for r in rows}
        assert len(rows)==len(releases[version])==640
    assert releases['v1'].keys()==releases['v2'].keys()
    families=sorted({r['scene_family_id'] for r in releases['v1'].values()})
    assert len(families)==128
    mode_names=['full','partial','no_op','overall']
    counts=np.zeros((128,4),dtype=int);gains=counts.copy()
    for i,family in enumerate(families):
        ids=[k for k,r in releases['v1'].items() if r['scene_family_id']==family]
        assert len(ids)==5
        for sid in ids:
            a,b=releases['v1'][sid],releases['v2'][sid]
            assert a['mode']==b['mode'] and a['scene_family_id']==b['scene_family_id']
            j=mode_names.index(a['mode'])
            for col in (j,3):
                counts[i,col]+=1;gains[i,col]+=int(b['task'])-int(a['task'])
    rng=random.Random(20261112)
    indices=np.array([[rng.randrange(128) for _ in range(128)] for _ in range(10000)])
    samples=100*gains[indices].sum(axis=1)/counts[indices].sum(axis=1)
    intervals={}
    for j,mode in enumerate(mode_names):
        expected=mode_audit['modes'][mode]
        assert int(counts[:,j].sum())==expected['n']
        assert int(gains[:,j].sum())==expected['paired_gain_count']
        close(100*gains[:,j].sum()/counts[:,j].sum(),expected['gain_pp'])
        lo,hi=np.sort(samples[:,j])[[250,9749]]
        for a,b in zip((lo,hi),expected['ci95_pp']):close(a,b)
        intervals[mode]=[float(lo),float(hi)]
    for row in mode_audit['error_counts']:
        for version in releases:
            keys={'horizontal_gap','vertical_gap'} if row['key']=='any_gap' else {row['key']}
            assert sum(bool(keys.intersection(r['errors'])) for r in releases[version].values())==row[version]
    constraint=read('audit/constraint_correspondence.json')
    objects={o['object_id']:o for o in read('audit/evidence/full8b_choice.json')['scene']['objects']}
    bounds={k:geometry(o['asset'],o['target_pose'],True).bounds for k,o in objects.items()}
    assert sorted(itertools.chain.from_iterable(r['ids'] for r in constraint['rows']))==sorted(objects)
    for row in constraint['rows']:
        assert all(objects[k]['color']==row['color'] for k in row['ids'])
        close(row['lower_y_m'],min(bounds[k][1] for k in row['ids']))
        close(row['upper_y_m'],max(bounds[k][3] for k in row['ids']))
    for item in constraint['horizontal_gaps']:
        actual=1000*(bounds[item['right']][0]-bounds[item['left']][2])
        close(actual,item['actual_mm']);assert abs(actual-item['requested_mm'])<=2
    for item in constraint['vertical_gaps']:
        actual=1000*(min(bounds[k][1] for k in item['above_ids'])-max(bounds[k][3] for k in item['below_ids']))
        close(actual,item['actual_mm']);assert abs(actual-item['requested_mm'])<=2
    gap_records=constraint['horizontal_gaps']+constraint['vertical_gaps']
    assert len(constraint['gap_residuals'])==len(gap_records)==7
    for plotted,measured in zip(constraint['gap_residuals'],gap_records):
        close(plotted['residual_mm'],measured['actual_mm']-measured['requested_mm'])
        close(plotted['requested_mm'],measured['requested_mm'])
    for seq in constraint['color_area_sequences'].values():
        sizes=[objects[k]['asset']['a_m']*objects[k]['asset']['b_m'] for k in seq]
        assert all(a>=b for a,b in zip(sizes,sizes[1:]))
    xbounds=[min(b[0] for b in bounds.values()),max(b[2] for b in bounds.values())]
    for a,b in zip(xbounds,constraint['evaluated_x_bounds_m']):close(a,b)
    assert .3<=xbounds[0]<xbounds[1]<=.6
    footprints=read('audit/footprint_clearance_analysis.json')
    reports=read('audit/evidence/v2_validation_smoke.json')
    sources=read('audit/evidence/v2_smoke_source_records.json')
    minima={}
    for mode in ('full','partial'):
        report=next(r for r in reports if r['mode']==mode)
        originals={o['object_id']:o for o in sources[report['sample_id']]['source']['objects']}
        meshes={};envelopes={}
        for pose in report['objects']:
            key=pose['object_id'];asset=originals[key]['asset']
            meshes[key]=geometry(asset,pose);envelopes[key]=geometry(asset,pose,True)
        entries=footprints['modes'][mode]['all_pairs']
        assert len(entries)==28
        observed=set()
        for entry in entries:
            a,b=entry['pair'];observed.add(tuple(sorted((a,b))))
            for prefix,polys in [('footprint',meshes),('envelope',envelopes)]:
                actual=1000*polys[a].distance(polys[b]);close(actual,entry[prefix+'_clearance_mm'])
                p,q=map(Point,entry[prefix+'_nearest_points_m'])
                assert polys[a].boundary.distance(p)<1e-9 and polys[b].boundary.distance(q)<1e-9
                close(1000*p.distance(q),actual)
        assert observed==set(itertools.combinations(sorted(meshes),2))
        minima[mode]={prefix:min(v[prefix+'_clearance_mm'] for v in entries) for prefix in ('footprint','envelope')}
    close(minima['partial']['footprint'],12.8198497339521)
    close(minima['partial']['envelope'],9.896102832824846)
    for manifest in (constraint,footprints):
        for path,sha in manifest['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    report={'status':'PASS','method':'Separate scalar geometry and array-based paired-family bootstrap; no figure-generator helpers.',
        'paired_records':640,'families':128,'bootstrap_resamples':10000,'mode_ci95_pp':intervals,
        'error_categories_checked':9,'target_objects_checked':8,'target_rows_checked':4,
        'pairwise_geometry_checks':56,'minimum_clearance_mm':minima,
        'target_envelope_x_bounds_m':xbounds,'source_hashes_verified':True,'new_inference':False}
    (ROOT/'audit/reference_analysis_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':print(json.dumps(verify(),indent=2))

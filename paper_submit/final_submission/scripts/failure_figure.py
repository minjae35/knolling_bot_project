"""Measured failures at saved poses: Blender crops plus honest numerical rulers.

Consumes only bundled frozen evidence and completed Blender renders. No model
inference or pose modification. Guides use true oriented rectangular envelopes,
not the narrower visible chamfer/capsule silhouettes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from shapely.geometry import Polygon as SPolygon
from shapely.ops import nearest_points
import make_figures as m

R=m.ROOT/'renders'
H=292
EXTENT=(-.02,.62,.2-.64/3,.2+.64/3)


def evidence():
    records=m.records()
    reports={r['mode']:r for r in json.loads((m.E/'v2_validation_smoke.json').read_text())}
    scenes={}
    for mode in ('full','partial'):
        report=reports[mode]
        objects={o['object_id']:o for o in records[report['sample_id']]['source']['objects']}
        poses={p['object_id']:p for p in report['objects']}
        polygons={k:SPolygon(m.envelope(objects[k],p)) for k,p in poses.items()}
        scenes[mode]=(objects,poses,polygons)
    objects,poses,polygons=scenes['full']
    rows=[]
    for p in sorted(poses.values(),key=lambda p:(-p['y_m'],p['x_m'])):
        if not rows or rows[-1][0]['y_m']-p['y_m']>.0101:rows.append([])
        rows[-1].append(p)
    gaps=[]
    for above,below in zip(rows,rows[1:]):
        bottom=min(polygons[p['object_id']].bounds[1] for p in above)
        top=max(polygons[p['object_id']].bounds[3] for p in below)
        gaps.append({'actual_mm':1000*(bottom-top),'upper_bottom_y_m':bottom,'lower_top_y_m':top,
                     'above_ids':[p['object_id'] for p in above],'below_ids':[p['object_id'] for p in below]})
    gap=max(gaps,key=lambda z:abs(z['actual_mm']-15))
    objects,poses,polygons=scenes['partial']
    pairs=[(polygons[a].distance(polygons[b]),a,b) for a in polygons for b in polygons if a<b]
    distance,a,b=min(pairs)
    near=nearest_points(polygons[a],polygons[b])
    result={'full':dict(gap,sample_id=reports['full']['sample_id'],allowed_band_mm=[13,17],
                        excess_over_upper_limit_mm=gap['actual_mm']-17,
                        guide_world_xy_m=[[.175,gap['upper_bottom_y_m']],[.175,gap['lower_top_y_m']]]),
            'partial':{'sample_id':reports['partial']['sample_id'],'closest_pair':[a,b],
                       'actual_clearance_mm':distance*1000,'nominal_minimum_mm':10.,'numerical_tolerance_mm':.1,
                       'effective_minimum_mm':9.9,'deficit_um':(9.9-distance*1000)*1000,
                       'deficit_native_pixels':(9.9-distance*1000)/(.64*1000/1920),
                       'nearest_points_world_xy_m':[[p.x,p.y] for p in near]},
            'provenance':{'method':'Geometry recomputed from frozen validation poses and original rectangular envelopes. These are illustrative saved predictions, not a new benchmark.',
                          'overhead_extent_world_m':EXTENT,
                          'input_sha256':{str(p.relative_to(m.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [m.E/'v2_validation_smoke.json',m.E/'v2_smoke_source_records.json',R/'smoke_full_overhead.png',R/'smoke_partial_overhead.png']}}}
    assert result['full']['excess_over_upper_limit_mm']>0
    assert 3.8<result['partial']['deficit_um']<4.
    return result,scenes


def crop(fig,mode,x,y,w,h,lims):
    ax=fig.add_axes([x/396,1-(y+h)/H,w/396,h/H])
    ax.imshow(plt.imread(R/f'smoke_{mode}_overhead.png'),extent=EXTENT)
    ax.set(xlim=lims[:2],ylim=lims[2:],aspect='equal');ax.axis('off')
    return ax


def ruler(ax,x,y,w,limits,allowed,actual,ticks):
    lo,hi=limits
    pos=lambda value:x+w*(value-lo)/(hi-lo)
    # This axis is explicitly numerical, not a claim that the visible error is large.
    ax.add_patch(Rectangle((pos(allowed[0]),y-5),pos(allowed[1])-pos(allowed[0]),10,fc='#dcece6',ec='none'))
    ax.plot([x,x+w],[y,y],color=m.MUTED,lw=.65)
    for tick,label in ticks:
        xp=pos(tick);ax.plot([xp,xp],[y+1,y+4],color=m.MUTED,lw=.65)
        m.text(ax,xp,y+12,label,7.5,m.MUTED,ha='right' if tick==hi else 'center')
    ax.plot(pos(actual),y,'o',color=m.RED,ms=4.5,zorder=5)
    for boundary in allowed:
        if lo<boundary<hi:ax.plot([pos(boundary),pos(boundary)],[y-8,y+4],color=m.TEAL,lw=.85)
    return pos


def generate():
    values,scenes=evidence()
    (m.ROOT/'audit/failure_annotations.json').write_text(json.dumps(values,indent=2)+'\n')
    fig,base=m.canvas(H)
    m.text(base,0,8,'a  Full arrangement: a row gap is too wide',9,weight='bold')
    a=crop(fig,'full',0,27,178,89,(.16,.44,.095,.22))
    g=values['full'];bottom,top=g['upper_bottom_y_m'],g['lower_top_y_m']
    # Limit guide is measured relative to the lower row's upper envelope.
    for y in [bottom,top]:a.plot([.17,.425],[y,y],color=m.RED,lw=.65,ls=(0,(3,2)))
    a.annotate('',(.178,bottom),(.178,top),arrowprops={'arrowstyle':'<->','lw':1,'color':m.RED,'shrinkA':0,'shrinkB':0})
    a.text(.19,(bottom+top)/2,'17.657 mm',fontsize=8,color=m.RED,va='center',
           bbox=dict(fc='white',ec='none',pad=1.5))
    a.axhspan(top+.017,bottom,xmin=.07,xmax=.94,fc=m.RED,alpha=.16,zorder=4)
    m.text(base,0,127,'Dashed guides: true row-envelope edges',7.4,m.MUTED)
    m.text(base,196,30,'17.657 mm',15,m.RED,'bold')
    m.text(base,196,48,'0.657 mm above the upper limit',8.5,m.RED,weight='bold')
    m.text(base,196,67,'Requested 15 mm; tolerance ±2 mm',8.0)
    m.text(base,196,84,'Allowed interval',7.6,m.TEAL)
    ruler(base,205,100,176,(12,19),(13,17),g['actual_mm'],[(13,'13'),(15,'15'),(17,'17'),(19,'19 mm')])
    m.text(base,196,128,'A tidy-looking row is still out of tolerance.',7.5,m.MUTED)
    base.plot([0,396],[143,143],color=m.LINE,lw=.7)
    m.text(base,0,157,'b  Blue-only placement: a borderline clearance failure',9,weight='bold')
    b=crop(fig,'partial',0,176,178,106.8,(.13,.32,.15,.264))
    objects,poses,polygons=scenes['partial']
    for oid in values['partial']['closest_pair']:
        b.add_patch(Polygon(m.envelope(objects[oid],poses[oid]),fill=False,ec=m.INK,lw=.8,ls=(0,(3,2))))
    near=np.array(values['partial']['nearest_points_world_xy_m'])
    b.plot(near[:,0],near[:,1],color=m.RED,lw=1.4,zorder=6)
    b.plot(near[:,0],near[:,1],'o',mfc='white',mec=m.RED,ms=3,mew=.9,zorder=7)
    center=near.mean(axis=0)
    b.annotate('9.896 mm',xy=center,xytext=(.248,.173),fontsize=8,color=m.RED,ha='center',
               arrowprops=dict(arrowstyle='-',color=m.RED,lw=.8),bbox=dict(fc='white',ec='none',pad=1.4),zorder=8)
    m.text(base,196,181,'9.896 mm',15,m.RED,'bold')
    m.text(base,196,199,'3.9 µm below the effective minimum',8.3,m.RED,weight='bold')
    m.text(base,196,216,'10 mm nominal − 0.1 mm tolerance',8)
    m.text(base,196,232,'Expanded numerical scale',7.6,m.MUTED)
    ruler(base,205,248,176,(9.89,9.91),(9.90,9.91),values['partial']['actual_clearance_mm'],
          [(9.89,'9.890'),(9.90,'9.900'),(9.91,'9.910 mm')])
    m.text(base,196,278,'Shortfall ≈0.012 render pixels; no collision.',7.4,m.MUTED)
    m.save(fig,'failure_explanation')
    return values

if __name__=='__main__':generate()

"""Publication artwork from immutable saved predictions and scores.

No model imports, inference, training, or image synthesis. Coordinates in the
architecture canvas are typographic points; output width is the paper's 5.5 in.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon as SPolygon
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'audit/evidence'
OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)
for font in (ROOT/'fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(font))
INK='#182530'; MUTED='#596874'; BLUE='#245c91'; TEAL='#087e81'; RED='#a83b39'
GRAY='#7d8b98'; LINE='#cbd3d9'; LIGHT='#f3f6f8'; PALETTE={'red':'#d45b59','green':'#55a870','blue':'#507bc2'}
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.titlesize':8.5,
 'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix','text.color':INK,
 'axes.labelcolor':MUTED,'axes.edgecolor':LINE,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'xtick.color':MUTED,'ytick.color':MUTED,
 'savefig.facecolor':'white','figure.facecolor':'white'})


def save(fig,name,replace_frozen=False):
    if name == 'qualitative_modes' and not replace_frozen:
        raise RuntimeError('Figure 6 is frozen at the user\'s request. Reuse its existing exports unchanged.')
    # Fixed physical dimensions ensure font sizes survive LaTeX inclusion.
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,metadata={'Creator':'Knolling Bot paper figure script'} if ext=='pdf' else None)
    plt.close(fig)


def canvas(h):
    fig=plt.figure(figsize=(5.5,h/72));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,396),ylim=(h,0));ax.axis('off')
    return fig,ax


def text(ax,x,y,s,size=8,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,**kw)


def route(ax,points,color=MUTED,dashed=False,head=True,lw=.85):
    if len(points)>2:
        xs,ys=zip(*points[:-1]);ax.plot(xs,ys,color=color,lw=lw,ls=(0,(3,2)) if dashed else '-',solid_joinstyle='round')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if head else '-',
      mutation_scale=7,lw=lw,color=color,linestyle=(0,(3,2)) if dashed else '-'))




def apply_pose(foot,pose):
    t=pose['theta_rad'];rot=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
    return np.asarray(foot)@rot.T+[pose['x_m'],pose['y_m']]


def envelope(obj,p):
    a,b=obj['asset']['a_m']/2,obj['asset']['b_m']/2
    return apply_pose([[-a,-b],[a,-b],[a,b],[-a,b]],p)








def quantitative():
    d=json.loads((ROOT/'audit/verified_results.json').read_text())
    fig=plt.figure(figsize=(5.5,2.35));ax=fig.add_axes([.09,.22,.34,.62]);bx=fig.add_axes([.69,.22,.20,.62])
    fig.text(.01,.95,'a  Scene complexity',fontsize=9,weight='bold')
    fig.text(.53,.95,'b  Constraint failures',fontsize=9,weight='bold')
    for name,color,marker in [('v1',GRAY,'o'),('v2',BLUE,'s')]:
        counts=d[name]['by_object_count'];x=np.array([int(k) for k in counts]);y=np.array([100*v['success']/v['total'] for v in counts.values()])
        ax.plot(x,y,color=color,marker=marker,ms=3.5,lw=1.4,label=name)
        ax.text(8.15,y[-1],f'{y[-1]:.1f}',fontsize=7.4,color=color,va='center',weight='bold')
    ax.set(xlim=(1.75,9.25),ylim=(0,103),xticks=range(2,9),yticks=[0,25,50,75,100],xlabel='Objects per scene',ylabel='Strict success (%)')
    ax.grid(axis='y',color='#e6ebef',lw=.5);ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=7.5,loc='lower left',ncol=2,handlelength=1.6,columnspacing=1)
    keys=['horizontal_gap','vertical_gap','size_order','clearance','count','requested_region','keep']
    labels=['Horizontal gap','Vertical gap','Size order','Clearance','Object count','Region','Preservation']
    for y,k in enumerate(keys):
        a,b=d['v1']['error_tags'][k],d['v2']['error_tags'][k]
        bx.plot([a,b],[y,y],color='#c6cfd7',lw=2,zorder=1)
        bx.plot(a,y,'o',mfc='white',mec=GRAY,ms=4.1,mew=1,zorder=2)
        bx.plot(b,y,'o',color=BLUE,ms=4.1,zorder=3)
        fig.text(.995,.22+.62*(6.6-y)/7.2,f'{a} → {b}',ha='right',va='center',fontsize=7.0,color=INK)
    bx.set(yticks=range(7),yticklabels=labels,xlim=(0,55),ylim=(6.6,-.6),xticks=[0,25,50],xlabel='Scenes (tags overlap)')
    bx.grid(axis='x',color='#e6ebef',lw=.5);bx.set_axisbelow(True);bx.spines['left'].set_visible(False);bx.tick_params(axis='y',length=0,pad=4)
    
    fig.text(.995,.865,'v1 → v2',fontsize=7.0,color=MUTED,ha='right')
    save(fig,'holdout_diagnostics')


def records():
    r=json.loads((E/'v2_smoke_source_records.json').read_text())
    if isinstance(r,dict):r=r.get('records',list(r.values()))
    return {x['sample_id']:x for x in r}


def measurements():
    rec=records();rep={r['mode']:r for r in json.loads((E/'v2_validation_smoke.json').read_text())}
    result={}
    for mode,r in rep.items():
        objs={o['object_id']:o for o in rec[r['sample_id']]['source']['objects']}
        poses={p['object_id']:p for p in r['objects']}
        polys={k:SPolygon(envelope(objs[k],p)) for k,p in poses.items()}
        pairs=[(a,b,polys[a].distance(polys[b])) for a in polys for b in polys if a<b]
        a,b,gap=min(pairs,key=lambda x:x[2]);near=nearest_points(polys[a],polys[b])
        result[mode]={'sample_id':r['sample_id'],'minimum_clearance_mm':1000*gap,'closest_pair':[a,b],
          'nearest_points':[[p.x,p.y] for p in near]}
        if mode=='full':
            rows=[]
            for p in sorted(poses.values(),key=lambda p:(-p['y_m'],p['x_m'])):
                if not rows or rows[-1][0]['y_m']-p['y_m']>.0101:rows.append([])
                rows[-1].append(p)
            gaps=[]
            for up,dn in zip(rows,rows[1:]):
                lo=min(polys[p['object_id']].bounds[1] for p in up);hi=max(polys[p['object_id']].bounds[3] for p in dn)
                gaps.append({'actual_mm':(lo-hi)*1000,'upper_bottom':lo,'lower_top':hi,
                  'upper_ids':[p['object_id'] for p in up],'lower_ids':[p['object_id'] for p in dn]})
            result[mode]['vertical_gaps']=gaps
        if mode=='no_op':
            result[mode]['max_position_error_mm']=max(p['shift_from_source_mm'] for p in r['objects'])
            result[mode]['max_angle_error_deg']=max(p['turn_from_source_deg'] for p in r['objects'])
    result['provenance']={'method':'Read-only geometry of saved predicted poses with true source envelopes, matching evaluator distance/row definitions. No new inference.',
      'inputs':{n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['v2_validation_smoke.json','v2_smoke_source_records.json']}}
    (ROOT/'audit/visual_measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['full']['vertical_gaps'][1]['actual_mm']>17
    assert result['partial']['minimum_clearance_mm']<9.9
    assert result['no_op']['max_position_error_mm']<1 and result['no_op']['max_angle_error_deg']<1
    return result,rec,rep




def development():
    d=json.loads((E/'selector_full_result.json').read_text());c=d['curve'];x=[v['epoch'] for v in c]
    fig=plt.figure(figsize=(5.5,2.0));ax=fig.add_axes([.09,.20,.34,.62]);bx=fig.add_axes([.62,.20,.35,.62])
    fig.text(.01,.93,'a  Structure improves',fontsize=9,weight='bold')
    fig.text(.53,.93,'b  Complete-task success plateaus',fontsize=9,weight='bold')
    for key,color,mark,label in [('val_row_accuracy',BLUE,'o','Row'),('val_slot_accuracy',TEAL,'s','Slot')]:
        ys=[100*v[key] for v in c];ax.plot(x,ys,color=color,marker=mark,ms=3,lw=1.3,label=label)
    ax.set(xlabel='Training epoch',ylabel='Validation accuracy (%)',ylim=(60,90),xticks=x,yticks=[60,70,80,90])
    ax.legend(frameon=False,fontsize=7.6,loc='lower right',ncol=2,handlelength=1.2,columnspacing=.8)
    vals=[100*v['val_task']/v['val_total'] for v in c]
    bx.axvspan(.8,1.2,fc='#dce9f3',ec='none');bx.plot(x,vals,color=BLUE,marker='o',ms=3,lw=1.3)
    bx.text(1.25,22,'235/1200',fontsize=7.5,color=BLUE);bx.text(7.9,22,'233/1200',fontsize=7.5,color=BLUE,ha='right')
    bx.text(1.25,3,'Selected epoch',fontsize=7.5,color=MUTED)
    bx.set(xlabel='Training epoch',ylabel='Strict success (%)',ylim=(0,25),xticks=x,yticks=[0,5,10,15,20,25])
    for a in [ax,bx]:a.grid(axis='y',color='#e6ebef',lw=.5);a.set_axisbelow(True)
    save(fig,'selector_training')

if __name__=='__main__':
    from publication_figures import generate
    generate()
    print('Seven figures regenerated; user-approved Figure 6 preserved byte-for-byte.')

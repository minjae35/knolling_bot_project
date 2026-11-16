"""Registered source/target footprints and exact pose changes from saved predictions.

This analytic world-plane view complements the Blender result photographs. Straight
arrows encode displacement, never an inferred robot path. No poses are changed.
"""
from pathlib import Path
import math,json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import matplotlib.patheffects as pe
import make_figures as m
R=m.ROOT/'renders';SOURCE=m.E/'full8b_choice.json';META=m.ROOT/'audit/blender_provenance.json'
COLORS={'red':'#b83c39','green':'#287a48','blue':'#3566b2'}


def build():
    scene=json.loads(SOURCE.read_text());cfg=json.loads(META.read_text())['oblique_camera']
    camera=np.array(cfg['location']);target=np.array(cfg['target']);back=(camera-target)/np.linalg.norm(camera-target)
    right=np.cross([0,0,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
    saved=json.loads((m.ROOT/'audit/blender_pose_checks.json').read_text())['renders']['full_before']['camera_euler']
    x,y,z=saved
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    rotation=rz@ry@rx;basis_error=float(max(abs(rotation[:,0]-right).max(),abs(rotation[:,1]-up).max()))
    rows=[]
    for o in scene['scene']['objects']:
        a,b=o['source_pose'],o['target_pose'];delta=(b['theta_rad']-a['theta_rad']+math.pi/2)%math.pi-math.pi/2
        rows.append({'object_id':o['object_id'],'number':int(o['object_id'][1:]),'color':o['color'],'source':a,'target':b,
          'height_m':o['asset']['h_m'],'major_m':o['asset']['a_m'],
          'translation_mm':1000*math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']), 'delta_theta_deg':math.degrees(delta)})
    height=232;fig,ax=m.canvas(height)
    m.text(ax,0,8,'a  Registered source and target poses',8.3,weight='bold')
    m.text(ax,278,8,'b  Pose change',8.3,weight='bold')
    # Common metric projection. No independent object placement or deconfliction.
    rect=(1,30,260,173.3333333333)
    def xy(x,y):return np.array([rect[0]+x/.6*rect[2],rect[1]+(1-y/.4)*rect[3]])
    def pose(p):return xy(p['x_m'],p['y_m'])
    ax.add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],fc='#fafafa',ec='#c6cdd2',lw=.55))
    rr=xy(.32,.38);ax.add_patch(Rectangle(rr,.26/.6*rect[2],.36/.4*rect[3],fc='#edf3f7',ec='none'))
    m.text(ax,254,39,'Requested right region',7,m.MUTED,ha='right')
    # Physical footprints, not surrogate bounding boxes.
    for o in scene['scene']['objects']:
        c=COLORS[o['color']]
        for field,fill,ls,lw in [('source_pose',True,(0,(2,1.4)),.7),('target_pose',False,'-',1.0)]:
            points=m.apply_pose(o['asset']['footprint'],o[field]);points=np.array([xy(*p) for p in points])
            ax.add_patch(Polygon(points,fc=c+'12' if fill else 'none',ec=c,ls=ls,lw=lw,zorder=2 if fill else 3))
    for r in rows:
        a,b=pose(r['source']),pose(r['target'])
        ar=FancyArrowPatch(a,b,arrowstyle='-|>',color='#b66a24',lw=.75,mutation_scale=5.5,shrinkA=1,shrinkB=4,zorder=4)
        ar.set_path_effects([pe.Stroke(linewidth=1.5,foreground='white',alpha=.85),pe.Normal()]);ax.add_patch(ar)
    # Source labels outside silhouettes; target IDs are filled circles at their centers.
    offsets={1:(-3,-14),2:(-8,-13),3:(12,-11),4:(-15,-2),5:(-14,-1),6:(-9,14),7:(-16,-5),8:(14,5)}
    for r in rows:
        c=COLORS[r['color']];a,b=pose(r['source']),pose(r['target']);q=a+offsets[r['number']]
        ax.plot([a[0],q[0]],[a[1],q[1]],color=c,lw=.35,zorder=5)
        ax.add_patch(Circle(q,4.1,fc='white',ec=c,lw=.6,zorder=6));m.text(ax,*q,str(r['number']),7.1,c,ha='center',zorder=7)
        ax.add_patch(Circle(b,4.3,fc=c,ec='white',lw=.5,zorder=7));m.text(ax,*b,str(r['number']),7.1,'white','bold',ha='center',zorder=8)
        # End segments of the long axis remain visible around target identity marks.
        for field in ['source','target']:
            p=r[field];d=.32*r['major_m'];t=p['theta_rad'];a0=xy(p['x_m']-d*math.cos(t),p['y_m']-d*math.sin(t));b0=xy(p['x_m']+d*math.cos(t),p['y_m']+d*math.sin(t))
            ax.plot([a0[0],b0[0]],[a0[1],b0[1]],color=c,lw=.45,ls=(0,(1.4,1)) if field=='source' else '-',zorder=3.5)
    # Readable ledger with an independent orientation glyph for every object.
    m.text(ax,280,29,'ID',7.2,weight='bold');m.text(ax,314,29,'axes',7.2,m.MUTED,ha='center')
    m.text(ax,351,29,'Δs (mm)',7.2,ha='right');m.text(ax,394,29,'Δθ (°)',7.2,ha='right')
    ax.plot([278,396],[37,37],color=m.LINE,lw=.6)
    for i,r in enumerate(rows):
        yy=48+i*19;c=COLORS[r['color']]
        ax.add_patch(Circle((284,yy),4.2,fc=c,ec='none'));m.text(ax,284,yy,str(r['number']),7.2,'white','bold',ha='center')
        xx=313;rad=7
        for key,col,lw,ls in [('source','#9ba6ae',.8,(0,(1.6,1))),('target',c,1.1,'-')]:
            t=r[key]['theta_rad'];v=np.array([math.cos(t),-math.sin(t)])*rad
            ax.plot([xx-v[0],xx+v[0]],[yy-v[1],yy+v[1]],color=col,lw=lw,ls=ls)
        t0=r['source']['theta_rad'];dt=math.radians(r['delta_theta_deg']);tt=np.linspace(t0,t0+dt,20)
        pp=np.c_[xx+5*np.cos(tt),yy-5*np.sin(tt)];ax.plot(pp[:,0],pp[:,1],color=c,lw=.6)
        m.text(ax,351,yy,f"{r['translation_mm']:.1f}",7.6,ha='right')
        m.text(ax,394,yy,f"{r['delta_theta_deg']:+.1f}",7.6,ha='right')
    # Compact explicit semantics, outside the scene.
    ax.plot([2,17],[215,215],color=m.GRAY,lw=.8,ls=(0,(2,1.4)));m.text(ax,21,215,'Source',7.2)
    ax.plot([68,83],[215,215],color=m.BLUE,lw=1);m.text(ax,87,215,'Prediction',7.2)
    ax.add_patch(FancyArrowPatch((148,215),(166,215),arrowstyle='-|>',mutation_scale=6,color='#b66a24',lw=.8));m.text(ax,170,215,'Center displacement',7.2)
    m.text(ax,394,215,'CCW +; π-periodic',7.2,m.MUTED,ha='right')
    m.text(ax,0,228,'Straight connections show pose differences, not robot trajectories.',7.2,m.MUTED)
    m.save(fig,'motion_correspondence')
    provenance={'sample_id':scene['sample_id'],'camera':cfg,'camera_right':right.tolist(),'camera_up':up.tolist(),'saved_blender_camera_basis_max_error':basis_error,
      'projection':'Analytic top-down world-plane view; all footprints share one isotropic meter-to-point map. Blender comparison uses the separately audited camera.',
      'angle_convention':'Signed target-source difference in [-90,90) degrees; positive CCW; undirected long-axis orientation.',
      'arrow_semantics':'Straight source-to-target center displacement, not robot trajectories.',
      'sources':{str(p.relative_to(m.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,META,m.ROOT/'audit/blender_pose_checks.json',R/'full_before.png',R/'full_after.png']},
      'objects':rows,'figures':{'motion_correspondence':{'width_pt':396,'height_pt':height}}}
    (m.ROOT/'audit/motion_annotations.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance

if __name__=='__main__':build()

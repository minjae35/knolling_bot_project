"""Recorded-source to predicted-target motion at immutable saved poses.

Actual Blender views supply object identity and spatial context. Two aligned
per-object plots separate direction, displacement magnitude and signed rotation.
The original pose/spacing audits remain immutable; current artwork has its own
motion_visualization.json provenance. No model, scene or render is modified.
"""
import math,json,hashlib
import numpy as np
from figure_tex import compile_figure, node, picture
from preserve_figure6 import verify as verify_figure6
import make_figures as m
R=m.ROOT/'renders';SOURCE=m.E/'full8b_choice.json';META=m.ROOT/'audit/blender_provenance.json'
COLORS={'red':'objectred','green':'objectgreen','blue':'objectblue'}
HEIGHT=318
# Air between the a/b status lines and panels (c) and (d). The axis labels
# keep only a short margin so the caption does not float below the plot.
PANEL_SHIFT=28
# a/b title ink ends 7.7 pt above the photo. The same clearance above the
# first chart row, whose circle extends 3.6 pt above its center.
CHART_DROP=2


def build():
    protected_before=verify_figure6()
    scene=json.loads(SOURCE.read_text());cfg=json.loads(META.read_text())['oblique_camera']
    camera=np.array(cfg['location']);target=np.array(cfg['target']);back=(camera-target)/np.linalg.norm(camera-target)
    right=np.cross([0,0,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
    saved=json.loads((m.ROOT/'audit/blender_pose_checks.json').read_text())['renders']['full_before']['camera_euler']
    x,y,z=saved
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    rotation=rz@ry@rx;basis_error=float(max(abs(rotation[:,0]-right).max(),abs(rotation[:,1]-up).max()))
    assert basis_error<1e-7
    rows=[]
    for o in scene['scene']['objects']:
        a,b=o['source_pose'],o['target_pose'];delta=(b['theta_rad']-a['theta_rad']+math.pi/2)%math.pi-math.pi/2
        rows.append({'object_id':o['object_id'],'number':int(o['object_id'][1:]),'color':o['color'],'source':a,'target':b,
          'height_m':o['asset']['h_m'],'major_m':o['asset']['a_m'],
          'translation_mm':1000*math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']), 'delta_theta_deg':math.degrees(delta)})
    artwork=[]
    def add(command):artwork.append(command)
    def line(a,b,options='leader'):
        add(rf'\draw[{options}] ({a[0]:.7f},{a[1]:.7f}) -- ({b[0]:.7f},{b[1]:.7f});'+'\n')
    def rectangle(a,b,options='frame'):
        add(rf'\path[{options}] ({a[0]:.7f},{a[1]:.7f}) rectangle ({b[0]:.7f},{b[1]:.7f});'+'\n')
    def title(x,y,letter,label):
        add(node(x,y,rf'\textbf{{{letter}}}\enspace {label}',
                 r'anchor=west,font=\fontsize{11}{13}\selectfont'))
    def project(p,z,ox):
        v=np.array([p[0],p[1],z])-target
        return np.array([ox+192*(.5+v@right/cfg['ortho_scale_m']),22+128*(.5-v@up/(cfg['ortho_scale_m']*2/3))])
    label_boxes=[]
    for ox,label,name,field in [(0,'a  Recorded source','full_before','source_pose'),(204,'b  Predicted target','full_after','target_pose')]:
        title(ox,8,label[0],label[3:])
        add(picture(f'renders/{name}.png',ox,22,192,128))
        for o in scene['scene']['objects']:
            p=o[field];z=o['asset']['h_m'];center=project([p['x_m'],p['y_m']],z,ox)
            footprint=m.apply_pose(o['asset']['footprint'],p)
            projected=np.array([project(v,z,ox) for v in footprint])
            marker=np.array([center[0],projected[:,1].min()-4.1])
            if field=='target_pose':
                split=np.median([v['target_pose']['x_m'] for v in scene['scene']['objects']])
                marker=np.array([projected[:,0].min()-5 if p['x_m']<split else projected[:,0].max()+5,center[1]])
            marker[0]=np.clip(marker[0],ox+5,ox+187);marker[1]=max(marker[1],27)
            color=COLORS[o['color']]
            line(center,marker,f'draw={color},line width=.35bp')
            add(rf'\path[draw={color},line width=.45bp,fill=white] ({marker[0]:.7f},{marker[1]:.7f}) circle[radius=3.6bp];'+'\n')
            add(node(*marker,o['object_id'][1:].lstrip('0'),f'anchor=center,text={color}'))
            label_boxes.append({'panel':field,'object_id':o['object_id'],'center_pt':marker.tolist()})
    status_font=r'font=\fontsize{9.5}{11.5}\selectfont'
    add(node(0,164,'Eight objects; stable IDs across both views',
             'anchor=west,text=muted,'+status_font))
    add(node(204,164,r'PASS $\cdot$ Right region $\cdot$ Color-grouped rows',
             'anchor=west,text=accent,'+status_font))

    # A photo edge holds a 8 pt gap. An open chart reads tighter, so these
    # titles sit 4 pt higher and the grid starts at the first row, not above it.
    title_y=162+PANEL_SHIFT
    title(0,title_y,'c','Translation (mm)')
    title(204,title_y,'d',r'Rotation $\Delta\theta$ ($^\circ$)')
    add(node(394,title_y,r'$-\,$CW $/\,+\,$CCW','anchor=east,text=muted'))
    # Common row order and the same object colors connect both measurements
    # to the source/target render IDs. Direction is normalized explicitly;
    # the shared quantitative axis, not arrow length, encodes displacement.
    distance_left,distance_right=42.,151.
    rotation_left,rotation_right=239.,361.
    def tx(value):return distance_left+value/300*(distance_right-distance_left)
    def rx(value):return rotation_left+(value+90)/180*(rotation_right-rotation_left)
    y_rows=[181.+PANEL_SHIFT+CHART_DROP+11*i for i in range(8)]
    axis_y=267.+PANEL_SHIFT+CHART_DROP;tick_y=277.+PANEL_SHIFT+CHART_DROP
    grid_top=y_rows[0]-3.6
    for tick in (0,100,200,300):
        xx=tx(tick)
        line((xx,grid_top),(xx,262+PANEL_SHIFT+CHART_DROP),'draw=hairline!55,line width=.25bp')
        line((xx,axis_y),(xx,axis_y+2),'draw=muted,line width=.35bp')
        add(node(xx,tick_y,str(tick),'anchor=center,text=muted'))
    for tick in (-90,0,90):
        xx=rx(tick)
        style='draw=muted,line width=.4bp' if tick==0 else 'draw=hairline!55,line width=.25bp'
        line((xx,grid_top),(xx,262+PANEL_SHIFT+CHART_DROP),style)
        line((xx,axis_y),(xx,axis_y+2),'draw=muted,line width=.35bp')
        add(node(xx,tick_y,rf'${tick:+d}$' if tick else '$0$','anchor=center,text=muted'))
    line((distance_left,axis_y),(distance_right,axis_y),'draw=muted,line width=.35bp')
    line((rotation_left,axis_y),(rotation_right,axis_y),'draw=muted,line width=.35bp')
    plotted=[]
    for row,yy in zip(sorted(rows,key=lambda r:r['number']),y_rows):
        col=COLORS[row['color']]
        for xx in (5,213):
            add(rf'\path[draw={col},line width=.45bp,fill=white] ({xx},{yy}) circle[radius=3.6bp];'+'\n')
            add(node(xx,yy,str(row['number']),f'anchor=center,text={col}'))
        a,b=row['source'],row['target']
        delta=np.array([b['x_m']-a['x_m'],b['y_m']-a['y_m']])
        direction=delta/np.linalg.norm(delta)
        mid=np.array([23.,yy]);screen_direction=direction*[1,-1]
        start=mid-4*screen_direction;end=mid+4*screen_direction
        line(start,end,f'draw={col},line width=.6bp,-{{Latex[length=2.4bp,width=1.8bp]}}')
        dist=row['translation_mm'];angle=row['delta_theta_deg']
        line((tx(0),yy),(tx(dist),yy),f'draw={col}!70,line width=.85bp')
        add(rf'\path[draw=white,line width=.4bp,fill={col}] ({tx(dist):.7f},{yy}) circle[radius=1.8bp];'+'\n')
        add(node(190,yy,rf'${dist:.1f}$','anchor=east'))
        line((rx(0),yy),(rx(angle),yy),f'draw={col}!70,line width=.85bp')
        add(rf'\path[draw=white,line width=.4bp,fill={col}] ({rx(angle):.7f},{yy}) circle[radius=1.8bp];'+'\n')
        add(node(394,yy,rf'${angle:+.1f}$','anchor=east'))
        plotted.append({'object_id':row['object_id'],'row_y_bp':yy,
            'source_pose':a,'predicted_target_pose':b,
            'dx_mm':float(delta[0]*1000),'dy_mm':float(delta[1]*1000),
            'translation_mm':dist,'delta_theta_deg':angle,
            'direction_world_unit':direction.tolist(),
            'direction_arrow_start_bp':start.tolist(),'direction_arrow_end_bp':end.tolist(),
            'distance_point_bp':[tx(dist),yy],'rotation_point_bp':[rx(angle),yy],
            'display_distance_mm':f'{dist:.1f}','display_delta_theta_deg':f'{angle:+.1f}'})
    # Independently compare every derived pose difference with the frozen audit.
    frozen_path=m.ROOT/'audit/motion_annotations.json'
    frozen=json.loads(frozen_path.read_text())
    for current,previous in zip(rows,frozen['objects']):
        assert current==previous
    assert [(b['panel'],b['object_id']) for b in label_boxes]==[(b['panel'],b['object_id']) for b in frozen['identity_markers']]
    typeset=compile_figure('motion_correspondence',''.join(artwork),HEIGHT,label_pt=8.5)
    protected_after=verify_figure6()
    report={'sample_id':scene['sample_id'],
        'source_sha256':{str(p.relative_to(m.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [SOURCE,META,m.ROOT/'audit/blender_pose_checks.json',frozen_path,R/'full_before.png',R/'full_after.png']},
        'physical_size_bp':[396,HEIGHT],
        'source_pose_semantics':'Recorded synthetic source pose, not detector-estimated pose.',
        'target_pose_semantics':'Frozen saved model prediction, not dataset teacher target.',
        'identity_semantics':'Source-asset correspondence retained across actual render views and both measurement plots; not a learned-tracking claim.',
        'displacement_semantics':'Equal-length 8 bp arrows encode world-xy direction only (+x right,+y up); a common 0..300 mm axis encodes Euclidean endpoint displacement.',
        'rotation_semantics':'Signed shortest undirected long-axis change in [-90,90) degrees; positive counterclockwise; exact unsnapped predicted angles used.',
        'not_robot_trajectories':True,'new_inference':False,'model_dataset_checkpoint_or_render_modified':False,
        'object_rows':plotted,'identity_markers':label_boxes,
        'distance_axis':{'range_mm':[0,300],'range_bp':[distance_left,distance_right]},
        'rotation_axis':{'range_deg':[-90,90],'range_bp':[rotation_left,rotation_right]},
        'frozen_pose_annotation_bytes_preserved':True,
        'figure6_before':protected_before,'figure6_after':protected_after,
        'editable_source':typeset['source'],'exports_sha256':typeset['exports_sha256']}
    (m.ROOT/'audit/motion_visualization.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':build()

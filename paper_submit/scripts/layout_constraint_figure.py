"""Preserve the saved region/spacing audit as the appendix's Figure 8.

The drawing uses the frozen constraint audit. The script
never writes that audit, the motion record, any saved pose, or other figures.
"""
from pathlib import Path
import hashlib
import json
import math
from figure_tex import compile_figure, node, picture
from preserve_figure6 import verify as preserve_figure6

ROOT=Path(__file__).resolve().parents[1]
CONSTRAINT=ROOT/'audit/constraint_correspondence.json'
SOURCE=ROOT/'audit/evidence/full8b_choice.json'
IMAGE=ROOT/'renders/detail_full_workspace.png'
WIDTH,HEIGHT=396,157
IMG_W,IMG_H,IMG_X,IMG_Y=198.0,132.0,0.0,2.0
PLOT_L,PLOT_R=216.0,390.0
DATA_L,DATA_R=308.0,384.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    protected=preserve_figure6()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (CONSTRAINT,SOURCE,IMAGE)}
    audit=json.loads(CONSTRAINT.read_text())
    choice=json.loads(SOURCE.read_text())
    assert choice['sample_id']==audit['sample_id']
    assert sha(IMAGE)==audit['overhead_render']['image_sha256']
    assert sha(SOURCE)==audit['source_sha256']['audit/evidence/full8b_choice.json']
    objects={o['object_id']:o for o in choice['scene']['objects']}
    art=[]
    def label(x,y,text,options='anchor=west'):
        art.append(node(x,y,text,options))
    def line(a,b,options='leader'):
        art.append(rf'\draw[{options}] ({a[0]:.7f},{a[1]:.7f}) -- ({b[0]:.7f},{b[1]:.7f});'+'\n')
    def rect(a,b,options):
        art.append(rf'\path[{options}] ({a[0]:.7f},{a[1]:.7f}) rectangle ({b[0]:.7f},{b[1]:.7f});'+'\n')
    xmin,xmax,ymin,ymax=audit['overview_world_m']
    scale=audit['overview_scale_pt_per_m']
    assert abs(scale-144/(xmax-xmin))<1e-9 and abs(scale-96/(ymax-ymin))<1e-9
    sx,sy=IMG_W/144.0,IMG_H/96.0
    def overview(x,y):
        return (IMG_X+(x-xmin)*scale*sx, IMG_Y+(ymax-y)*scale*sy)
    art.append(picture('renders/detail_full_workspace.png',IMG_X,IMG_Y,IMG_W,IMG_H))
    line(overview(.3,.4),overview(.3,0),'draw=teal,line width=.6bp,dash pattern=on 2.4bp off 1.8bp')
    label(IMG_X+78*sx,IMG_Y+11*sy,'Right region','anchor=west,text=teal')
    for i,row in enumerate(audit['rows'],1):
        cy=sum(objects[oid]['target_pose']['y_m'] for oid in row['ids'])/len(row['ids'])
        label(IMG_X+62*sx,overview(0,cy)[1],f'r{i}','anchor=east')
    panel_bot=IMG_Y+IMG_H
    label(0,panel_bot+14,r'0.60 $\times$ 0.40 m workspace','anchor=west,text=muted')

    lo,hi=audit['chart_x_limits_mm']
    plot_l,plot_r=PLOT_L,PLOT_R
    def old_chart_x(value):return 227+(value-lo)/(hi-lo)*(390-227)
    def chart_x(value):return DATA_L+(value-lo)/(hi-lo)*(DATA_R-DATA_L)
    within_ys=(36,48,60,72)
    between_ys=(96,108,120)
    rect((plot_l,IMG_Y),(plot_r,panel_bot),'draw=none,fill=teal!7')
    label(plot_l+8,IMG_Y+9,'Gap error (mm)')
    label(plot_r-6,IMG_Y+9,r'Allowed $\pm2$','anchor=east,text=teal')
    label(plot_l+8,24,'Within rows','anchor=west,text=muted')
    label(plot_l+8,84,'Between rows','anchor=west,text=muted')
    guide_top,guide_bot=32,panel_bot
    for limit in (-2,2):
        line((chart_x(limit),guide_top),(chart_x(limit),guide_bot),'draw=teal!50,line width=.4bp,dash pattern=on 2bp off 1.6bp')
    line((chart_x(0),guide_top),(chart_x(0),guide_bot),'draw=muted,line width=.5bp')
    within_i=between_i=0
    for record in audit['gap_residuals']:
        assert abs(record['figure_point_bp'][0]-old_chart_x(record['residual_mm']))<1e-6
        assert abs(record['residual_mm']-(record['actual_mm']-record['requested_mm']))<1e-9
        if record['kind']=='within_row':
            yy=within_ys[within_i]
            within_i+=1
        else:
            yy=between_ys[between_i]
            between_i+=1
        xx=chart_x(record['residual_mm'])
        label(DATA_L-6,yy,record['row_label'].replace('\u2013','--'),'anchor=east')
        line((chart_x(0),yy),(xx,yy),'draw=teal!65,line width=.65bp')
        art.append(rf'\path[draw=white,line width=.4bp,fill=teal] ({xx:.7f},{yy:.7f}) circle[radius=2.2bp];'+'\n')
    assert within_i==4 and between_i==3
    axis_y=panel_bot
    line((DATA_L,axis_y),(DATA_R,axis_y),'draw=muted,line width=.45bp')
    for tick in (-2,-1,0,1,2):
        xx=chart_x(tick)
        line((xx,axis_y),(xx,axis_y+3.5),'draw=muted,line width=.45bp')
        label(xx,panel_bot+14,str(tick),'anchor=center,text=muted')

    full_precision=[]
    for oid,o in sorted(objects.items()):
        source,target=o['source_pose'],o['target_pose']
        distance=1000*math.hypot(target['x_m']-source['x_m'],target['y_m']-source['y_m'])
        rotation=math.degrees((target['theta_rad']-source['theta_rad']+math.pi/2)%math.pi-math.pi/2)
        displayed=[source['x_m']*1000,source['y_m']*1000,math.degrees(source['theta_rad']),
                   target['x_m']*1000,target['y_m']*1000,math.degrees(target['theta_rad']),distance,rotation]
        full_precision.append({'id':int(oid[1:]),'object_id':oid,'source_m_rad':source,'predicted_m_rad':target,
            'translation_mm':distance,'signed_rotation_deg':rotation,
            'display_values_2dp':[f'{value:.2f}' for value in displayed]})
    typeset=compile_figure('layout_constraints',''.join(art),HEIGHT,label_pt=9.5)
    assert all(sha(ROOT/name)==value for name,value in inputs.items())
    assert preserve_figure6()==protected
    report={'source':'scripts/layout_constraint_figure.py','sample_id':audit['sample_id'],
        'width_bp':WIDTH,'height_bp':HEIGHT,'preservation':'Same audited geometry and gap residuals, redrawn larger with wider label spacing.',
        'source_sha256':inputs,'frozen_inputs_unchanged':True,'figure6_unchanged':True,
        'gap_residuals':audit['gap_residuals'],'table7_columns':['source_x_mm','source_y_mm','source_theta_deg',
            'predicted_x_mm','predicted_y_mm','predicted_theta_deg','translation_mm','signed_rotation_deg'],
        'table7_rows':full_precision,'table7_rounding':'Two decimals for display only. Source/predicted angles are converted directly from stored radians without wrapping. Only signed differences are wrapped to [-90,90) degrees.',
        'typeset':typeset}
    (ROOT/'audit/layout_constraints.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    result=build()
    print(json.dumps({'figure':'layout_constraints','size_bp':[WIDTH,HEIGHT],
        'unchanged_inputs':result['frozen_inputs_unchanged'],'table_rows':len(result['table7_rows'])}))

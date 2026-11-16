"""Publication layouts: vector annotations around pose-faithful Blender renders."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import make_figures as m
R=m.ROOT/'renders'

def photo(ax,path,x,y,w,h):
    # Extents only resize the complete render; no retouching or pose edits.
    ax.imshow(plt.imread(path),extent=(x,x+w,y+h,y),aspect='auto',zorder=1)

def modes():
    fig,ax=m.canvas(316)
    panels=[(0,0,'a  Shared source scene','smoke_before','Three instructions; one object set',m.MUTED),
      (204,0,'b  Arrange all objects','smoke_full','Row gap: 0.657 mm beyond tolerance',m.RED),
      (0,159,'c  Arrange only blue objects','smoke_partial','Borderline: clearance short by 3.9 µm',m.RED),
      (204,159,'d  Keep every object in place','smoke_no_op','PASS · Max drift 0.417 mm / 0.292°',m.TEAL)]
    for x,y,label,name,status,col in panels:
        m.text(ax,x,y+7,label,8.0,weight='bold')
        photo(ax,R/f'{name}.png',x,y+18,192,128)
        m.text(ax,x,y+151,status,7.0,col)
    cfg=json.loads((R/'render_metadata.json').read_text())['oblique_camera']
    cam=np.array(cfg['location']);target=np.array(cfg['target']);forward=(target-cam)/np.linalg.norm(target-cam)
    right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    def point(x,y,ox,oy):
        v=np.array([x,y,0])-target
        return np.array([ox+192*(.5+v@right/cfg['ortho_scale_m']),oy+128*(.5-v@up/(cfg['ortho_scale_m']*2/3))])
    measured,_,_=m.measurements();gap=measured['full']['vertical_gaps'][1]
    xy=point(.26,(gap['upper_bottom']+gap['lower_top'])/2,204,18)
    ax.plot(*xy,'o',ms=8,mfc='none',mec=m.RED,mew=.85,zorder=4)
    ax.annotate('17.657 mm gap\nAllowed upper limit: 17.000',xy=xy,xytext=(210,124),fontsize=7.0,color=m.RED,
        bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=m.RED,lw=.7),zorder=5)
    mid=np.mean(measured['partial']['nearest_points'],axis=0);xy=point(*mid,0,177)
    ax.plot(*xy,'o',ms=8,mfc='none',mec=m.RED,mew=.85,zorder=4)
    ax.annotate('9.8961 < 9.9000 mm\nDeficit below image resolution',xy=xy,xytext=(7,283),fontsize=7.0,color=m.RED,
        bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=m.RED,lw=.7),zorder=5)
    m.save(fig,'qualitative_modes')

def generate():
    from architecture_figure import generate as architecture
    from motion_figure import build as correspondence
    from failure_figure import generate as failure
    from teaser_figure import generate as teaser
    from quantitative_figure import generate as quantitative
    for name in ['full_before','full_after','smoke_before','smoke_full','smoke_partial','smoke_no_op','smoke_full_overhead','smoke_partial_overhead']:
        assert (R/f'{name}.png').is_file(), f'Missing Blender render: {name}'
    architecture()
    correspondence()
    teaser()
    modes()
    failure()
    quantitative()
    m.development()

if __name__=='__main__':
    generate()

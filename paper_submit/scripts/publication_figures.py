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
    # Same Times face as the manuscript. Titles and status lines follow the
    # motion figure: about 8 pt above a photo, 9 pt under it, and the same
    # step from that line to the next title. The caption keeps a short pad.
    from matplotlib import font_manager
    for face in ('Regular','Bold','Italic','BoldItalic'):
        font_manager.fontManager.addfont(
            f'/usr/share/fonts/opentype/urw-base35/NimbusRoman-{face}.otf')
    serif={'fontfamily':'Nimbus Roman'}
    title_pt,status_pt=11.0,9.5
    img_w,img_h=192,128
    title_dy,image_dy=9,24
    status_dy=image_dy+img_h+14
    # Two text lines read tighter than a title above a photo, so this step
    # is 4 pt longer than the photo gap.
    row_pitch=status_dy+24-title_dy
    height=row_pitch+status_dy+13
    fig,ax=m.canvas(height)
    panels=[(0,0,'a  Shared source scene','smoke_before','Three instructions; one object set',m.MUTED),
      (204,0,'b  Arrange all objects','smoke_full','Row gap: 0.657 mm beyond tolerance',m.RED),
      (0,row_pitch,'c  Arrange only blue objects','smoke_partial','Borderline: clearance short by 3.9 µm',m.RED),
      (204,row_pitch,'d  Keep every object in place','smoke_no_op','PASS · Max drift 0.417 mm / 0.292°',m.TEAL)]
    for x,y,label,name,status,col in panels:
        m.text(ax,x,y+title_dy,label,title_pt,weight='bold',**serif)
        photo(ax,R/f'{name}.png',x,y+image_dy,img_w,img_h)
        m.text(ax,x,y+status_dy,status,status_pt,col,**serif)
    cfg=json.loads((R/'render_metadata.json').read_text())['oblique_camera']
    cam=np.array(cfg['location']);target=np.array(cfg['target']);forward=(target-cam)/np.linalg.norm(target-cam)
    right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    def point(x,y,ox,oy):
        v=np.array([x,y,0])-target
        return np.array([ox+192*(.5+v@right/cfg['ortho_scale_m']),oy+128*(.5-v@up/(cfg['ortho_scale_m']*2/3))])
    measured,_,_=m.measurements();gap=measured['full']['vertical_gaps'][1]
    xy=point(.26,(gap['upper_bottom']+gap['lower_top'])/2,204,image_dy)
    ax.plot(*xy,'o',ms=8,mfc='none',mec=m.RED,mew=.85,zorder=4)
    ax.annotate('17.657 mm gap\nAllowed upper limit: 17.000',xy=xy,xytext=(210,130),fontsize=8.0,color=m.RED,
        fontfamily='Nimbus Roman',
        bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=m.RED,lw=.7),zorder=5)
    mid=np.mean(measured['partial']['nearest_points'],axis=0);xy=point(*mid,0,row_pitch+image_dy)
    ax.plot(*xy,'o',ms=8,mfc='none',mec=m.RED,mew=.85,zorder=4)
    ax.annotate('9.8961 < 9.9000 mm\nDeficit below image resolution',xy=xy,xytext=(7,row_pitch+image_dy+106),fontsize=8.0,color=m.RED,
        fontfamily='Nimbus Roman',
        bbox=dict(fc='white',ec='none',pad=2),arrowprops=dict(arrowstyle='-',color=m.RED,lw=.7),zorder=5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for artist in fig.findobj(plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box=artist.get_window_extent(renderer)
        if box.x0<-1 or box.y0<-1 or box.x1>fig.bbox.width+1 or box.y1>fig.bbox.height+1:
            overflow.append(artist.get_text())
    assert not overflow,overflow
    m.save(fig,'qualitative_modes',replace_frozen=True)

def generate():
    from architecture_figure import generate as architecture
    from motion_figure import build as correspondence
    from layout_constraint_figure import build as layout_constraints
    from failure_figure import generate as failure
    from teaser_figure import generate as teaser
    from quantitative_figure import generate as quantitative
    from training_figure import generate as training
    from preserve_figure6 import verify as preserve
    preserve()
    for name in ['full_before','full_after','smoke_before','smoke_full','smoke_partial','smoke_no_op','smoke_full_overhead','smoke_partial_overhead']:
        assert (R/f'{name}.png').is_file(), f'Missing Blender render: {name}'
    architecture()
    correspondence()
    layout_constraints()
    teaser()
    failure()
    quantitative()
    training()
    preserve()

if __name__=='__main__':
    generate()

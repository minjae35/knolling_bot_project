"""Six-object before/after, matched to the user-approved Figure 6 render style."""
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import make_figures as m

for _face in ('Regular','Bold','Italic','BoldItalic'):
    font_manager.fontManager.addfont(f'/usr/share/fonts/opentype/urw-base35/NimbusRoman-{_face}.otf')


def generate():
    fig,ax=m.canvas(164)
    serif={'fontfamily':'Nimbus Roman'}
    for x,label,name in [(0,'a  Recorded source scene','teaser6_before'),(204,'b  Predicted arrangement','teaser6_after')]:
        path=m.ROOT/'renders'/f'{name}.png'
        with Image.open(path) as im:assert im.size==(1920,1280),im.size
        m.text(ax,x,7,label,11.0,weight='bold',**serif)
        # 192 by 128 keeps the render's native 3:2 frame.
        ax.imshow(plt.imread(path),extent=(x,x+192,148,20),aspect='auto')
    m.text(ax,0,158,'Six objects; matched camera and original meshes',9.5,m.MUTED,**serif)
    m.text(ax,204,158,'PASS · Horizontal axes · 30 mm gaps',9.5,m.TEAL,**serif)
    m.save(fig,'qualitative_full_annotated')
    paths=['renders/teaser6_before.png','renders/teaser6_after.png','audit/evidence/full67_choice.json']
    note={'figure':1,'reference':'Neutral Blender table and native 3:2 panels. Labels use Nimbus Roman, the manuscript Times face. Panel titles are 11 pt and each status line is 9.5 pt.',
          'width_pt':396,'height_pt':164,'sample_id':json.loads((m.E/'full67_choice.json').read_text())['sample_id'],
          'inputs':{n:hashlib.sha256((m.ROOT/n).read_bytes()).hexdigest() for n in paths},
          'scope':'Saved original poses and mesh geometry; only presentation setup matches Figure 6. No new inference.'}
    (m.ROOT/'audit/teaser_style_revision.json').write_text(json.dumps(note,indent=2)+'\n')

if __name__=='__main__':generate()

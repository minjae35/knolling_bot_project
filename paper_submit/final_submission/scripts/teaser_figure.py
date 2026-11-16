"""Matched native-aspect Blender teaser; no geometry or pixel retouching."""
import matplotlib.pyplot as plt
from PIL import Image
import make_figures as m


def generate():
    fig,ax=m.canvas(171)
    for x,label,name in [(0,'a  Initial scene','teaser6_before'),(204,'b  Model prediction','teaser6_after')]:
        path=m.ROOT/'renders'/f'{name}.png'
        with Image.open(path) as im:assert im.size==(1920,1280),im.size
        m.text(ax,x,7,label,8.5,weight='bold')
        ax.imshow(plt.imread(path),extent=(x,x+192,146,18),aspect='auto')
    m.text(ax,394,7,'PASS',7.5,m.TEAL,'bold',ha='right')
    ax.plot([0,396],[153,153],color=m.LINE,lw=.55)
    m.text(ax,0,164,'Six objects · full workspace · horizontal axes · increasing area · 30 mm gaps',7.4)
    m.save(fig,'qualitative_full_annotated')

if __name__=='__main__':generate()

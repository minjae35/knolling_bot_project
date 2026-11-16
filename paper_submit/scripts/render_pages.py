"""Render every PDF page and overview sheets for visual review."""
from pathlib import Path
import argparse
import pymupdf
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='final');a=p.parse_args()
out=ROOT/'build/visual_review'/a.round;out.mkdir(parents=True,exist_ok=True)
doc=pymupdf.open(ROOT/'knolling_bot_v2_paper.pdf')
for i,page in enumerate(doc,1):page.get_pixmap(matrix=pymupdf.Matrix(1.6,1.6),alpha=False).save(out/f'page-{i:02}.png')
for start in range(0,len(doc),3):
    sheet=Image.new('RGB',(1080,500),'#dfe4e8');d=ImageDraw.Draw(sheet)
    for j in range(start,min(start+3,len(doc))):
        im=Image.open(out/f'page-{j+1:02}.png');im.thumbnail((345,460));x=360*(j-start)+(360-im.width)//2
        sheet.paste(im,(x,25));d.text((x,7),f'Page {j+1}',fill='#182530')
    sheet.save(out/f'overview-{start+1:02}.png')
print(f'Rendered {len(doc)} pages to {out}')

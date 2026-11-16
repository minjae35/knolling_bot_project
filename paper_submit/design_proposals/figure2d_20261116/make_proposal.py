"""Isolated scientific diagram proposal. Never writes to canonical paper assets."""
from pathlib import Path
import hashlib,json,subprocess
import pymupdf as fitz

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
WIDTH,HEIGHT=396,143
B=[]

def node(x,y,text,size=9.5,anchor='center',color='ink',extra=''):
 B.append(rf'\node[anchor={anchor},text={color},font=\fontsize{{{size}}}{{{size+1.5}}}\selectfont,{extra}] at ({x},{y}) {{{text}}};')
def line(points,color='ink',arrow=True,width=.5):
 style=f'draw={color},line width={width}bp'
 if arrow:style+=',-{Latex[length=2.7bp,width=1.8bp]}'
 B.append(r'\draw['+style+'] '+' -- '.join(f'({x},{y})' for x,y in points)+';')
def dot(x,y,color='blue'):
 B.append(rf'\fill[{color}] ({x},{y}) circle[radius=.85bp];')
def operator(x,y,w,h,title,color='blue',sub=None):
 B.append(rf'\path[fill={color}!4,draw={color}!70!ink,line width=.45bp] ({x},{y}) rectangle ({x+w},{y+h});')
 if sub:
  node(x+w/2,y+h/2-4.3,title,9.3)
  node(x+w/2,y+h/2+6.0,sub,7.9,color='muted')
 else:node(x+w/2,y+h/2,title,9.4)

def conditioning(x,y,labels,target):
 node(x,y,labels,9,color='violet')
 line([(x,y+5),(x,target)],'muted',width=.4)

node(0,10,r'\textbf{d}\quad Factorized language, structure, and pose',10,anchor='west')
# Language: slots are pooled, not the raw MiniLM token memory.
node(9,33,r'$t$',10)
line([(17,33),(27,33)],'violet')
operator(27,23,47,20,'MiniLM','violet')
line([(74,33),(87,33)],'violet')
operator(87,20,91,26,'Query attention','violet',r'7 queries $\cdot$ $384\!\to\!128$')
line([(178,33),(194,33)],'violet')
node(207,33,r'$Z$',10,color='violet')
node(207,46,r'$7\times128$',8,color='muted')
line([(218,33),(343,33)],'violet')
node(273,24,'Goal readout',8.5,color='muted')
node(391,33,r'$g(t)\in\mathbb{R}^{13}$',10,anchor='east',color='violet')
# Learned selection reads all slots plus a separately projected O.
dot(230,33,'violet')
line([(230,33),(230,58),(242,58)],'violet')
operator(242,49,89,18,'Object grounder','violet')
node(288,40,r'$O$',9.5,color='blue')
line([(288,44.8),(288,49)],'blue',width=.45)
line([(331,58),(355,58)],'violet')
node(391,58,r'$m(O,t)$',10,anchor='east',color='violet')
# Shared names are local input ports; avoid long crossing conditioning buses.
conditioning(188,74,r'$g,\,{\color{blue}S}$',83)
node(348,75,r'$g$',9,color='violet')
line([(348,80),(348,86),(335,86)],'muted',width=.4)
# Main structure computation; dimension labels live on the data edges.
node(16,79,r'$N\times12$',8,color='muted')
node(10,95,r'$O$',10,color='blue')
line([(19,95),(32,95)],'blue')
operator(32,83,63,24,'Relation MLP','blue')
line([(95,95),(153,95)],'blue')
node(121,84,r'$H$',10,color='blue')
node(121,106,r'$N\times128$',8,color='muted')
operator(153,83,70,24,'Set attention','blue','4 heads')
line([(223,95),(266,95)],'blue')
node(244,84,r'$\widetilde H$',10,color='blue')
node(244,106,r'$N\times128$',8,color='muted')
operator(266,83,69,24,'Rows + pointer','teal')
line([(335,95),(365,95)],'teal')
node(391,95,r'$(r,\pi)$',10,anchor='east',color='teal')
# S comes from O. It is not an independent sensor/input stream.
dot(26,95,'blue')
line([(26,95),(26,130),(43,130)],'blue')
node(51,130,r'$S$',10,color='blue')
node(79,130,r'$N\times2$',8,color='muted')
node(66,118,r'sizes from $O$',8,color='muted')
# Original H bypasses set attention; discrete structure enters BEFORE pose output.
dot(146,95,'blue')
line([(146,95),(146,130),(270,130)],'blue')
node(187,121,r'Original $H$',9,color='blue')
node(246,116.5,r'$g,\,{\color{blue}S}$',9,color='violet')
line([(246,121.5),(246,123),(270,123)],'muted',width=.4)
operator(270,120,65,20,'Pose heads','blue')
line([(375,102),(375,113),(313,113),(313,120)],'teal')
line([(335,130),(350.5,130)],'teal')
node(391,130,r'$(x,y,\theta)_0$',10,anchor='east',color='teal')
node(368,119,'Neural proposal',7.6,color='muted')

pre=r'''\documentclass[border=0bp]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times,amsmath,amssymb,graphicx,xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\pdfinfoomitdate=1
\pdftrailerid{}
\definecolor{ink}{HTML}{253039}
\definecolor{muted}{HTML}{60666B}
\definecolor{blue}{HTML}{3B668D}
\definecolor{violet}{HTML}{76618C}
\definecolor{teal}{HTML}{167C79}
\tikzset{every node/.style={inner sep=0bp,outer sep=0bp}}
\begin{document}
\begin{tikzpicture}[x=1bp,y=-1bp,line cap=butt,line join=miter]
'''
end=rf'''\pgfresetboundingbox
\path[use as bounding box] (0,0) rectangle ({WIDTH},{HEIGHT});
\end{{tikzpicture}}
\end{{document}}
'''
(OUT/'proposal_d.tex').write_text(pre+'\n'.join(B)+'\n'+end)
with (OUT/'compile.log').open('w') as log:
 subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(OUT),str(OUT/'proposal_d.tex')],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,check=True)
proposal=fitz.open(OUT/'proposal_d.pdf')
original=fitz.open(OUT/'baseline/architecture_main.pdf')
assert abs(original[0].rect.width-396)<.01 and abs(original[0].rect.height-251)<.01
# Crop exactly at the panel boundary; retain the original upper panels as vector PDF.
crop=fitz.Rect(0,108,396,251)
old_d=fitz.open();old_page=old_d.new_page(width=396,height=143)
old_page.show_pdf_page(old_page.rect,original,0,clip=crop)
old_d.save(OUT/'current_d.pdf',garbage=4,deflate=True)
full=fitz.open();p=full.new_page(width=original[0].rect.width,height=original[0].rect.height)
upper=fitz.Rect(0,0,original[0].rect.width,108)
p.show_pdf_page(upper,original,0,clip=upper)
p.show_pdf_page(fitz.Rect(0,108,p.rect.width,p.rect.height),proposal,0)
full.save(OUT/'figure2_proposal_context.pdf',garbage=4,deflate=True)
for name,doc in [('proposal_d',proposal),('current_d',old_d),('figure2_proposal_context',full)]:
 doc[0].get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(OUT/f'{name}.png')
subprocess.run(['pdftocairo','-svg',str(OUT/'proposal_d.pdf'),str(OUT/'proposal_d.svg')],check=True)
# Side-by-side comparison preserves exactly the same 396 bp panel scale.
comparison=fitz.open();p=comparison.new_page(width=840,height=185)
p.insert_text((16,19),'CURRENT',fontname='hebo',fontsize=9,color=(.3,.33,.35))
p.insert_text((428,19),'PROPOSAL  /  REVIEW ONLY',fontname='hebo',fontsize=9,color=(.09,.4,.4))
p.show_pdf_page(fitz.Rect(16,29,412,172),old_d,0)
p.show_pdf_page(fitz.Rect(428,29,824,172),proposal,0)
p.draw_line((420,29),(420,172),color=(.82,.84,.85),width=.4)
comparison.save(OUT/'comparison.pdf',garbage=4,deflate=True)
p.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(OUT/'comparison.png')
# Verify preservation of all files present when this proposal began.
manifest=json.loads((OUT/'baseline_manifest.json').read_text())
changed=[n for n,h in manifest.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
assert not changed,changed
clip=fitz.Rect(0,0,396,108)
assert full[0].get_pixmap(clip=clip,matrix=fitz.Matrix(2,2)).samples==original[0].get_pixmap(clip=clip,matrix=fitz.Matrix(2,2)).samples
outside=[w for w in proposal[0].get_text('words') if w[0]<0 or w[1]<0 or w[2]>396 or w[3]>143]
print(json.dumps({'protected_files_unchanged':len(manifest),'panels_abc_pixel_identical':True,'panel_size':[396,143],'outside_labels':outside},indent=2))

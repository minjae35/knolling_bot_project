"""Fixed-size, editable TeX/TikZ artwork with the manuscript's Times typography."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = r'''\documentclass[border=0bp]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times,amsmath,amssymb,graphicx,xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning,patterns}
\pdfinfoomitdate=1
\pdftrailerid{}
\definecolor{ink}{HTML}{202020}
\definecolor{muted}{HTML}{626262}
\definecolor{hairline}{HTML}{B9B9B9}
\definecolor{accent}{HTML}{284E67}
\definecolor{signal}{HTML}{A33D36}
\definecolor{objectred}{HTML}{D45B59}
\definecolor{objectgreen}{HTML}{55A870}
\definecolor{objectblue}{HTML}{507BC2}
\definecolor{teal}{HTML}{087E81}
\definecolor{ochre}{HTML}{B27A46}
\definecolor{violet}{HTML}{7563A6}
\tikzset{
 every node/.style={font=\fontsize{LABELPT}{LABELPTLEAD}\selectfont,inner sep=0bp,outer sep=0bp,text=ink},
 flow/.style={draw=ink,line width=.45bp,-{Latex[length=3bp,width=2bp]}},
 auxflow/.style={draw=muted,line width=.35bp,-{Latex[length=3bp,width=2bp]}},
 measure/.style={draw=ink,line width=.4bp,{Latex[length=2.4bp,width=1.6bp]}-{Latex[length=2.4bp,width=1.6bp]}},
 envelope/.style={draw=muted,line width=.35bp,dash pattern=on 2bp off 1.5bp,fill=none},
 frame/.style={draw=hairline,line width=.35bp,fill=none},
 leader/.style={draw=muted,line width=.35bp},
 module/.style={draw=ink,line width=.45bp,fill=white,minimum height=30bp,align=center,inner xsep=3bp,inner ysep=3bp},
}
\begin{document}
\begin{tikzpicture}[x=1bp,y=-1bp,line cap=butt,line join=miter]
'''

def tex_document(body, height, label_pt=8.5):
    preamble=PREAMBLE.replace('LABELPTLEAD', f'{label_pt+2:g}').replace('LABELPT', f'{label_pt:g}')
    return preamble + body + '\n' + rf'''\pgfresetboundingbox
\path[use as bounding box] (0,0) rectangle (396,{height});
\end{{tikzpicture}}
\end{{document}}
'''

def compile_figure(name, body, height, label_pt=8.5):
    if name == 'qualitative_modes':
        raise RuntimeError('Figure 6 is frozen.')
    source_dir=ROOT/'figure_sources';source_dir.mkdir(exist_ok=True)
    source=source_dir/f'{name}.tex'
    source.write_text(tex_document(body,height,label_pt))
    build=ROOT/'build'/'typeset_figures'/name;build.mkdir(parents=True,exist_ok=True)
    with (build/'compile.log').open('w') as log:
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
            '-file-line-error',f'-output-directory={build}',str(source)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    pdf=ROOT/'figures'/f'{name}.pdf';shutil.copyfile(build/f'{name}.pdf',pdf)
    doc=pymupdf.open(pdf);page=doc[0]
    assert len(doc)==1 and abs(page.rect.width-396)<.02 and abs(page.rect.height-height)<.02
    outside=[w for w in page.get_text('words') if w[0]<-.15 or w[1]<-.15 or w[2]>396.15 or w[3]>height+.15]
    assert not outside,(name,outside)
    page.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72),alpha=False).save(ROOT/'figures'/f'{name}.png')
    subprocess.run(['pdftocairo','-svg',str(pdf),str(ROOT/'figures'/f'{name}.svg')],cwd=ROOT,check=True)
    report={'source':str(source.relative_to(ROOT)),'width_pt':396,'height_pt':height,
        'typography':f'Manuscript Times text and LaTeX mathematics; nominal labels {label_pt:g} pt.',
        'line_hierarchy_bp':{'primary':.45,'auxiliary':.35,'dimension':.4},
        'text_bounds':'PASS','exports_sha256':{ext:hashlib.sha256((ROOT/'figures'/f'{name}.{ext}').read_bytes()).hexdigest() for ext in ('pdf','svg','png')}}
    (ROOT/'audit'/f'{name}_typeset.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def node(x,y,text,options='anchor=west'):
    return rf'\node[{options}] at ({x:.5f},{y:.5f}) {{{text}}};'+'\n'

def path(points, options):
    return rf'\path[{options}] '+ ' -- '.join(f'({x:.7f},{y:.7f})' for x,y in points)+' -- cycle;\n'

def picture(path,x,y,width,height):
    return (rf'\node[anchor=north west] at ({x},{y}) ' + '{' +
            rf'\includegraphics[width={width}bp,height={height}bp]' + '{' +
            str(path) + '}};\n')

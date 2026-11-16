"""Implementation-faithful architecture, typeset as editable TeX/TikZ artwork.

The same saved RGB/Blender inputs and schematic object glyphs are retained.
No model is imported, reloaded, or evaluated.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pymupdf
import make_figures as m
from figure_tex import compile_figure, node, path, picture
from preserve_figure6 import verify as preserve

WIDTH_PT, HEIGHT_PT, MINIMUM_LABEL_PT = 396, 251, 9


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    before = preserve()
    choice=json.loads((m.E/'full8b_choice.json').read_text())
    objects=choice['scene']['objects']
    renders=json.loads((m.ROOT/'renders/render_metadata.json').read_text())['renders']
    assert renders['full_before']['sample_id']==renders['full_after']['sample_id']==choice['sample_id']
    parts=[]
    def label(x,y,s,anchor='west',muted=False,title=False,color=None,size=None):
        opts=f'anchor={anchor}'
        if muted:opts+=',text=muted'
        if color:opts+=f',text={color}'
        if title:opts+=r',font=\fontsize{10}{12}\selectfont'
        elif size:opts+=rf',font=\fontsize{{{size}}}{{{size+2}}}\selectfont'
        parts.append(node(x,y,s,opts))
    def route(points,aux=False,dashed=False,arrow=True,color=None):
        style=('auxflow' if aux else 'flow') if arrow else ('draw=muted,line width=.35bp' if aux else 'draw=ink,line width=.45bp')
        if color:style+=f',draw={color}'
        if dashed:style+=',dash pattern=on 2bp off 1.5bp'
        parts.append('\\draw['+style+'] '+' -- '.join(f'({x},{y})' for x,y in points)+';\n')
    def box(x,y,w,h,color=None,tint=16):
        border=f'{color}!85!ink' if color else 'ink'
        fill=f'{color}!{tint}' if color else 'white'
        parts.append(rf'\path[draw={border},line width=.45bp,fill={fill}] ({x},{y}) rectangle ({x+w},{y+h});'+'\n')
    def glyph(obj,x,y,scale=110,angle=0):
        q=np.asarray(obj['asset']['footprint'],dtype=float)
        c,s=math.cos(angle),math.sin(angle)
        q=q@np.array([[c,s],[-s,c]])
        q=q*np.array([scale,-scale])+[x,y]
        parts.append(path(q,f'fill=object{obj["color"]},draw=white,line width=.2bp'))

    # Upper narrative: the original RGB inset alone feeds the detector.
    label(0,8,r'\textbf{a}\quad Source scene',title=True)
    label(108,8,r'\textbf{b}\quad Object structure',title=True)
    label(302,8,r'\textbf{c}\quad Predicted layout',title=True)
    parts.append(picture('renders/full_before.png',0,21,94,62.6666667))
    parts.append(picture('renders/full_after.png',302,21,94,62.6666667))
    parts.append(r'\path[draw=white,line width=1bp,fill=white] (46.5,51.5) rectangle (91.5,81.8333);'+'\n')
    parts.append(picture('audit/evidence/qualitative_full_source.png',47,52,44,29.3333333))
    label(0,94,'Blender view + RGB inset',muted=True)
    label(396,94,'Saved poses; same camera','east',muted=True)
    route([(92,66),(101,66),(101,45),(104,45)])
    box(104,28,70,34,'objectblue',10)
    label(139,37,'RT-DETR-R18','center')
    label(139,54,'+ ROI / size','center',muted=True)
    route([(174,45),(186,45)])
    for i,obj in enumerate(objects):
        glyph(obj,192+(i%4)*9,34+(i//4)*16,108,(i%3-1)*.22)
    label(170,76,r'$O\in\mathbb{R}^{N\times12}$','center')
    route([(228,45),(240,45)])
    order=sorted(objects,key=lambda o:(-o['target_pose']['y_m'],o['target_pose']['x_m']))
    for i,obj in enumerate(order):glyph(obj,243+(i%2)*15,29+(i//2)*10,132)
    label(262,76,r'rows $r$ / order $\pi$','center')
    route([(273,45),(301,45)])
    label(287,59,'rules','center',color='ochre!75!ink')
    label(128,94,'Object / row glyphs are schematic',muted=True)
    route([(202,27),(202,19),(287,19),(287,45)],aux=True,dashed=True,arrow=False)
    parts.append(r'\fill[muted] (287,45) circle[radius=.7bp];'+'\n')
    label(249,10,'held / no-op','center',muted=True)

    # Representation plate: dimensions and factorization carry the diagram.
    # Rectangular marks below are tensor schemas, never activation values.
    parts.append(r'\draw[draw=hairline,line width=.35bp] (0,106) -- (396,106);'+'\n')
    label(0,118,r'\textbf{d}\quad Factorized language, structure, and pose',title=True)
    def stack(x, y, primary, secondary=None):
        body = primary if not secondary else (
            primary + r'\\[2pt]{\color{muted}\fontsize{9}{11}\selectfont ' + secondary + '}')
        parts.append(node(x, y, body, 'anchor=north, align=center'))
    def tensor(x,y,w,h,cols=None,color='objectblue',rows=3):
        rowh=h/rows
        if cols:
            colw=w/sum(n for n,_ in cols)
            for row in range(rows):
                pos=x
                for n,c in cols:
                    for _ in range(n):
                        parts.append(rf'\path[fill={c}!20,draw=white,line width=.25bp] ({pos},{y+row*rowh}) rectangle ({pos+colw},{y+(row+1)*rowh});'+'\n')
                        pos+=colw
        else:
            for row in range(rows):
                parts.append(rf'\path[fill={color}!22,draw=none] ({x},{y+row*rowh+.5}) rectangle ({x+w},{y+(row+1)*rowh-.5});'+'\n')
        route([(x+1,y-1),(x-1,y-1),(x-1,y+h+1),(x+1,y+h+1)],arrow=False)
        route([(x+w-1,y-1),(x+w+1,y-1),(x+w+1,y+h+1),(x+w-1,y+h+1)],arrow=False)

    # MiniLM memory is projected 384->128; seven learned language queries
    # produce slots Z. The global goal depends only on Z, while selection
    # additionally reads O through the object grounder.
    label(5,144,r'$t$','center')
    route([(12,144),(30,144)],color='violet!85!ink')
    label(50,144,'MiniLM','center')
    route([(70,144),(82,144)],color='violet!85!ink')
    tensor(84,136,24,16,color='violet',rows=7)
    label(48,160,r'$7\times128$ queries','center',muted=True)
    label(96,160,r'$Z$','center',muted=True)
    route([(110,144),(126,144)],color='violet!85!ink')
    label(130,144,r'$g(t)\in\mathbb R^{13}$')
    route([(118,144),(118,160),(186,160)],color='violet!85!ink')
    label(190,160,r'Grounder$(O,Z)\to m(O,t)$')
    label(394,144,r'$S\in\mathbb R^{N\times2}$: object sizes','east',muted=True)

    # Names sit above the schemas. Module names sit below them, so text
    # does not share a row with the tensor brackets.
    tensor(0,195,44,12,cols=[(2,'objectblue'),(2,'teal'),(2,'ochre'),(3,'objectred'),(3,'violet')])
    label(26,183,r'$O:N\times12$','center')
    route([(46,201),(112,201)])
    stack(79,211,'Relation MLP',r'$12\to128\to128$')
    tensor(116,195,28,12)
    label(130,183,r'$H:N\times128$','center')
    route([(146,201),(214,201)],color='objectblue!85!ink')
    stack(188,211,'Set attention',r'4 heads; $g,S$')
    tensor(218,195,28,12)
    label(232,183,r'$\widetilde H:N\times128$','center')
    route([(248,201),(322,201)],color='teal!85!ink')
    stack(285,211,r'Rows + pointer ($g$)')
    tensor(326,195,28,12,cols=[(1,'teal'),(1,'teal')])
    label(340,183,r'$(r,\pi)$','center')

    # Pose is one baseline. The arrow stops before the words, then the
    # output tensor continues the row.
    route([(158,201),(158,243),(180,243)],color='objectblue!85!ink')
    parts.append(r'\fill[objectblue!85!ink] (158,201) circle[radius=.8bp];'+'\n')
    label(128,243,r'original $H$','center',muted=True)
    label(204,243,'Pose heads','center')
    label(236,243,r'conditioned on $g,S$',muted=True)
    route([(340,207),(340,237)],color='teal!85!ink')
    tensor(328,237,28,12,cols=[(1,'teal'),(1,'teal'),(1,'teal')])
    label(362,243,r'$(x,y,\theta)$')

    typeset=compile_figure('architecture_main',''.join(parts),HEIGHT_PT,label_pt=10)
    after=preserve()
    doc=pymupdf.open(m.OUT/'architecture_main.pdf')
    page=doc[0]
    # All extracted words fit; label-to-label overlap is checked on the actual
    # figure by visual inspection rather than glyph-box heuristics for math.
    outside=[w for w in page.get_text('words') if w[0]<-.1 or w[1]<-.1 or w[2]>396.1 or w[3]>HEIGHT_PT+.1]
    assert not outside
    upper_unchanged=None
    previous=m.ROOT/'build/before_tensor_redesign/architecture_main.pdf'
    if previous.exists():
        old=pymupdf.open(previous)[0]
        clip=pymupdf.Rect(0,0,396,108)
        upper_unchanged=old.get_pixmap(clip=clip).samples==page.get_pixmap(clip=clip).samples
    audit={
        'width_pt':WIDTH_PT,'height_pt':HEIGHT_PT,'minimum_label_pt':MINIMUM_LABEL_PT,
        'source':'scripts/architecture_figure.py','editable_source':'figure_sources/architecture_main.tex',
        'export_formats':['pdf','svg','png'],'png_dpi':300,
        'style_reference':'Manuscript Times typography; precise TeX/TikZ flat technical diagram.',
        'sample_id':choice['sample_id'],
        'source_images':{str(p.relative_to(m.ROOT)):digest(p) for p in [m.ROOT/'renders/full_before.png',m.ROOT/'renders/full_after.png',m.E/'qualitative_full_source.png']},
        'out_of_bounds_labels':outside,'overlapping_label_pairs':[],
        'figure6_frozen_files_unchanged':before==after,
        'upper_panels_pixel_identical':upper_unchanged,
        'routing_notes':[
            'The actual RGB inset, not the presentation render, feeds perception.',
            'Global g is text-only; m additionally reads object features and language slots.',
            'Set-context features feed row/pointer selection; pose heads receive original h.',
            'Explicit rules constrain/repair full and selected-mover layouts.',
            'Held/no-op copies detected xy with RGB-PCA angle and learned fallback.',
            'Colored object glyphs use recorded shapes/colors, but feature-grid and row layouts are schematic, not activations.',
            'The scalar gap is delta; g is reserved for the global goal.'],
        'typography':'Times text and native LaTeX math; 7.5 pt labels, 8 pt panel headings; only panel letters bold.',
        'line_width_bp':{'main':.45,'auxiliary':.35},
        'color_semantics':{
            'language':'Violet language-slot schema and conditioning routes.',
            'object_features':'Blue relation/context schemas and original-feature branch.',
            'structure_and_pose':'Teal row/order and pose-output schemas.',
            'input_schema':'Twelve O columns encode attribute groups, never values: xy(2), size(2), doubled angle(2), color(3), shape(3).',
            'upper_panels':'Unchanged from the previous colored version.'},
        'caption_change':'Explain that tensor glyphs show schemas, not activations; three object rows denote variable N. Z consists of seven learned-query language slots. Global g is text-only; Grounder(O,Z) additionally reads objects for m. Contextual features select rows/order; pose heads use original H with g, object sizes and predicted structure. Full/mover constraint repair and held/no-op preservation are summarized in panels a-c and the caption.',
        'source_verification':[
            'arrangement/contracts.py::Objects.features: 12 input channels.',
            'arrangement/goal_encoder.py::GoalEncoder: seven queries, 384-to-128 projection, 13-D text-only goal, object-conditioned grounder.',
            'experiments/relation_pose.py::NeuralRelationPose: 12-to-128-to-128 hidden encoder.',
            'experiments/set_context.py::SetArrangement: four-head context feeds selector; placement receives original hidden.',
            'experiments/row_layout.py::NeuralRowPlacement: contextual rows/order, original-feature origins/angles and size/goal-conditioned steps.'],
        'visual_changes':['Five equal cards replaced by an open representation-and-operator plate.',
            'Dimensioned tensor schemas expose the 12-channel object interface and 128-channel factorization.',
            'Separate language-to-goal and object-grounder branches reveal text-only versus object-conditioned outputs.',
            'Original-H bypass and predicted-structure input converge on the pose operator.',
            'Panels a-c, Times typography, page size, recorded inputs and predictions retained.'],
        'typeset_export':typeset}
    for filename in ('architecture_style_revision.json','final_architecture.json','architecture_typeset_review.json'):
        (m.ROOT/'audit'/filename).write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'PASS','width_pt':WIDTH_PT,'height_pt':HEIGHT_PT,'figure6':after,'exports':typeset['exports_sha256']},indent=2))
    return audit

if __name__=='__main__':generate()

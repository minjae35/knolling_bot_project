"""Figure 3: paired command-wise gains and retained inference bottlenecks.

All estimates derive from archived paired score rows. Family resampling preserves
the five related image–instruction records; no model or runtime is imported.
"""
import hashlib
import json
import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import make_figures as m

HEIGHT=199
PANEL='#F0F1F1'
BLUE,GREEN,TEAL,RED='#507BC2','#55A870','#087E81','#A83B39'
GRAY,INK,MUTED='#7D8B98','#182530','#596874'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frozen_reference():
    frozen=json.loads((m.ROOT/'audit/figure6_frozen.json').read_text())
    for name,expected in frozen['sha256'].items():
        assert sha(m.ROOT/name)==expected, f'Frozen Figure 6 changed: {name}'
    assert frozen['latex_figure_block'] in (m.ROOT/'appendix.tex').read_text()
    return frozen['sha256']['figures/qualitative_modes.png']

def old_exports(name):
    return {ext:sha(m.OUT/f'{name}.{ext}') for ext in ('pdf','svg','png')
            if (m.OUT/f'{name}.{ext}').exists()}

def check_text(fig):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bad=[]
    for artist in fig.findobj(plt.Text):
        if not artist.get_visible() or not artist.get_text():continue
        b=artist.get_window_extent(renderer)
        if b.x0<-.2 or b.y0<-.2 or b.x1>fig.bbox.width+.2 or b.y1>fig.bbox.height+.2:
            bad.append(artist.get_text())
    assert not bad,bad

def finish_audit(name,record):
    path=m.ROOT/'audit/chart_style_revision.json'
    audit=json.loads(path.read_text()) if path.exists() else {}
    old=audit.get('figures',{}).get(name,{})
    audit.pop('style',None)
    if 'previous_exports_sha256' in old:
        record['previous_exports_sha256']=old['previous_exports_sha256']
    audit.update({'reference':'figures/qualitative_modes.png',
        'reference_sha256':frozen_reference(),
        'scope':'Figures 3 and 5 only. Frozen Figure 6 exports, renders, scenes and LaTeX are unchanged.',
        'shared_style':{'minimum_chart_label_pt':7.0,
            'fonts':{'holdout_diagnostics':'Times / native LaTeX math',
                     'selector_training':'Nimbus Roman (manuscript Times face)'},
            'panel_backgrounds':{'selector_training':PANEL,'holdout_diagnostics':'white'},
            'principle':'Individual figure records specify typography, geometry, and palette; Figure 3 now uses editable TikZ while Figure 5 retains its existing artwork.'}})
    record['exports_sha256']=old_exports(name)
    audit.setdefault('figures',{})[name]=record
    path.write_text(json.dumps(audit,indent=2)+'\n')

def evidence():
    d=json.loads((m.ROOT/'audit/verified_results.json').read_text());records={}
    for version in ('v1','v2'):
        path=m.E/f'{version}_holdout.jsonl'
        assert sha(path)==d['sources'][path.name]
        records[version]={r['sample_id']:r for r in map(json.loads,path.read_text().splitlines())}
    assert records['v1'].keys()==records['v2'].keys()
    pairs=Counter((records['v1'][k]['task'],records['v2'][k]['task']) for k in records['v1'])
    assert [pairs[True,True],pairs[False,True],pairs[True,False],pairs[False,False]]==[489,51,0,100]
    partition=Counter()
    for record in records['v2'].values():
        if not record['task']:
            tags=set(record['errors'])
            partition['horizontal_gap' in tags,'vertical_gap' in tags]+=1
    spacing=[partition[True,False],partition[True,True],partition[False,True],partition[False,False]]
    assert spacing==[29,20,22,29]
    assert sum(spacing[:3])==d['v2']['unique_spacing_failure_scenes']==71
    return d,spacing

def paired_mode_analysis():
    """Reaggregate paired gains, pointwise family-bootstrap CIs, and error tags."""
    verified,_=evidence()
    records={version:{r['sample_id']:r for r in map(json.loads,
              (m.E/f'{version}_holdout.jsonl').read_text().splitlines())}
             for version in ('v1','v2')}
    for sid,row in records['v1'].items():
        assert all(row[key]==records['v2'][sid][key]
                   for key in ('scene_family_id','mode','objects'))
    families=sorted({row['scene_family_id'] for row in records['v1'].values()})
    assert len(families)==128
    family_counts=Counter(row['scene_family_id'] for row in records['v1'].values())
    assert set(family_counts.values())=={5}
    modes=['full','partial','no_op','overall']
    totals={mode:{fid:[0,0] for fid in families} for mode in modes}
    for sid,row in records['v1'].items():
        change=int(records['v2'][sid]['task'])-int(row['task'])
        for mode in (row['mode'],'overall'):
            totals[mode][row['scene_family_id']][0]+=change
            totals[mode][row['scene_family_id']][1]+=1
    rng=random.Random(20261112)
    draws={mode:[] for mode in modes}
    for _ in range(10000):
        sampled=[rng.choice(families) for _ in families]
        for mode in modes:
            numerator=sum(totals[mode][fid][0] for fid in sampled)
            denominator=sum(totals[mode][fid][1] for fid in sampled)
            assert denominator>0
            draws[mode].append(100*numerator/denominator)
    results={}
    for mode in modes:
        values=sorted(draws[mode])
        numerator=sum(x[0] for x in totals[mode].values())
        denominator=sum(x[1] for x in totals[mode].values())
        results[mode]={'paired_gain_count':numerator,'n':denominator,
            'families_with_mode':sum(x[1]>0 for x in totals[mode].values()),
            'v1_success':verified['v1'][mode]['success'],
            'v2_success':verified['v2'][mode]['success'],
            'gain_pp':100*numerator/denominator,
            'ci95_pp':[values[250],values[9749]]}
        assert denominator==verified['v1'][mode]['total']==verified['v2'][mode]['total']
        assert numerator==results[mode]['v2_success']-results[mode]['v1_success']
    assert results['overall']['ci95_pp']==verified['paired']['family_bootstrap']['ci95_pp']
    assert abs(results['overall']['gain_pp']-verified['paired']['difference_pp'])<1e-12
    definitions=[('any_gap','Any gap'),('count','Count'),('requested_region','Region'),
        ('size_order','Size order'),('keep','Preservation'),('clearance','Clearance'),
        ('mixed_color_row','Color rows'),('collision','Collision'),('boundary','Boundary')]
    errors=[]
    for key,label in definitions:
        entry={'key':key,'label':label}
        for version in records:
            entry[version]=sum(
                any(t in row['errors'] for t in ('horizontal_gap','vertical_gap'))
                if key=='any_gap' else key in row['errors']
                for row in records[version].values())
            expected=(verified[version]['unique_spacing_failure_scenes']
                if key=='any_gap' else verified[version]['error_tags'].get(key,0))
            assert entry[version]==expected
        errors.append(entry)
    assert [(r['v1'],r['v2']) for r in errors]==[
        (77,71),(45,5),(17,4),(13,11),(9,4),(9,7),(2,0),(1,0),(1,0)]
    report={'status':'PASS','source_sha256':verified['sources'],
        'source_paths':[str((m.E/f'{v}_holdout.jsonl').relative_to(m.ROOT)) for v in records],
        'paired_records':640,'families':128,'records_per_family':5,
        'bootstrap':{'seed':20261112,'resamples':10000,
            'resampling_unit':'128 scene families sampled with replacement; the same family indices are used for every mode and both releases.',
            'estimator':'100 * sum(v2_task - v1_task) / number of sampled records of that mode; mode denominator is recomputed within each family resample.',
            'interval':'Pointwise 95% percentile intervals; sorted bootstrap samples at zero-based indices 250 and 9749.',
            'percentile_indices':[250,9749],
            'scope':'Evaluation-family variation conditional on frozen releases, not training-seed uncertainty. Intervals are not simultaneous and do not test differences between modes.'},
        'modes':results,'error_counts':errors,
        'error_scope':'Counts of emitted error tags across the same 640 records. Any gap is the horizontal/vertical union; other categories may overlap. Count mismatch terminates downstream checks, so tag reductions are not isolated causal effects or uncensored latent-error prevalence.',
        'new_inference':False,'existing_overall_statistics_match':True}
    (m.ROOT/'audit/paired_mode_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def generate():
    from figure_tex import compile_figure, node
    frozen_reference();before=old_exports('holdout_diagnostics')
    analysis_before=(m.ROOT/'audit/paired_mode_analysis.json').read_bytes()
    data=paired_mode_analysis()
    assert (m.ROOT/'audit/paired_mode_analysis.json').read_bytes()==analysis_before
    pieces=[]
    def text(x,y,value,anchor='west',color='ink',size=10.5):
        pieces.append(node(x,y,value,
            rf'anchor={anchor},text={color},font=\fontsize{{{size}}}{{{size+1.5:.1f}}}\selectfont'))
    def line(x1,y1,x2,y2,style='draw=ink,line width=.45bp'):
        pieces.append(rf'\draw[{style}] ({x1:.5f},{y1:.5f}) -- ({x2:.5f},{y2:.5f});'+'\n')
    def mark(x,y,style,radius=1.4):
        pieces.append(rf'\path[{style}] ({x:.5f},{y:.5f}) circle[radius={radius}bp];'+'\n')

    # The two plot areas share their top, baseline and tick-label grid. Text and
    # mathematics are typeset by the same Times/LaTeX stack as the manuscript.
    text(0,8,r'\textbf{a}\quad Paired gain',size=11.5)
    text(205,8,r'\textbf{b}\quad Recorded failures',size=11.5)
    pieces.append(r'\begin{scope}[shift={(0,4)}]'+'\n')
    text(42,25,r'95\% family-bootstrap interval',color='muted')
    for x,label,style in [(274,'v1','draw=ochre,fill=white,line width=.45bp'),
                          (326,'v2','draw=teal,fill=teal,line width=.35bp')]:
        mark(x,25,style);text(x+6,25,label)

    lx0,lx1=42.,177.
    rx0,rx1=264.,390.
    top,baseline=39.,166.
    left_x=lambda value:lx0+(lx1-lx0)*value/25
    right_x=lambda value:rx0+(rx1-rx0)*value/80
    # A single low-contrast band distinguishes the pooled estimate from modes;
    # geometry and the redundant diamond encoding remain unchanged.
    pieces.append(r'\path[fill=teal!6,draw=none] (0,143) rectangle (183,163);'+'\n')
    # Thin reference lines distinguish chart structure from measured values.
    line(lx0,top,lx0,baseline,'draw=hairline,line width=.35bp,dash pattern=on 1.6bp off 1.6bp')
    line(lx0,139,lx1,139,'draw=hairline,line width=.3bp')
    for tick in [0,20,40,60,80]:
        xx=right_x(tick)
        line(xx,top,xx,baseline,'draw=black!10,line width=.3bp')
    for x0,x1,ticks,mapper in [(lx0,lx1,[0,10,20],left_x),
                              (rx0,rx1,[0,20,40,60,80],right_x)]:
        line(x0-2,baseline,x1,baseline,'draw=muted,line width=.4bp')
        for tick in ticks:
            xx=mapper(tick);line(xx,baseline,xx,baseline+2.5,'draw=muted,line width=.4bp')
            text(xx,175,str(tick),'center','muted')

    modes=['full','partial','no_op','overall']
    for mode,label,y in zip(modes,['Full','Partial','No-op','Overall'],[53,86,119,153]):
        text(lx0-7,y,label,'east')
        row=data['modes'][mode];lo,hi=row['ci95_pp'];point=row['gain_pp']
        xl,xh,xp=left_x(lo),left_x(hi),left_x(point)
        color='teal' if mode=='overall' else 'objectblue'
        line(xl,y,xh,y,f'draw={color},line width=.55bp')
        for xx in (xl,xh):line(xx,y-2,xx,y+2,f'draw={color},line width=.4bp')
        if mode=='overall':
            pieces.append(rf'\path[draw=teal,fill=teal,line width=.35bp] ({xp:.5f},{y-1.9:.5f}) -- ({xp+1.9:.5f},{y:.5f}) -- ({xp:.5f},{y+1.9:.5f}) -- ({xp-1.9:.5f},{y:.5f}) -- cycle;'+'\n')
        else:
            mark(xp,y,'draw=objectblue,fill=objectblue,line width=.35bp',1.5)
    low_count_offsets=[]
    for i,row in enumerate(data['error_counts']):
        y=45+i*14
        text(rx0-8,y,row['label'],'east')
        a,b=right_x(row['v1']),right_x(row['v2'])
        # For 1→0 pairs, a sub-row offset keeps both symbols distinguishable;
        # their horizontal positions are exact, with no numerical jitter.
        offset=1.2 if abs(a-b)<2.8 else 0
        ya,yb=y-offset,y+offset
        line(a,ya,b,yb,'draw=hairline,line width=.5bp')
        mark(a,ya,'draw=ochre,fill=white,line width=.45bp')
        mark(b,yb,'draw=teal,fill=teal,line width=.35bp')
        low_count_offsets.append({'category':row['key'],'v1_vertical_offset_bp':-offset,
                                  'v2_vertical_offset_bp':offset})
    text((lx0+lx1)/2,188,'Success gain (percentage points)','center')
    text((rx0+rx1)/2,188,'Scenes (of 640)','center')
    pieces.append(r'\end{scope}'+'\n')
    typeset=compile_figure('holdout_diagnostics',''.join(pieces),HEIGHT,label_pt=10.5)
    colors={'v1':'#B27A46','v2':'#087E81','mode_intervals':'#507BC2',
            'pooled_interval':'#087E81','pooled_band':'teal!6'}
    record={'size_pt':[396,HEIGHT],'source':'scripts/quantitative_figure.py',
        'previous_exports_sha256':before,'sources_sha256':data['source_sha256'],
        'analysis':'audit/paired_mode_analysis.json',
        'modes':data['modes'],'error_counts':data['error_counts'],
        'style':{'background':'white','minimum_label_pt':10.5,'axis_label_pt':10.5,
            'panel_title_pt':11.5,'palette':colors,'error_count_ticks':[0,20,40,60,80],
            'font':'Times text / LaTeX mathematics','plot_widths_bp':[135,126],
            'marker_diameter_bp':2.8,'primary_line_bp':.45,'secondary_line_bp':.35,
            'pooled_band_bounds_bp':[0,143,183,163],
            'color_semantics':'Blue mode estimates; teal pooled estimate and v2; warm ochre open v1 markers. Marker shape and labels retain all distinctions without color.',
            'low_count_vertical_offsets':low_count_offsets,'typeset_source':typeset['source']},
        'scientific_question':'Which command modes gain, and which inference bottleneck persists?',
        'visible_changes':['Pointwise paired family-bootstrap intervals for each command mode',
            'Before/after comparison of all nine emitted error categories (spacing union)',
            'One connected analytical argument rather than unrelated complexity and residual-count plots',
            'Aligned plot baselines; understated panel titles and Times labels; fine capped intervals and small release endpoints',
            'Restrained blue/teal estimate hierarchy and warm-ochre/teal release comparison; one subtle pooled-row band'],
        'references':[{'title':'CLIPort','url':'https://arxiv.org/pdf/2109.12098',
            'inspected':'Figure 3, PDF page 6, and Figure 4, PDF page 8',
            'observation':'Figure 3 uses grouped bar charts, not curves; a repeated comparison isolates seen/unseen performance. Adopt explicit empirical contrasts, not its exact graphic form.'},
            {'title':'DINOv2','url':'https://arxiv.org/pdf/2304.07193',
            'inspected':'Figure 2, PDF page 3, and Figure 6, PDF page 10',
            'observation':'Consistent axes and visual codes expose model/condition differences. Adopt claim-led comparisons and small labels with limited accents.'},
            {'title':'Transporter Networks','url':'https://proceedings.mlr.press/v155/zeng21a/zeng21a.pdf',
            'inspected':'Figures 9–10, PDF pages 18–19',
            'observation':'Repeated model comparisons connect generalization and failure patterns. Adopt a comparison that interrogates the bottleneck; do not copy unmeasured breadth.'}],
        'scope':data['bootstrap']['scope'] if 'scope' in data['bootstrap'] else data['bootstrap']['interval'],
        'caveats':[data['error_scope'],'Mode intervals are pointwise; observed differences between modes are not a causal ablation.'],
        'text_bounds':'PASS','frozen_figure6_integrity':'PASS','new_inference':False}
    finish_audit('holdout_diagnostics',record)
    record['exports_sha256']=old_exports('holdout_diagnostics')
    record['generator_sha256']=sha(m.ROOT/'scripts/quantitative_figure.py')
    (m.ROOT/'audit/quantitative_reference_revision.json').write_text(json.dumps(record,indent=2)+'\n')
    (m.ROOT/'audit/quantitative_typeset_review.json').write_text(json.dumps({
        'figure':3,'scientific_analysis_bytes_unchanged':True,'typesetting':typeset,
        'style':record['style'],'text_bounds':'PASS','new_inference':False,
        'figure6_integrity':'PASS','minimum_body_text_pt':10.5},indent=2)+'\n')
    frozen_reference()

if __name__=='__main__':generate()

"""Paired release outcomes, complexity and disjoint spacing-failure counts.

Only frozen score summaries and JSONL records are read. No inference is run.
"""
import hashlib
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import make_figures as m

HEIGHT = 200


def evidence():
    d = json.loads((m.ROOT/'audit/verified_results.json').read_text())
    records = {}
    for version in ('v1','v2'):
        path = m.E/f'{version}_holdout.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == d['sources'][path.name]
        records[version] = {r['sample_id']:r for r in map(json.loads,path.read_text().splitlines())}
    assert records['v1'].keys() == records['v2'].keys()
    pairs = Counter((records['v1'][k]['task'], records['v2'][k]['task']) for k in records['v1'])
    assert [pairs[True,True],pairs[False,True],pairs[True,False],pairs[False,False]] == [489,51,0,100]
    partition = Counter()
    for r in records['v2'].values():
        if not r['task']:
            tags = set(r['errors'])
            partition['horizontal_gap' in tags, 'vertical_gap' in tags] += 1
    counts = [partition[True,False],partition[True,True],partition[False,True],partition[False,False]]
    assert counts == [29,20,22,29]
    assert sum(counts[:3]) == d['v2']['unique_spacing_failure_scenes'] == 71
    return d, counts


def generate():
    d, spacing = evidence()
    fig, base = m.canvas(HEIGHT)
    def t(x,y,s,size=8,color=m.INK,weight='normal',ha='left',**kw):
        return m.text(base,x,y,s,size,color,weight,ha=ha,**kw)

    # Pairing is the primary result, not two independent percentage bars.
    t(0,8,'a  Paired outcomes · 640 scenes',8.4,weight='bold')
    t(394,8,'51 gains · 0 regressions',8,m.BLUE,'bold',ha='right')
    groups=[(489,'Retained success','#c7d4df',m.INK),
            (51,'Gained',m.BLUE,'white'),
            (100,'Still fail','#e9edef',m.INK)]
    x=0
    for count,label,color,text_color in groups:
        w=396*count/640
        base.add_patch(Rectangle((x,22),w,16,fc=color,ec='white',lw=.6))
        t(x+w/2,30,str(count),8.2,text_color,'bold',ha='center')
        t(x+w/2,46,label,7.2,m.MUTED,ha='center')
        x+=w
    lo,hi=d['paired']['family_bootstrap']['ci95_pp']
    t(0,62,f"+{d['paired']['difference_pp']:.1f} pp",10.1,m.BLUE,'bold')
    t(58,62,f'95% family-bootstrap CI: +{lo:.1f} to +{hi:.1f} pp',7.6,m.MUTED)
    base.plot([0,396],[71,71],color=m.LINE,lw=.55)

    # Complexity curve retains the complete 0–100% success scale.
    t(0,83,'b  Success vs. scene complexity',8.2,weight='bold')
    ax=fig.add_axes([29/396,(HEIGHT-171)/HEIGHT,159/396,72/HEIGHT])
    for version,color,marker in [('v1',m.GRAY,'o'),('v2',m.BLUE,'s')]:
        records=d[version]['by_object_count']
        xs=np.array([int(k) for k in records])
        ys=np.array([100*r['success']/r['total'] for r in records.values()])
        ax.plot(xs,ys,color=color,marker=marker,ms=2.7,lw=1.05,label=version)
        ax.text(8.15,ys[-1],f'{ys[-1]:.1f}',fontsize=7.2,color=color,va='center',weight='bold')
    ax.set(xlim=(1.75,9.20),ylim=(0,103),xticks=list(range(2,9)),yticks=[0,50,100])
    ax.tick_params(labelsize=7,length=2.5,pad=2)
    ax.set_ylabel('Success (%)',fontsize=7.4,labelpad=3)
    ax.set_xlabel('Objects per scene',fontsize=7.4,labelpad=3)
    ax.grid(axis='y',color='#e4e9ed',lw=.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=7,loc='lower left',ncol=2,
              borderpad=0,handlelength=1.4,handletextpad=.4,columnspacing=1.0)

    # A true partition of 100 failures: overlapping raw tags are not added.
    t(215,83,'c  Remaining failures',8.2,weight='bold')
    t(215,96,'71 / 100 have a spacing error',7.5,m.TEAL,'bold')
    labels=['Horizontal only','Both gaps','Vertical only','Neither gap']
    bar_x=286; unit=2.9
    for y,count,label in zip([111,129,147,165],spacing,labels):
        t(215,y,label,7.2)
        color=m.TEAL if label!='Neither gap' else '#b9c4cc'
        base.add_patch(Rectangle((bar_x,y-3.5),count*unit,7,fc=color,ec='none'))
        t(bar_x+count*unit+4,y,str(count),7.3,m.TEAL if label!='Neither gap' else m.MUTED,'bold')
    base.plot([bar_x,bar_x+30*unit],[177,177],color=m.LINE,lw=.65)
    for value in [0,15,30]:
        xx=bar_x+value*unit
        base.plot([xx,xx],[177,180],color=m.MUTED,lw=.6)
        t(xx,186,str(value),7,m.MUTED,ha='center')
    t(bar_x+15*unit,196,'Failed scenes',7.4,m.MUTED,ha='center')

    # Validate artist text at the intended physical insertion size.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bad=[]
    for artist in fig.findobj(plt.Text):
        if not artist.get_visible() or not artist.get_text():continue
        bbox=artist.get_window_extent(renderer)
        if bbox.x0 < -.2 or bbox.y0 < -.2 or bbox.x1 > fig.bbox.width+.2 or bbox.y1 > fig.bbox.height+.2:
            bad.append(artist.get_text())
    assert not bad,bad
    m.save(fig,'holdout_diagnostics')
    notes=f'''# Quantitative redesign — 2026-11-15

The main results figure now answers three different questions in a 396 × {HEIGHT} pt composition. Entry point: `scripts/quantitative_figure.py::generate()`.

- **Paired outcomes:** 489 retained successes, 51 gains, no regressions, 100 persistent failures among 640 pairs. Segment widths are proportional to counts. The gain is {d['paired']['difference_pp']:.8f} percentage points, with the existing paired family-bootstrap interval [{lo}, {hi}] pp displayed to one decimal. This is not an interval for independent trials or training seeds.
- **Scene complexity:** the saved success curves for 2–8 objects remain, using the full 0–100% scale. Bins contain 90 or 95 pairs; no uncomputed uncertainty bands are invented.
- **Spacing:** the 100 v2 failures are partitioned into 29 horizontal-only, 20 both-gap, 22 vertical-only, and 29 neither-gap scenes. Thus 71 have one or more spacing violations. Unlike a sum of raw error tags, these four bars are mutually exclusive. A scene in a spacing category may additionally violate another condition; “neither gap” does not mean absence of other failures.

The paired counts and partition are independently rederived from frozen JSONL records on every generation. Source hashes are checked against `audit/verified_results.json`. The original data, model, and predictions are not modified. All labels are at least 7 pt; the composition is exported to vector PDF/SVG and a 300 dpi PNG. Text extents are checked against the canvas before export.

Suggested caption: **Paired release gains and remaining difficulty.** (a) All 489 v1 successes remain valid; v2 gains 51 scenes and loses none. The interval resamples the 128 paired families. (b) Success generally decreases with object count (90–95 pairs per bin). (c) Gap categories partition the 100 remaining failures: horizontal-only, both, vertical-only, or neither. Other error types may coexist with any gap category.

The exact mode counts remain in the adjacent manuscript table. This figure intentionally does not repeat that complete table. All nonzero raw error tags remain documented in the manuscript and appendix.

Input SHA-256:

- `v1_holdout.jsonl`: `{d['sources']['v1_holdout.jsonl']}`
- `v2_holdout.jsonl`: `{d['sources']['v2_holdout.jsonl']}`
'''
    (m.ROOT/'audit/v3_quantitative.md').write_text(notes)


if __name__=='__main__':
    generate()

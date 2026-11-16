"""Figure 5: historical selector curves, distinct from sealed release scores."""
import json
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import make_figures as m
from quantitative_figure import (
    PANEL,BLUE,GREEN,TEAL,GRAY,INK,MUTED,sha,
    frozen_reference,old_exports,check_text,finish_audit,
)

for _face in ('Regular','Bold','Italic','BoldItalic'):
    font_manager.fontManager.addfont(f'/usr/share/fonts/opentype/urw-base35/NimbusRoman-{_face}.otf')

# Same face as the manuscript Times stack and the other matplotlib figures.
SERIF={'fontfamily':'Nimbus Roman'}
HEIGHT=176
PANEL_TOP,PANEL_BOTTOM=24,152
PLOT_TOP,PLOT_H=36,80
PLOT_BOTTOM=PLOT_TOP+PLOT_H

def generate():
    frozen_reference();before=old_exports('selector_training')
    path=m.E/'selector_full_result.json';d=json.loads(path.read_text());curve=d['curve']
    epochs=[row['epoch'] for row in curve]
    assert epochs==list(range(1,9))
    assert d['best_epoch']==1 and all(row['val_total']==1200 for row in curve)
    assert [curve[0]['val_task'],curve[-1]['val_task']]==[235,233]
    assert d['baseline_val']['task']==222 and d['baseline_val']['total']==1200
    plt.rcParams['font.family']='Nimbus Roman'
    fig,base=m.canvas(HEIGHT)
    def text(x,y,s,size=9.0,color=INK,bold=False,ha='left'):
        return m.text(base,x,y,s,size,color,'bold' if bold else 'normal',ha=ha,**SERIF)
    text(0,8,'a  Structure prediction',11.0,bold=True)
    text(204,8,'b  Complete-scene success',11.0,bold=True)
    for x in (0,204):base.add_patch(Rectangle((x,PANEL_TOP),192,PANEL_BOTTOM-PANEL_TOP,fc=PANEL,ec='none'))
    plot_box=[149/396,PLOT_H/HEIGHT]
    ax=fig.add_axes([29/396,(HEIGHT-PLOT_BOTTOM)/HEIGHT,*plot_box],facecolor=PANEL)
    bx=fig.add_axes([233/396,(HEIGHT-PLOT_BOTTOM)/HEIGHT,*plot_box],facecolor=PANEL)
    metrics={}
    for key,color,marker,label in [('val_row_accuracy',BLUE,'o','Row'),('val_slot_accuracy',GREEN,'s','Slot')]:
        ys=[100*row[key] for row in curve];metrics[key]=ys
        ax.plot(epochs,ys,color=color,marker=marker,ms=3.2,lw=1.35,label=label)
        ax.text(8.2,ys[-1],f'{ys[-1]:.1f}',fontsize=9.0,color=color,weight='bold',va='center',fontfamily='Nimbus Roman')
    ax.set(xlim=(.7,9.55),ylim=(60,90),xticks=epochs,yticks=[60,70,80,90])
    ax.set_ylabel('Validation accuracy (%)',fontsize=9.0,labelpad=4)
    ax.legend(frameon=False,fontsize=9.0,loc='lower right',ncol=2,
              handlelength=1.3,columnspacing=.9,handletextpad=.45,borderpad=.25)
    values=[100*row['val_task']/row['val_total'] for row in curve]
    baseline=100*d['baseline_val']['task']/d['baseline_val']['total']
    bx.axvspan(.7,1.3,fc='#D7E6E1',ec='none',zorder=1)
    bx.axhline(baseline,color=GRAY,ls=(0,(2,2)),lw=.8,zorder=2)
    bx.plot(epochs,values,color=TEAL,marker='o',ms=3.2,lw=1.35,zorder=3)
    bx.plot(1,values[0],'o',ms=5.2,mfc='white',mec=TEAL,mew=1.15,zorder=4)
    bx.text(1.22,22.6,'235/1200',fontsize=9.0,color=TEAL,fontfamily='Nimbus Roman')
    bx.text(8.05,22.6,'233/1200',fontsize=9.0,color=TEAL,ha='right',fontfamily='Nimbus Roman')
    bx.text(8.05,14.2,'baseline 222/1200',fontsize=9.0,color=MUTED,ha='right',fontfamily='Nimbus Roman')
    bx.set(xlim=(.7,8.35),ylim=(0,25),xticks=epochs,yticks=[0,5,10,15,20,25])
    bx.set_ylabel('Strict success (%)',fontsize=9.0,labelpad=4)
    for chart in (ax,bx):
        chart.tick_params(labelsize=8.5,length=2.2,pad=2.5)
        chart.grid(axis='y',color='white',lw=.8);chart.set_axisbelow(True)
    text(96,141,'Training epoch',9.0,MUTED,ha='center')
    text(300,141,'Training epoch · selected: 1',9.0,TEAL,ha='center')
    text(198,166,'Historical validation · 1,200 full-layout pairs · before release refinements',9.5,INK,ha='center')
    check_text(fig);m.save(fig,'selector_training')
    finish_audit('selector_training',{
        'size_pt':[396,HEIGHT],'source':'scripts/training_figure.py',
        'previous_exports_sha256':before,'source_sha256':{str(path.relative_to(m.ROOT)):sha(path)},
        'epochs':epochs,'metrics_percent':metrics,'strict_success_counts':[row['val_task'] for row in curve],
        'validation_pairs':1200,'baseline_count':222,'selected_epoch':1,
        'visible_changes':['gray equal-size panels match Figure 6 hierarchy and alignment',
            'Nimbus Roman, the manuscript Times face, for titles, ticks, and the scope line',
            'blue row and green slot curves with direct endpoint labels',
            'teal complete-scene curve with selected-epoch marker',
            'genuine 222/1200 baseline drawn as a neutral dashed reference',
            'scope line set at 9.5 pt so it reads with the axis labels'],
        'scientific_scope':'Saved eight-epoch selector-stage validation before final inference rules. Not sealed holdout performance. No new measurements or uncertainty bands.',
        'text_bounds':'PASS','frozen_figure6_integrity':'PASS'})

if __name__=='__main__':generate()

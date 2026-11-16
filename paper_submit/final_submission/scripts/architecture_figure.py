"""Publication architecture from the adopted inference source.

This figure contains schematic vector nodes and the frozen real input RGB.
No inferred features, fabricated activation maps, or generated evidence appear.
The diagram intentionally distinguishes global text-only goals from referents,
selector context from pose features, and mover decoding from held/no-op copying.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import make_figures as m

WIDTH_PT = 396
HEIGHT_PT = 232
MINIMUM_LABEL_PT = 7.0
INK, MUTED, BLUE = m.INK, m.MUTED, m.BLUE
RULE, WASH, RULE_WASH = '#8A5A2A', '#F1F6FA', '#FCF8F2'


def generate():
    fig, ax = m.canvas(HEIGHT_PT)
    texts, cards = [], []

    def text(x, y, value, size=7.2, color=INK, bold=False, ha='left'):
        assert size >= MINIMUM_LABEL_PT
        item = m.text(ax, x, y, value, size, color,
                      'bold' if bold else 'normal', ha=ha, zorder=5)
        texts.append(item)
        return item

    def card(x, y, width, height, rule=False):
        item = FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0,rounding_size=2',
            facecolor=RULE_WASH if rule else WASH,
            edgecolor=RULE if rule else BLUE, linewidth=.75,
            linestyle=(0, (2.5, 1.8)) if rule else '-', zorder=2)
        ax.add_patch(item)
        cards.append((x, y, width, height))

    def arrow(points, color=BLUE, dashed=False):
        # Axis-aligned routing deliberately leaves node interiors clear.
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, lw=.78,
                    linestyle=(0, (2.5, 1.8)) if dashed else '-',
                    solid_joinstyle='round', zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=6.8, lw=.78, color=color, shrinkA=.6, shrinkB=.6,
            linestyle=(0, (2.5, 1.8)) if dashed else '-', zorder=3))

    # (a) The two input streams and final routing. Text-only g is deliberately
    # not reached by an object arrow. Selection explicitly includes O as input.
    text(0, 8, 'a  Hybrid inference', 8.3, bold=True)
    text(178, 8, r'held / no-op: detected $xy$ + RGB angle', 7.0, MUTED)

    image_path = m.E / 'qualitative_full_source.png'
    ax.imshow(plt.imread(image_path), extent=(0, 33, 50, 28), aspect='auto', zorder=2)
    text(16.5, 58, 'RGB', 7.0, MUTED, ha='center')
    card(43, 25, 72, 31)
    text(79, 35, 'RT-DETR-R18', 7.3, BLUE, True, 'center')
    text(79, 46, 'ROI + size heads', 7.0, MUTED, ha='center')
    arrow([(34, 40), (42, 40)])

    card(126, 25, 49, 31)
    text(150.5, 35, r'$O\in\mathbb{R}^{N\times12}$', 7.5, BLUE, ha='center')
    text(150.5, 46, 'objects', 7.0, MUTED, ha='center')
    arrow([(116, 40), (125, 40)])

    card(187, 25, 87, 31)
    text(230.5, 35, 'Layout core (b)', 7.3, BLUE, True, 'center')
    text(230.5, 46, 'full / selected movers', 7.0, MUTED, ha='center')
    arrow([(176, 40), (186, 40)])

    card(286, 25, 76, 31, rule=True)
    text(324, 35, 'Geometry', 7.3, RULE, True, 'center')
    text(324, 46, 'constrain / repair', 7.0, RULE, ha='center')
    arrow([(275, 40), (285, 40)], RULE)
    arrow([(363, 40), (380, 40)], RULE)
    text(386, 39, r'$\hat P$', 10.0, INK, ha='center')
    text(383.5, 54, 'poses', 7.0, MUTED, ha='center')
    # This is a bypass of layout/geometry for held objects, then a merge into P.
    arrow([(150.5, 24), (150.5, 18), (386, 18), (386, 29)], MUTED, dashed=True)

    card(0, 73, 101, 28)
    text(50.5, 82, 'English → MiniLM', 7.3, BLUE, True, 'center')
    text(50.5, 93, '7-query slot attention', 7.0, MUTED, ha='center')

    card(115, 73, 101, 28)
    text(165.5, 82, r'Text-only goal $g(t)\in\mathbb{R}^{13}$', 7.0, BLUE, ha='center')
    text(165.5, 93, 'region / gap correction', 7.0, RULE, ha='center')
    arrow([(102, 87), (114, 87)])
    arrow([(201, 72), (201, 57)])

    card(229, 73, 133, 28)
    text(295.5, 82, r'Selection $m(O,\mathrm{language})$', 7.2, BLUE, ha='center')
    text(295.5, 93, 'named-color refinement', 7.0, RULE, ha='center')
    arrow([(50.5, 102), (50.5, 110), (295.5, 110), (295.5, 102)])
    arrow([(252, 72), (252, 57)])
    # O is written in m's argument list to avoid a crossing over the g branch.

    ax.plot([0, WIDTH_PT], [120, 120], color=m.LINE, lw=.7)

    # (b) An explicit context-only selector path and untouched feature bypass.
    text(0, 131, 'b  Learned layout', 8.3, bold=True)
    text(104, 131, 'Context selects structure; original features predict poses.', 7.0, MUTED)

    text(5, 174, r'$O$', 9.0, BLUE, ha='center')
    card(21, 155, 75, 39)
    text(58.5, 166, 'Relation MLP', 7.4, BLUE, True, 'center')
    text(58.5, 181, '12 → 128 → 128', 7.0, MUTED, ha='center')
    arrow([(12, 174), (20, 174)])
    text(106, 174, r'$h$', 9.0, BLUE, ha='center')
    arrow([(97, 174), (100, 174)])

    card(120, 155, 82, 39)
    text(161, 166, '4-head attention', 7.3, BLUE, True, 'center')
    text(161, 181, 'goal g + object sizes', 7.0, MUTED, ha='center')
    arrow([(111, 174), (119, 174)])
    text(212, 161, r'$\tilde h$', 8.3, BLUE, ha='center')

    card(222, 155, 82, 39)
    text(263, 165, 'Row assignment', 7.3, BLUE, True, 'center')
    text(263, 180, 'Pointer order', 7.3, BLUE, True, 'center')
    arrow([(203, 174), (221, 174)])
    text(312, 159, r'$r,\pi$', 7.5, BLUE, ha='center')

    card(320, 155, 76, 39)
    text(358, 165, 'Pose heads', 7.3, BLUE, True, 'center')
    text(358, 180, 'origins · steps · θ', 7.0, MUTED, ha='center')
    arrow([(305, 174), (319, 174)])
    arrow([(106, 183), (106, 218), (358, 218), (358, 195)])
    text(229, 209, r'original $h$ + goal $g$ + sizes', 7.0, BLUE, ha='center')

    # Labels are sized in physical points and exported at the full paper width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer) for item in texts]
    frame = fig.bbox
    out_of_bounds = [item.get_text() for item, box in zip(texts, boxes)
        if box.x0 < frame.x0 - .05 or box.y0 < frame.y0 - .05
        or box.x1 > frame.x1 + .05 or box.y1 > frame.y1 + .05]
    overlaps = []
    for i, (first, a) in enumerate(zip(texts, boxes)):
        for second, b in zip(texts[i+1:], boxes[i+1:]):
            if a.overlaps(b):
                overlaps.append([first.get_text(), second.get_text()])
    audit = {
        'width_pt': WIDTH_PT, 'height_pt': HEIGHT_PT,
        'minimum_label_pt': min(item.get_fontsize() for item in texts),
        'labels': len(texts), 'out_of_bounds_labels': out_of_bounds,
        'overlapping_label_pairs': overlaps,
        'source': 'scripts/architecture_figure.py',
        'export_formats': ['pdf', 'svg', 'png'],
        'png_dpi': 300,
        'rgb_input': str(image_path.relative_to(m.ROOT)),
        'routing_notes': [
            'The global goal g is text-only; selection m additionally reads O and language slots.',
            'Region/gap correction and named-color refinement are explicit rules (brown labels).',
            'Full-scene or selected-mover proposals enter geometric decoding and bounded repair.',
            'Held/no-op outputs bypass layout; centers are detected and angles use RGB PCA with learned fallback.',
            'Set context changes row/pointer features only; pose heads retain original relation h.',
            'The scalar requested gap is delta; g denotes only the 13-dimensional global goal.',
            'Thumbnails show frozen actual RGB; no model activations or synthetic predictions are depicted.'
        ],
        'implementation_references': [
            'src/knolling_model/experiments/size_refinement.py:299-340',
            'src/knolling_model/arrangement/goal_encoder.py:28-40',
            'src/knolling_model/experiments/set_context.py:43-58',
            'src/knolling_model/experiments/row_layout.py:250-274',
            'src/knolling_model/experiments/partial_arrangement.py:182-249'
        ]
    }
    (m.ROOT/'audit/final_architecture.json').write_text(json.dumps(audit, indent=2)+'\n')
    if out_of_bounds or overlaps:
        raise RuntimeError(f'Architecture label check failed: {out_of_bounds=}, {overlaps=}')
    m.save(fig, 'architecture_main')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    generate()

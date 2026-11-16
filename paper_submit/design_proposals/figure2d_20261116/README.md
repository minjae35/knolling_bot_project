# Figure 2(d) — isolated review proposal

**Status: proposal only. Not integrated into the manuscript.**

The canonical `paper_submit/knolling_bot_v2_paper.pdf`, LaTeX, figure assets, scientific evidence and source ZIP have not been changed. All pre-existing working-tree changes are preserved. The comparison uses the current on-disk figure captured on 2026-11-16, not an older revision.

## View

- `comparison.png` / `comparison.pdf`: current panel on the left, proposal on the right, at equal scale.
- `proposal_d.png` / `proposal_d.pdf` / `proposal_d.svg`: the proposed panel by itself.
- `figure2_proposal_context.png` / `.pdf`: the proposed panel with the current panels (a)–(c), which are pixel-identical.
- `proposal_d.tex`: editable Times/TikZ source.
- `make_proposal.py`: isolated builder; writes only within this proposal directory and checks the canonical file hashes.
- `baseline/`: snapshot of the current figure and its source, retained for comparison.

The existing 396 × 143 bp panel footprint is retained. This is a design preview, not a rebuilt paper or an approved replacement.

## Assessment of the current panel

The current figure has worthwhile foundations: Times typography matches the manuscript, dimensions are explicit, language/object/structure colors are restrained, and the original-H bypass expresses an important architectural distinction. Its open layout avoids oversized decorative cards.

The visual grammar is inconsistent, however. MiniLM and Pose heads act as nodes, while Relation MLP, Set attention and Rows + pointer read as captions under arrows. Tensor miniatures occupy the principal reading line without explaining more than their dimension labels. The goal and size terms have no visible downstream inputs; the grounder's O argument has no explicit input port. The structure wire joins the pose output after the operator, obscuring the real dependency. The current O bracket reaches x = −1 bp, and the bottom pose tensor is crowded against the canvas edge. Empty upper-right space coexists with a compressed lower branch.

## Proposed changes

The proposal uses operators as nodes and representations as edge labels. This changes the reading sequence, rather than only changing colors or box shapes.

1. A compact language sequence explicitly inserts learned-query attention between MiniLM and Z. The text-only goal readout and object-conditioned grounder branch separate before the lower structure path.
2. The O input to the grounder is a named local port. S is visibly derived from O. Repeated g/S labels are local references to these shared quantities, not additional encoders or independent inputs.
3. A single horizontal structure path carries O → MLP → H → set attention → H-tilde → rows/pointer. Dimension labels have a secondary typographic weight.
4. The original H bypass remains distinct. The predicted row/order pair enters Pose heads before any coordinates are emitted. The bottom output is labeled **Neural proposal**, separating it from the subsequent rules/search/refinement summarized in the unchanged upper panels and manuscript caption.
5. Four right-hand outputs are aligned, operator treatment is consistent, and decorative matrix miniatures are removed while their dimensions are retained. Color remains redundant with labels and arrow direction.

Main operator labels are approximately 9.3–9.4 pt, the title and principal tensor symbols 10 pt, and secondary labels approximately 7.6–8.5 pt. Compared with the current mostly 10-pt labels, the new hierarchy gives supporting dimensions less visual weight. Both the enlarged comparison and the figure at publication width were inspected. The proposal is structurally clearer, but its more explicit operator framing is an aesthetic choice for the user's review.

## Implementation fidelity and level of detail

- `src/knolling_model/arrangement/goal_encoder.py:10–40`: MiniLM tokens are projected from 384 to 128; seven learned queries produce Z with 7 × 128 entries. Z supports the global 13-dimensional goal. The grounder uses projected object features and all language slots. Seven slots do not imply seven global semantic fields.
- `src/knolling_model/arrangement/contracts.py:48–53`: O has 12 channels, including the two object sizes S. These are estimated/refined sizes, not ground truth.
- `src/knolling_model/experiments/relation_pose.py:16–22,52–53`: the adopted object-wise hidden encoder maps 12 → 128 → 128. Its original relation-pose prediction head is not shown as an active module.
- `src/knolling_model/experiments/set_context.py:28–40,43–58`: four-head context uses H, g and S; contextual features feed the selector, and original H remains the placement input.
- `src/knolling_model/experiments/row_layout.py:238–274`: row/pointer selection uses contextual features and g. Pose computation uses original H, g, sizes and predicted structure. Within the grouped pose operator, the angle head reads H/g, while coordinate construction additionally uses S/r/order.
- `src/knolling_model/experiments/size_refinement.py:299–346`: command consistency/routing, bounded structure search and geometric repair surround the neural prediction. This proposal intentionally focuses on the learned factorization already assigned to panel (d), without expanding it into a second full-pipeline diagram.

A separate technical reviewer checked the rendered proposal against these dependencies. A separate visual reviewer assessed the comparison. No model, checkpoint or dataset was loaded or changed, and no inference or training was run.

## Preservation

`baseline_manifest.json` records 263 pre-existing canonical files. `review.json` reports their final integrity, the unchanged upper panels, equal canvas footprint, and diagram text-bound checks. No manuscript build or source-package update was performed. Integration is deferred until the user has reviewed the comparison.

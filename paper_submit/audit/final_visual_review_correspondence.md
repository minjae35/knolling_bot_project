# Independent final visual review — 2026-11-15

Reviewed the twelve rendered pages in `build/visual_review/correspondence_round1/` individually, including Figures 1–7 at their actual page scale. The reviewed PDF SHA-256 is `787c0f8978ef2bda56b7086ff4cc181ec3ff7abe6a2bb4adc3844fb64776c16d`. This is a review record, not a model evaluation.

## Result

No blocking clipping, overlap, or page-layout defects were found. The main paper occupies pages 1–5, references page 6, and appendices pages 7–12. The original two-line title and stacked author/department/university are retained. There are no running headers. All text remains within the PDF page boundaries.

## Visual assessment

- Page 1: the matched Blender pair provides an immediate task example; IDs and long-axis bars make object correspondence recoverable. Caption explicitly separates presentation renders from the RGB consumed by the model.
- Page 2: representation-based architecture is materially clearer than the prior rounded-box layout. Actual input/output thumbnails anchor the symbolic object matrix and row/pointer representation. The object-conditioned language route, held/no-op bypass, and original-feature pose branch are visible. The seven-point secondary labels remain readable at page scale; no labels are clipped.
- Pages 3–4: the protocol and result tables use consistent rules and typography. The success versus object-count plot and error comparison are legible, with denominators and overlapping-tag semantics supplied by the caption and text.
- Page 5: paired IDs, orientation bars, displacement arrows, and an eight-object numerical ledger jointly explain translations and signed rotations. The source-panel arrows cross near several objects, so their interpretation relies on the retained IDs and numerical ledger. This is an inherent density tradeoff, not a missing identity or fabricated path. The figure and caption explicitly state that arrows are displacement vectors rather than robot trajectories.
- Pages 6–9: references, implementation/loss details, and training/statistical tables fit without clipping. The selector curve distinguishes local accuracy from complete-task success.
- Page 10: full/partial/no-op panels retain shared framing. Failure callouts identify the actual region and state its margin. The partial case is explicitly marked borderline, and its deficit is below image resolution.
- Page 11: the two failure case studies are particularly effective. The actual Blender crop is paired with a measured value, allowed interval, and failure margin. The partial ruler declares its expanded numerical scale, and the text correctly avoids implying a collision or micrometer robot precision.
- Page 12: evidence paths, compute scope, rebuild instructions, and assistance disclosure are legible. The source bundle scope is distinguished from reproduction of training.

## Technical correspondence checked

Read `src/knolling_model/experiments/set_context.py`, `row_layout.py` and `relation_pose.py`, alongside the method audit. The diagram correctly depicts a 12-to-128 relation feature route, one four-head object attention block conditioned by 13-dimensional goal and sizes, context-fed row/pointer selection, and pose prediction from the original features. `SetArrangement.forward` supplies `selector=mixed` while passing the original `hidden`; the diagram follows this implementation rather than the less precise module comment.

Read the motion-figure generator: translations derive from stored source/target centers, angular differences wrap modulo pi, orientation bars follow the same world poses, and projection uses the saved Blender camera basis. No model inference, pose editing, or pixel retouching occurs in this annotation step.

No changes to manuscript content, figures, models, datasets, or checkpoints were made by this review. Only this audit note was written.

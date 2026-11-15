# English arrangement conventions

These conventions explain the v1.3 labels produced by `balanced_rows_v1.3`. They do not change that teacher or the active grammar `english_rows_v1`. The 20,000-scene training set and the v2 holdout use a later teacher, `continuous_rows_v3`, and they store the longer sentences emitted by `english_instruction`. The compact examples at the end of this note are proposals. They are not instructions in either published set, and the canonical-text audit does not accept them in place of the current sentences.

## Meaning that must survive shorter wording

- The image is the current overhead observation of the calibrated 600 × 400 mm workspace. Left and right are image left and right. The fixed usable regions in metres are full `(0.02, 0.02, 0.58, 0.38)`, left `(0.02, 0.02, 0.28, 0.38)`, and right `(0.32, 0.02, 0.58, 0.38)`, in `(xmin, ymin, xmax, ymax)` order. The metric scale comes from this fixed setup. It cannot be inferred from arbitrary uncalibrated RGB.
- An object reference is grounded in the current source image. Color selects all objects of that color. Leftmost refers to the object's center and must be unambiguous. Hidden identifiers and earlier observations are not instruction inputs.
- Rows run left to right and are read from top to bottom. Horizontal and vertical describe each object's long side, not whether rows become columns. A row may contain one object. Objects share the row's vertical center, and rows share the arrangement block's horizontal center. Equal column alignment and equal object counts per row are not required.
- Size means the area of the object's oriented bounding rectangle, not visible colored-pixel area, major-axis length, volume, or a shape category. Read the requested smallest or largest order left to right, then top to bottom. With color grouping, apply this order separately within each color.
- Color grouping means single-color rows and consecutive rows of each color. A request that does not group by color stays ungrouped. When the v1.3 teacher makes a new arrangement, its color-block order is red, green, blue, omitting absent colors. That is a labeling convention. The task validator permits other contiguous color-block orders.
- A requested gap `g` means horizontal edge-to-edge bounding-rectangle spacing within a row and vertical edge-to-edge spacing between row bands. A row band extends from the bottom to the top of the tallest bounding rectangle in that row. Gaps between consecutive different-color bands are `2g` when color grouping is requested. The distance between every pair of individual shapes is not necessarily `g`, especially across rows with different object sizes.
- Partial requests apply arrangement rules only to the selected objects. All other objects keep their exact poses. Safety applies to the whole scene. Final bounding-envelope clearance and table-boundary margin remain at least 10 mm under the existing numerical tolerances. A selected singleton has no pairwise sorting, grouping, or gap relation to enforce. Those clauses do not authorize moving locked objects.
- Keep-all preserves every pose, including a safe scattered scene. An ordinary arrangement instruction also yields no-op when its conditions and final safety are already satisfied. No-op takes priority over choosing a different but slightly preferred arrangement. Reissuing a positional reference in a new observation can select a different object.

## How the v1.3 teacher chooses one target

The words describe acceptable arrangements, not a unique coordinate list. `balanced_rows_v1.3` chooses one label from visible geometry and the structured request. New full blocks are centered in the requested region. The finite sequential and balanced row candidates are ranked by `max(block_width / region_width, block_height / region_height)`, then area, row count, and the observable-order template key. Partial ranking puts mean squared selected-center translation first, then that same shape ordering. It uses the existing feasible 5 mm translation lattice. It is not unrestricted continuous minimum movement. `continuous_rows_v3`, the teacher of the published sets, replaces that lattice with continuous envelope projection. Existing valid arrangements are preserved before this search.

Object-ordering ties use major dimension, minor dimension, color, and source center `(y, x)` after signed bounding-rectangle area. Candidate and translation ties follow the existing rounded lexicographic keys. No asset identifier or random seed is a layout preference. The numerical contract is `teacher_balanced.py` for the v1.3 policy and `teacher_continuous.py` together with `teacher_v3.py` for the published sets. The 130 pilot samples have no conflicting target labels for an identical source RGB and an identical instruction string.

The network has to learn these conventions together with perception. A reader cannot recover every teacher coordinate from the English sentence alone. Success evaluation therefore keeps semantic and geometry success separate from teacher-coordinate error. Inference does not call an external planner.

## Compact wording proposals

With the conventions above fixed, shorter sentences could take this shape:

| Existing sample | Proposed compact instruction |
| --- | --- |
| `scene_0000_primary` | Arrange all objects on the right in rows, smallest first, with their long sides horizontal and 15 mm gaps. Do not group by color. |
| `scene_0004_primary` | Arrange only the red objects on the left in single-color rows, smallest first, with their long sides horizontal and 15 mm gaps. Keep everything else fixed. |
| `scene_0075_primary` | Leave every object exactly where it is, without rotating it. |

The second example keeps the full request even when the observed selection is a singleton. It must not become a different instruction when another red object appears in a later scene. Dropping a vacuous clause is safe only when the remaining request and the declared defaults still identify the task.

Activating shorter text would require a new grammar version, the same grouping, order, axis, gap, region, selection, and lock semantics, and a check of every label. A paraphrase set would also have to stay inside each scene's existing split. Slot substitutions are not paraphrase coverage. None of that text is in the published training set or the v2 holdout.

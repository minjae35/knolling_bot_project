# English arrangement conventions

These conventions explain the existing v1.3 labels. They do not change the
Teacher, dataset or supported active grammar (`english_rows_v1`). In particular,
the compact examples below are proposals for a later version, not instructions
already represented by this pilot or accepted by its canonical-text audit.

## Meaning that must survive shorter wording

- The image is the current overhead observation of the calibrated 600 × 400 mm
  workspace. Left/right are image left/right. The fixed usable regions in metres
  are full `(0.02, 0.02, 0.58, 0.38)`, left `(0.02, 0.02, 0.28, 0.38)` and right
  `(0.32, 0.02, 0.58, 0.38)`, in `(xmin, ymin, xmax, ymax)` order. The metric scale
  comes from this fixed setup; it cannot be inferred from arbitrary uncalibrated RGB.
- An object reference is grounded in the current source image. Color selects
  **all** objects of that color; leftmost refers to the object's center and must
  be unambiguous. Hidden IDs and earlier observations are not instruction inputs.
- Rows run left to right and are read from top to bottom. Horizontal/vertical
  describes each object's **long side**, not whether rows become columns.
  A row may contain one object. Objects share the row's vertical center; rows
  share the arrangement block's horizontal center. Equal column alignment and
  equal object counts per row are not required.
- Size means the area of the object's oriented bounding rectangle, not visible
  colored-pixel area, major-axis length, volume or a shape category. Read the
  requested smallest/largest order left to right, then top to bottom. With color
  grouping, apply this order separately within each color.
- Color grouping means single-color rows and consecutive rows of each color.
  Do not silently turn "do not group by color" into grouped labels. When making
  a new arrangement, the Teacher's deterministic color-block order is red,
  green, blue, omitting absent colors. This is a labeling convention; the task
  validator permits other contiguous color-block orders.
- A requested gap `g` means horizontal edge-to-edge bounding-rectangle spacing
  within a row and vertical edge-to-edge spacing between **row bands**. A row
  band extends from the bottom to the top of the tallest bounding rectangle in
  that row. Gaps between consecutive different-color bands are `2g` when color
  grouping is requested. Distance between every pair of individual shapes is
  not necessarily `g`, especially across rows with different object sizes.
- Partial requests apply arrangement rules only to the selected objects. All
  other objects keep their exact poses. Safety applies to the whole scene.
  Final bounding-envelope clearance and table-boundary margin remain at least
  10 mm under the existing numerical tolerances; shorter text must not relax them.
  A selected singleton has no pairwise sorting/grouping/gap relation to enforce;
  those clauses do not authorize moving locked objects.
- Keep-all preserves every pose, including a safe scattered scene. An ordinary
  arrangement instruction also yields No-op when its conditions and final
  safety are already satisfied. No-op takes priority over choosing a different
  but slightly preferred arrangement. Reissuing a positional reference in a new
  observation can select a different object; it is not unconditional No-op.

## Deterministic target choice is an additional learned convention

The words describe acceptable arrangements, not a unique coordinate list.
`balanced_rows_v1.3` chooses one consistent label using visible geometry and the
structured request. New Full blocks are centered in the requested region. The
finite sequential/balanced row candidates are ranked by
`max(block_width / region_width, block_height / region_height)`, then area,
row count and observable-order template key. Partial ranking puts mean squared
selected-center translation first, then that same shape ordering; it uses the
existing feasible 5 mm translation lattice. It is not unrestricted continuous
minimum movement. Existing valid arrangements are preserved before this search.

Object ordering ties use major dimension, minor dimension, color and source
center `(y, x)` after signed bounding-rectangle area. Candidate and translation
ties follow the existing rounded lexicographic keys; no asset ID or random seed
is a layout preference. See `teacher_balanced.py` and the specification for the
complete numerical contract. The 130 current samples have no conflicting target
labels for an identical source RGB and identical instruction string.

The network must learn these conventions along with perception. A person cannot
recover every exact Teacher coordinate from ordinary English alone. Success
evaluation should therefore retain semantic/geometry success separately from
Teacher-coordinate error. This is not permission for an external inference planner.

## Compact wording proposals — not activated

With the above conventions explicitly established, examples could be:

| Existing sample | Proposed compact instruction |
|---|---|
| `scene_0000_primary` | Arrange all objects on the right in rows, smallest first, with their long sides horizontal and 15 mm gaps. Do not group by color. |
| `scene_0004_primary` | Arrange only the red objects on the left in single-color rows, smallest first, with their long sides horizontal and 15 mm gaps. Keep everything else fixed. |
| `scene_0075_primary` | Leave every object exactly where it is, without rotating it. |

The second example keeps the full request even though the observed selection is
a singleton; it must not become a different instruction when another red object
appears in a later scene. Removing vacuous clauses is safe only if the remaining
request and declared defaults still identify the intended task.

Before activating shorter text, version its grammar, preserve the explicit
grouping/order/axis/gap/region/selection/lock semantics, and recheck all labels.
Add a small set of genuinely different English paraphrases within each scene's
existing split; never treat template-slot substitutions as paraphrase coverage.
This is useful before claims about natural user instructions, but is not a
prerequisite for the initial S0 memorization diagnostic. No such new data or
language migration is part of the final quality review.

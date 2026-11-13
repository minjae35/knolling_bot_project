# Phase 1 decisions and scope

Source of truth: final Technical Specification v1.2 in the conversation.
The new task explicitly authorizes M0–M8 implementation and verified milestone
commits/pushes to main. No previous project implementation is reused.

Environment checked: existing Conda `knolling`, Python 3.12.16; local Blender
5.1.0 (adfe2921d5f3). Host dependencies are pinned after installation.

## Rendering backend choice

Use Blender's own Python API directly, launched by the Conda-hosted builder.
Blender/BlenderProc is authorized by the Phase 1 request. This is a documented
departure from v1.2's BlenderProc wrapper, preserving the specified camera,
geometry, Cycles rendering, labels and outputs. No Blender Python packages are
installed into the Conda environment and no second Blender download is needed.

## Hard contracts

- No-op source scenes already satisfy final geometry and the requested task;
  targets exactly equal sources and relation labels are empty.
- Color groups are independently split into rows; never fill a color's short
  row with objects from another color.
- Partial locked poses are exact copies, not merely within evaluation tolerance.
- Teacher search exhaustion is generation_failure, not a proof of infeasibility.
- The independent validator uses polygon geometry, not the teacher's placement
  computations. Target matching cannot replace source identity.
- Split decisions precede rendering; all descendants stay in their family split.

## Blender rendering verification

RGB uses Cycles CPU, 32 samples and a fixed per-family seed. Instance IDs use
the same meshes and camera in Workbench flat object colors with anti-aliasing
disabled, then losslessly convert red-channel IDs into uint16 PNG labels.
This avoids mixed IDs at silhouette edges. It does not approximate masks with
bounding rectangles. Geometry-to-mask IoU must be >= 0.98, centroid error <= 1 px,
and interior RGB dominant-color agreement >= 99%. More than 1% clipped white
pixels rejects a render. The actual Blender integration test also checks
byte-identical repeated renders on the installed version/platform.

API references: [SceneDisplay anti-aliasing](https://docs.blender.org/api/5.0/bpy.types.SceneDisplay.html)
and [camera projection](https://docs.blender.org/manual/en/2.90/render/cameras.html).
Runtime properties were also checked directly on installed Blender 5.1.0.
Bitwise reproducibility is scoped to a fixed Blender build, device and platform.
Blender's variable PNG timestamp/render-time metadata is removed by the host;
RGB values remain in sRGB. Initial pixel equality passed, while raw-file equality
exposed these stamps. Commit f372f76 was inadvertently pushed before reading
that additional test result; the subsequent fix is additive, with no history
rewrite, and the full integration suite must pass before its commit.

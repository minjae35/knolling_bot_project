# Matched Blender publication renders

Ten final images and editable `.blend` scenes are included. Each image is 1920×1280, rendered with Blender 5.1.0 / Cycles CPU, adaptive sampling up to 128 samples, and denoising.

- `teaser6_before` / `teaser6_after`: six-object passing validation example from `audit/evidence/full67_choice.json`; used in Figure 1.
- `full_before` / `full_after`: eight-object passing validation example from `audit/evidence/full8b_choice.json`; also the geometry source for Figure 4.
- `smoke_before`, `smoke_full`, `smoke_partial`, `smoke_no_op`: one validation family and three saved predictions; used in Figure 6.
- `smoke_full_overhead`, `smoke_partial_overhead`: the same failing predictions under a calibrated overhead camera; used in Figure 7.

Each comparison uses a fixed camera, lighting, background, materials, and renderer. The six-object teaser has its own matched presentation setup. Meshes extrude original footprint polygons to their recorded heights; no object is moved, scaled, bevelled, or rearranged for appearance. The tabletop/background are presentation surfaces.

These are presentation renders, not model input images. Original inference RGB and saved predictions are bundled under `audit/evidence/`. The overhead diagnostic camera spans x=[−0.02, 0.62] m and y=[−0.0133333, 0.4133333] m; top-row pixels have larger world y. Annotations are computed from world-space bounding envelopes, not estimated from perspective pixel distances.

See the submission README for rebuild commands. `audit/blender_pose_checks.json` and `audit/teaser6_provenance.json` record independent saved-file mesh/pose checks. No inference, training, or robot execution is required.

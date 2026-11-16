# Matched Blender publication renders

The current paper uses thirteen Blender 5.1.0 / Cycles CPU images with adaptive sampling (128-sample maximum) and denoising: eleven at 1920 × 1280, plus two dedicated 2040-pixel-wide closeups. Each has a complete editable `.blend` scene.

- `teaser6_before.png`, `teaser6_after.png`: six-object opening example from `audit/evidence/full67_choice.json`, rerendered to match the approved Figure 6 camera, lighting, tabletop, and rendering settings.
- `full_before.png`, `full_after.png`: saved successful eight-object validation example from `audit/evidence/full8b_choice.json`.
- `smoke_before.png`, `smoke_full.png`, `smoke_partial.png`, `smoke_no_op.png`: the four approved Figure 6 inputs. These PNGs and their Blender scenes are frozen; do not overwrite them.
- `smoke_full_overhead.png`, `smoke_partial_overhead.png`: the same recorded predicted layouts through a calibrated overhead camera, for gap/clearance details.
- `detail_full_workspace.png`: an overhead view of `full_after.blend`, with the original meshes, poses, materials, lights and world unchanged; 1920 × 1280.
- `detail_row_gap.png`: dedicated 2040 × 765 overhead crop of `smoke_full_overhead.blend`, x = [0.17, 0.45], y = [0.115, 0.22] m.
- `detail_clearance.png`: dedicated 2040 × 885 overhead crop of `smoke_partial_overhead.blend`, x = [0.13, 0.31], y = [0.162, 0.2400882353] m.

All eight oblique images share the same camera and illumination. The two overhead images share a second camera with the same scene/light/material settings. Original project footprints are extruded to their actual heights; object geometry, material palette/roughness, and recorded poses are preserved. The tabletop is a presentation surface.

These are presentation renders, **not the RGB images supplied to inference**. Original inference inputs remain in `audit/evidence/`. No new inference, training, evaluation, or robot execution was performed.

The overhead camera is centered at (0.3, 0.2) m with orthographic width 0.64 m. At 3:2 aspect, image extents are x = [−0.02, 0.62] m and y = [−0.0133333333, 0.4133333333] m. Top-row pixels have larger y. Pixel centers use a half-pixel offset.

## Provenance and integrity

- `audit/blender_provenance.json`, `audit/blender_pose_checks.json`, and `renders/render_metadata.json`: original eight-render hashes, settings, saved-file mesh/pose checks.
- `audit/teaser6_provenance.json`: current six-object pair, including independent saved-file transform/mesh checks and hashes.
- `audit/figure_style_comparison.json`: confirms that the teaser uses Figure 6's camera/light/render settings.
- `audit/figure6_frozen.json`: exact hashes of the protected Figure 6 exports and inputs, plus its unchanged LaTeX block.
- `audit/detail_render_provenance.json`: three new camera-only views; original and derived scene/image hashes, calibrated extents, camera settings and resolution, and equality checks on all noncamera transforms, mesh vertices/faces, materials, lights, world and view settings after reopening each saved file.

The supplied finished renders are sufficient for every paper/figure rebuild. `scripts/render_publication.py` preserves the original scene-construction source for inspection; running its complete render batch would overwrite approved Figure 6 inputs and is not part of the current rebuild. `scripts/render_teaser.py` constructs only the six-object opening pair. `scripts/render_detail_views.py` writes only `detail_*` scenes/images, preserving its source files and checking Figure 6 before and after. These optional rendering scripts use eight CPU threads and do not require a GPU or checkpoints.

From the paper directory, verify that Figure 6 is preserved:

```bash
.venv/bin/python scripts/preserve_figure6.py
```

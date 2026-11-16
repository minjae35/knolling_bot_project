# Knolling Bot v2: Language-Guided Object Arrangement

**Author:** Minjae Bang, Department of Computer Science, Columbia University.

The submission file is **knolling_bot_v2_paper.pdf**: **5 pages of main text, 1 page of references, and 6 appendix pages**. It contains 7 figures, 6 tables, and 9 cited research papers. The official ICLR 2026 style and BibTeX style are preserved unchanged; conference acceptance/review branding is omitted for this course paper.

This directory is the reviewed final revision. Earlier drafts in the parent directory are preserved separately.

## Deliverables

- `knolling_bot_v2_paper.pdf`: final paper, references, and appendix.
- `main.tex`, `appendix.tex`, `references.bib`: editable manuscript and bibliography.
- `figures/`: seven figures in vector PDF, editable-text SVG, and 300-dpi PNG.
- `renders/`: ten full-resolution Blender PNGs, ten editable `.blend` scenes, and camera/pose metadata.
- `scripts/`: figure sources, statistical reaggregation, Blender scene construction, PDF review, verification, and packaging.
- `audit/evidence/`: frozen score logs, recorded predictions, source object geometry, and training summaries.
- `audit/`: provenance, numerical checks, independent implementation/experiment reviews, and final visual review.
- `template/`: official ICLR files with upstream commit and SHA-256 hashes.
- `fonts/`: bundled figure fonts and redistribution licenses.
- `knolling_bot_v2_source.zip`: portable source, figure, Blender, evidence, and PDF package.

## Compile

A TeX Live installation with `latexmk`, pdfLaTeX, BibTeX, and the standard/recommended/extra LaTeX packages is sufficient:

```bash
bash build.sh
```

The bundled figure PDFs make compilation independent of Python, Blender, model weights, datasets, and a GPU. On Overleaf, choose `main.tex` as the main document and pdfLaTeX as the compiler.

## Rebuild analysis and figures

Use Python 3.12 in a separate environment; these dependencies do not change the model environment.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-paper.txt
mkdir -p build
.venv/bin/python scripts/audit_results.py > build/statistics.log
.venv/bin/python scripts/make_figures.py
bash build.sh
.venv/bin/python scripts/verify_submission.py
.venv/bin/python scripts/render_pages.py --round final
.venv/bin/python scripts/package_submission.py
```

Packaging extracts a fresh copy, regenerates all seven figure previews, recompiles the manuscript, and compares both extracted text and rendered page pixels with the final PDF. It does not rerun inference or training. Historical Python files under `audit/evidence/` are evidence snapshots, not rebuild entry points.

## Figure sources

| Figure | Source | Evidence |
|---|---|---|
| 1. Before/after teaser | `scripts/teaser_figure.py` | Six-object saved validation prediction; `full67_choice.json` |
| 2. Architecture | `scripts/architecture_figure.py` | Audited inference and feature routing; `audit/final_architecture.json` |
| 3. Paired outcomes and diagnostics | `scripts/quantitative_figure.py` | Frozen v1/v2 sealed score logs |
| 4. Displacement and rotation | `scripts/motion_figure.py` | Eight-object saved validation poses; `full8b_choice.json` |
| 5. Selector learning curve | `scripts/make_figures.py::development` | Saved eight-epoch selector history |
| 6. Full/partial/no-op | `scripts/publication_figures.py::modes` | Three predictions on one validation scene |
| 7. Failure margins | `scripts/failure_figure.py` | True bounding-envelope geometry at saved predicted poses |

Install the bundled Liberation Sans fonts for consistent SVG editing. Scientific plots use fixed physical dimensions; the architecture's labels are at least 7 pt. Token glyphs are schematic, not recorded activations.

## Blender sources

The render pairs preserve original project footprint meshes, heights, colors, and recorded poses. Matched views share camera, lighting, background, materials, and render settings. Presentation renders are not inference inputs. The original overhead RGB is shown in the architecture diagram.

To regenerate the renders with Blender 5.1.0:

```bash
blender --background --threads 8 --python scripts/render_publication.py
blender --background --threads 8 --python scripts/render_teaser.py
blender --background --threads 2 --python renders/check_saved_blends.py
blender --background --threads 2 --python scripts/render_teaser.py -- --check-only
```

Both renderers use Cycles on the CPU, with 1920×1280 output, denoising, and a 128-sample maximum. Rendering is optional: complete editable scenes and final PNGs are already included. Independent saved-scene checks compare every mesh vertex and object pose to frozen evidence; residuals below 1e-7 reflect Blender float32 storage. These are visualization-fidelity checks, not claims of robotic precision.

## Evidence and limitations

The principal result is **489/640 → 540/640** strict synthetic-scene successes: **+8.0 percentage points**, paired family-bootstrap 95% interval **+4.8 to +11.6 points**. All five instruction pairs from each scene family remain together during resampling. The comparison evaluates an inference-refinement bundle over the same learned base; it is not a factorial ablation.

The project aimed at end-to-end learning. The achieved inference interface maps RGB and English to target poses through staged neural modules and explicit rules. Restricted joint tuning includes a stop-gradient boundary, and the broader joint pilot was rejected. No full-stack joint-training success, real-image benchmark, or physical robot execution is claimed.

The sealed score logs do not contain predicted poses. Qualitative figures therefore use clearly identified, selected validation predictions, whose per-record checkpoint digests were not retained. They illustrate behavior, not an additional benchmark. The final set is disjoint from the documented 20,000-pair corpus; this is not an asset-novelty claim against every historical dataset.

Original source code, models, checkpoints, and datasets were not changed. This package reproduces the paper and its analysis, not the unavailable historical training pipeline. Final checks are recorded in `audit/submission_checks.json`, `audit/portable_build_check.json`, and `audit/final_review.md`.

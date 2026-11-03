# Knolling dataset pipeline

New implementation for `minjae35/knolling_bot_project`, based on the finalized
v1.2 specification in the project conversation. Scope: M0–M8 only. No neural
network, training pipeline, or robot simulation.

Use the existing Conda environment `knolling` (Python 3.12):

```sh
conda run -n knolling python -m pip install -r requirements-lock.txt
conda run -n knolling python -m pip install --no-deps -e .
conda run -n knolling python -m pytest -q
```

All geometry is in meters and radians. Pixel and normalized coordinates are
explicit derived values. Generated datasets and QC reports belong under the
Git-ignored `artifacts/` directory.

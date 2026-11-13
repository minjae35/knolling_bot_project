# GPU preflight and live S0 monitoring

Use the Conda environment `knolling` at `/venv/knolling`, Python 3.12, with the
pinned neural requirements and torch 2.9.1+cu128 / torchvision 0.24.1+cu128.
The CUDA driver is host-provided and must not be replaced. Public pretrained
assets use `HF_HOME=/workspace/cache/huggingface` and
`TORCH_HOME=/workspace/cache/torch`; keep those variables consistent on resume.

`scripts/gpu_preflight_s0.py --report-dir NEW_DIRECTORY` verifies the complete
pilot, all S0 samples, the pinned MiniLM revision/LFS SHA-256 and ResNet hash,
full-resolution CUDA forward/backward, actual configured effective-batch AdamW
updates, finite gradients/parameters/optimizer state, and exact checkpoint
load/next-update replay. It records allocated/reserved peak VRAM including Adam
state. Its three optimizer calls are disposable checks on a separate model;
the actual S0 run starts fresh from the configured seed and pretrained weights.
It never starts training automatically. Inspect `report.json` for `valid:true`
and run the regression suite before starting the authorized S0 run.

The trainer now displays one tqdm line updated at most about once a second:
optimizer update / configured maximum, percentage, elapsed/remaining time,
current loss, cumulative average update loss, GPU reserved VRAM, and latest
Count/target-P95/gate diagnostics. Updates count `optimizer.step()` calls, not
accumulation micro-batches. No training hyperparameter or model computation
changes. Existing `metrics.jsonl` and per-evaluation JSON remain the scientific
record; `progress.json` is an atomically replaced status snapshot. Cumulative
loss totals and elapsed time are included in checkpoints so resumed runs keep
their update count and average. Resume overhead before initialization is not
counted as training time; checkpoint-free crash tails are replayed normally.

Run inside a named tmux session with its working directory set to the repo.
Capture its pane to a separate `terminal.log` using `tmux pipe-pane`, and keep
`metrics.jsonl`, checkpoints, `evaluation-latest.json`, `progress.json`, resolved
requirements and provenance under the run directory. Each run must use a new
directory; the trainer refuses to overwrite an existing run. Keep the tmux pane
open after Python exits and record its exit status outside the trainer's new
output directory. tmux protects against SSH disconnection, not host destruction.

For the session name supplied in the run report:

```sh
tmux attach -t SESSION_NAME
# Detach without stopping training: Ctrl-b, then d.
tmux list-sessions
tmux capture-pane -p -t SESSION_NAME:0.0 | tail -8
cat RUN_DIRECTORY/progress.json
cat RUN_DIRECTORY/result.json  # exists only after normal completion
```

Resume a checkpoint into a new output directory, using the same code and cache:

```sh
conda activate knolling
export HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch
python -m knolling_model train --config configs/s0.json --device cuda \
  --resume PRIOR_RUN/last.pt --run-dir NEW_RUN --start-training
```

Do not proceed to S1, generate more data, or alter architecture/loss/configuration
as part of this S0 run. A failed or budget-exhausted S0 run requires diagnosis.

# Main consolidation — 2026-11-08

## Selected implementation and history

The primary checkout now develops on `main`. Arrangement's accepted
`codex/v4-neural-keep` history is retained, including common Dataset/Teacher,
Goal Encoder, evaluation and checkpoint fixes. Five independent Perception
commits (`47b9735` through `c043144`) add the reviewed source-only model and its
results. They change no Arrangement graph or loss.

| Previous branch | Disposition |
|---|---|
| `codex/v4-neural-keep` (`6debc61`) | Accepted model and history integrated |
| `codex/perception` (`c043144`) | Independent Perception implementation integrated |
| `codex/v4-relative-task` (`1965663`) | Already ancestor of accepted Keep model |
| `feat/neural-recovery` (`a037610`) | Already ancestor; restoration/evaluation preserved |
| `fix/s0-bounded-repairs` (`dbd68d0`) | Already ancestor; safety fixes preserved |
| `codex/v4-pairwise-xy` (`73c3dcc`) | Three failed experiment commits excluded from main |

All local branches matched their remote tips before integration; no unpushed
branch commits or dirty worktree edits were present. The Perception chat's last
turn was interrupted and idle; no running process had a worktree as its cwd.
The detached Telegram worktree (`9b909b3`) was already preserved by the existing
`archive/telegram-read-only-commands` tag. Its live services use independent
copies under `/opt/knolling-notify`, with no links into the worktree. No notifier
or listener was changed, stopped or restarted.

Before deletion, refs, worktrees, 3,562 local artifact files and complete Git
history were inventoried under `artifacts/repository_consolidation/20261108/`.
`branches-before.bundle` was verified and retains even excluded Pairwise
history. It can be inspected/restored without adding old code to main.

## Assets preserved

No model or dataset binary was added to Git, no Checkpoint/data was deleted,
and no force push or hard reset was used. Perception's 78 artifact files
(about 1.7 GiB) were relocated from its worktree to `artifacts/perception/`;
every file's SHA-256 matched before and after relocation. Historical result
JSONs retain their original recorded paths; the registry provides current paths.

| Asset | Current path / integrity |
|---|---|
| Arrangement Best | `artifacts/recovery_runs/recovery_v4_neural_keep_once_20261108T101334Z/checkpoint-000400.pt` |
| Arrangement SHA-256 | `d92ad1ffcabf1842c7d805e8a8445b3dfca7cf3bfad05aa92318db3c010ba3d9` |
| Perception Best | `artifacts/perception/gpu-once-20261108/training/best.pt` |
| Perception SHA-256 | `c02dc5e9501a39cfcb9d1de8d512ace765050d675155e66bbde0ec2889105491` |
| Perception Last | `artifacts/perception/gpu-once-20261108/training/last.pt` |
| Perception Last SHA-256 | `d41bbbbf0564238ce027cb3f6930d790824f2223f5aaed2ef40f23540ce477ab` |
| Dataset / annotations | `artifacts/pilot_v1_3_en/`, `artifacts/recovery_v3_verification_r1/` |
| Shared checkpoint blobs | `artifacts/checkpoint_store/`, unchanged; required for lightweight restores |

All other Recovery/S0 checkpoints, optimizer states, journals, anchors, RGB,
predictions, PASS/FAIL evidence and experiment logs remain preserved.

## Code cleanup

- `configs/current_models.json` is the single pinned model registry.
- `knolling_model.current` composes existing Perception and Arrangement forward;
  it adds no architecture, planner, target input or training step.
- Five historical one-off scripts moved to `archive/experiments/scripts/`:
  S0 GPU preflight, RT-DETR preflight, SVD verification, initial Recovery CPU
  verification, and the completed Oracle Keep intervention.
- Three obsolete approval/extension configuration examples moved to
  `archive/experiments/configs/`. Configurations still used by restoration,
  dataset builders or regression tests remain at their original paths.
- Root README now documents current models and main-only development. The old
  S0 workflow is retained in `docs/legacy-s0-readme.md` as historical material.
- AR/v3 constructors and shared primitives are compatibility code, retained to
  avoid breaking old checkpoints/tests. Pairwise modules/configs were not merged.
  Duplicated model binaries were not introduced by integration.

## Verification protocol

Run CPU regression tests in `knolling`. Actual pinned checkpoints are loaded
with strict keys and SHA-256/code/fingerprint checks. Read-only CUDA evaluates
Arrangement's 76 scenes/35 pairs and Perception's 10 Validation sources, then
runs real RGB→predicted physical objects→language→Neural Keep forward.
No optimizer step or GPU training is part of this verification.

The first Arrangement replay reproduced all task successes but failed a strict
bitwise pose check because it omitted the original token-only FrozenTextCache.
Repeated commands were re-encoded at differing padded lengths. The canonical
replay now uses exactly the original cache warm-up order and shapes. Neither
model parameters nor tolerance/gate definitions were altered. The initial
failure log is preserved; branch deletion waits for canonical replay and tests.

Perception's existing eight-object limit is enforced: overflow is an explicit
rejection, never a GT count filter. Interface compatibility is distinct from
model quality. Arrangement's 40/76 and 7/35 remain training-set oracle-source
metrics; Perception does not yet meet source precision or End-to-End task gates.

## Verification result

- Full CPU suite: **481 passed, 95 skipped**, 350.78 seconds in `knolling`.
  Skips retain the existing optional/CUDA opt-in test policy.
- Actual CUDA evaluation: **zero optimizer steps**, model-state hashes unchanged.
- Arrangement: **40/76**, **7/35**, all 76 predicted pose dictionaries exactly
  equal to the saved Best using the original evaluation protocol.
- Perception: all saved Validation metrics exactly equal after relocation and
  reload; count accuracy **4/10**, mAP50:95 **0.9450662903967599**.
- Real RGB bridge: **8/10** Validation inputs complete finite Arrangement
  forward; **2/10** are explicitly rejected for the pre-existing capacity
  overflow. No prediction is dropped to manufacture a compatible count.
- All 3,562 pre-existing local artifact files retained identical sizes/mtimes;
  the 78 relocated Perception files additionally passed full SHA-256 comparison.

Logs: `artifacts/repository_consolidation/20261108/pytest-cpu.log`,
`canonical-forward/result.json`, `canonical-forward/arrangement.json`,
`canonical-forward/perception.json`, and `assets-preservation.json`.
The registry contains no checkpoint binary; all large assets remain ignored.

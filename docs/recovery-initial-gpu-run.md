# Recovery initial GPU run

## Authorization and scope

The user explicitly authorized initial GPU training after the timeout fix, with
bounded stages and approval required before any extension. This authorization
supersedes the earlier training hold; it does not authorize automatic retries or
larger budgets. Each launch receipt binds the final code commit, config digest
and verified dataset fingerprint to that user authorization.

| User stage | Driver stage | Budget (first limit reached) | Batch | Precision |
|---|---|---|---|---|
| 1 Arrangement | oracle_layout | 500 updates / 600 s | 8 | BF16 Transformer, FP32 pose heads |
| 2 Perception | perception | 1,000 / 2,700 s | 4 | FP32 |
| 3 End-to-End, connection | predicted_layout | 1,000 / 1,800 s | 8 | FP32 |
| 3 End-to-End, bounded joint tuning | joint | 300 / 900 s | 4 | FP32 |

The existing measured stage gates are unchanged. An unsuccessful stage stops the
sequence. Final evaluation and joint regression/rollback still run at a budget
boundary. No next stage, retry, or extension is launched after a failed gate.

## Runtime choices

- Conda `knolling`; cached pinned R18/MiniLM, offline mode, deterministic kernels.
- BF16 is limited to Arrangement warmup. Coordinate heads stay FP32; perception
  and connected training retain FP32 for the 1 mm source/target criteria.
- The 96-record S0 suite is cached in RAM, sharing 25 source images. A pinned
  DataLoader uses zero workers to avoid process overhead for this small cached
  dataset. Its private RNG preserves model/DN RNG across resume.
- Frozen text states are cached by actual token IDs, without labels or source
  metadata. This cache is rebuilt on resume and never enters checkpoints.
- `tqdm` shows Stage, step, loss and ETA. Each stage writes `history.jsonl`,
  immutable checkpoints, and `result.json`; the sequence writes stage summaries
  and a final result. Existing datasets and checkpoints are not overwritten.
- `scripts/run_recovery_stages.py` runs under tmux and survives SSH disconnects.
  The tmux session is retained after termination for inspection.
- Registration uses the existing `/opt/knolling-notify/notify.py` watcher and
  durable terminal-event claims. No progress-message worker or direct send is
  added; the existing read-only command listener is preserved.

## Preflight evidence

- Full CPU checks and regression tests cover unchanged stage gates,
  pause/resume/rollback, DataLoader RNG isolation, token-cache semantics and
  stopping the sequence after budget exhaustion or regression.
- Real GPU forward/backward without optimizer updates passed for Arrangement
  BF16 batch 8 (peak allocated 641.5 MiB) and Perception FP32 batch 4 (2,836.7 MiB).
  Weights/buffers were unchanged. These are smoke-test memory figures, not
  training peak estimates or evidence of accuracy.

## Launch and results

- Run: `recovery_initial_20261108T021741Z` (UTC timestamp).
- Executed code: `4b1cbd5a51518b5ddd84a883e6f26c1a951da0e7`.
- Session: `knolling-recovery-initial`; reconnect with
  `tmux attach -t knolling-recovery-initial`.
- Launch config, authorization receipt and console log:
  `artifacts/recovery_launches/recovery_initial_20261108T021741Z/`.
- Results and checkpoints:
  `artifacts/recovery_runs/recovery_initial_20261108T021741Z/`.
- Preflight CPU: **201 passed / 21 skipped**; GPU runtime smoke: **1 passed**.

### Stage 1 outcome: budget exhausted, gate failed

Stage 1 stopped at **500 updates in 172.09 seconds**. The update limit was reached
before the 600-second time limit. Final evaluation covered all 96 records and 21
pure-axis pairs. Stage 2, Stage 3a and Stage 3b **were not started**. There was no
automatic retry or budget extension and no runtime/nonfinite error.

| Metric | Initial | Final | Oracle gate requirement |
|---|---:|---:|---:|
| Target position P95 | 301.00 mm | 140.84 mm | ≤1 mm |
| Target angle P95 | about 90° | 89.84° | ≤2° |
| Move accuracy | 29.52% | 76.75% | 100% |
| Pure-axis success | 0/21 | 0/21 | 21/21 |
| Full task success | 15/96 | 15/96 | Diagnostic at this stage |

All source attributes in this stage are GT. Its perfect source/count metrics
are **not a Perception training result**.

| Updates | Mean total loss | Position | Orientation | Move |
|---|---:|---:|---:|---:|
| 1–100 | 16.389 | 13.508 | 1.384 | 0.690 |
| 101–200 | 12.202 | 10.200 | 0.808 | 0.641 |
| 201–300 | 10.536 | 8.681 | 0.737 | 0.629 |
| 301–400 | 5.575 | 3.834 | 0.699 | 0.613 |
| 401–500 | 4.118 | 2.529 | 0.634 | 0.578 |

The sequence result is `stopped/update_budget`, not a successful end-to-end
system. The final checkpoint is
`oracle_layout/checkpoint-000500-0022.pt`; all intermediate checkpoints remain.

### Read-only failure diagnosis

`arrangement-diagnosis.json` records an additional forward-only examination of
the final checkpoint (zero optimizer updates):

- Of 185 objects that should move, 41 were incorrectly kept. Of 86 that should
  stay, 22 were incorrectly moved. The hard move/keep branch therefore amplifies
  classification mistakes into large final pose errors.
- Candidate positions for moved objects have median **25.58 mm**, P95
  **75.07 mm** error. After move/keep selection, moved-object P95 is **151.68 mm**.
  Both position regression and move classification still require learning.
- **55/185** moved objects select the wrong horizontal/vertical orientation bin.
  Candidate-angle median is **0.57°**, but P95 is **89.82°**. The major angle error
  is bin selection, not merely within-bin residual precision. The axis auxiliary
  head is correct on **69.89%** of 93 defined commands.
- BF16 versus FP32 forward comparison has candidate-position P95 difference
  **0.46 mm**, much smaller than the model's 75 mm regression error. One of 271
  move decisions flips near its threshold, causing a 54.92 mm maximum deployed
  difference. BF16 is not the main failure cause, but this supports retaining
  FP32 for the later connected stages as configured.

These are observations, not proof that a particular loss-weight change would
solve the problem. Position loss dominates early training; orientation/move
losses improve more slowly. An undertrained language/decision path is a plausible
explanation. No clear runtime or GT-query replacement defect was observed.

### Proposed extension — NOT approved or executed

Continued improvement in position loss, total loss and move accuracy provides a
reason to test **one additional 500-update Arrangement block**, starting from
the saved checkpoint rather than restarting. Proposed total caps: **1,000
updates and 600 cumulative seconds**, with the same full evaluation every 25
updates. No automatic second extension or admission to Stage 2 without its gate.

This requires explicit user approval and an audited continuation path for a
budget-exhausted checkpoint; the current driver intentionally refuses ordinary
resume of that status. The checkpoint's original code/config/data contract must
not be silently edited or bypassed. If approved, lack of improvement in position,
move and pure-axis metrics over 200 additional updates should trigger an early
stop and a loss/conditioning review. Passing the strict gate is not guaranteed.

### Preservation

All **2,107 pre-existing artifact files** retained their paths, sizes and mtimes;
`preservation-check.json` records the check. The new run occupies approximately
13 GiB, primarily full checkpoints; no checkpoint was deleted. Remaining storage
must be considered before extending or running later stages. The tmux session
remains available at a shell prompt; no GPU training process remains active.

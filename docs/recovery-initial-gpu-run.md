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

The launch directory records the exact receipt, command and console log. Actual
metrics and the stopping decision will be recorded here after the initial run.
Budget extension requires a separate user decision based on those results.

# Recovery checkpoint storage

New Recovery writes share immutable MiniLM/Perception state instead of copying it
into every checkpoint and rollback Anchor. Architecture, loss, optimization,
datasets, budgets, and Gate decisions are unchanged. Existing artifacts are never
rewritten. This implementation was validated without training an actual model.

## On-disk format and state ownership

The on-disk envelope is `recovery_split_storage_v1`; its reconstructed payload
remains `recovery_checkpoint_v2`. Common objects use `recovery_shared_state_v1` in
`artifacts/checkpoint_store/<tensor-content-sha256>.pt`. The reference separately
pins the **serialized file SHA256**. Content naming allows deduplication even if
torch serialization differs. No per-tensor files or delta chains are introduced.

| Stage | Shared | Local model state |
|---|---|---|
| Grounding | MiniLM + complete frozen Perception | Entire Goal + Pose |
| Micro / Oracle / predicted-layout | MiniLM + exact frozen Perception version | Entire Goal + Pose |
| Perception | MiniLM | Entire Perception + Goal + Pose |
| Joint | MiniLM | Entire Perception + Goal + Pose |

Grounding trains Goal; layout stages train Pose; Perception trains Perception;
Joint trains Pose and the existing selected detector/physical heads. Storage keeps
**all persistent buffers** as well as parameters, including the conservatively
local entire Perception in Joint. It does not infer immutability from
`requires_grad` alone. Unexpected frozen-state changes, including `.data` buffer
mutations, produce a new immutable version. Later predicted-layout saves the
trained Perception version, never substitutes pretrained defaults.

Local checkpoints retain every existing non-model field: optimizer parameter
mapping/moments/steps, PoseAdam state, solver history, RNG, cursors, elapsed and
phase budgets, pending evaluation/convergence decisions, reset count, metrics,
provenance and dependency references. Anchors retain the complete reconstructable
model. Head Journals, terminal markers and feature caches keep their existing
format and dependency checks; their base checkpoints and Anchors use the new
storage resolver. There is no optimizer stripping or automated deletion.

## Integrity, atomicity and compatibility

- Shared objects are published first using the existing temporary-file,
  `fsync`, exclusive atomic-publication writer. Only then is the manifest
  published. Existing paths cannot be overwritten. An interrupted write may
  leave an unreferenced shared object but cannot expose a partial manifest.
- Every load checks shared file SHA256, canonical tensor-content hash, exact
  partition keys, shapes/dtypes, overlap, local tensor hash and full key order.
  State-dict module metadata is retained. Missing, modified or incompatible
  dependencies fail closed; no default initialization or network recovery occurs.
- Repeated saves in one process reuse verified immutable file hashes with
  inode/size/mtime/ctime checks and exact bitwise tensor comparisons. This includes
  signed zero and NaN bits. Resume always verifies shared hashes again.
- `load_artifact()` reconstructs the full state before existing code/config/data,
  purpose, parent-approval and strict model-load checks. Driver, integrated
  Journal/base/Anchor, evidence transfer and extension readers use this resolver.
- Legacy full checkpoints and legacy plain Anchors remain readable. Their hashes
  and original approval bindings are unchanged. A converted copy has a new file
  hash and does **not** inherit an approval for the original file automatically.
  Existing commit/config checks are intentionally not bypassed.

For a diagnostic/export tool, use:

```python
from knolling_model.recovery.storage import load_artifact
from knolling_model.recovery.driver import save_exclusive

complete_state = load_artifact(checkpoint_path)
# Optional, explicitly requested standalone export into a NEW path:
save_exclusive(new_export_path, complete_state)
```

Plain `torch.load()` of a new checkpoint returns the storage envelope. External
scripts must use the resolver. A standalone export reproduces tensor/execution
state exactly, not the original torch ZIP bytes or original file SHA256.

## Measured validation — 2026-11-08

Environment: `knolling` (`/venv/knolling`), NVIDIA RTX 4090. Measurements and source
hashes: `artifacts/storage_validation/20261108T062828Z/result.json`.

| Actual artifact | Legacy bytes / MiB | New local bytes / MiB | Reduction excluding shared copy |
|---|---:|---:|---:|
| Grounding PASS update 450 | 561,415,535 / 535.41 | 8,108,129 / 7.73 | 98.56% |
| Integrated H/V update 173 | 560,629,215 / 534.66 | 7,357,393 / 7.02 | 98.69% |
| H/V rollback Anchor | — | 3,905,950 / 3.73 | — |

Common files total **553,440,652 bytes / 527.80 MiB**, shared by both actual
checkpoints and the Anchor. Initial save including common publication took 4.51 s;
the subsequent save reusing those objects took 0.291 s in this CPU-source
measurement. These are single storage measurements, not GPU training benchmarks.
Perception/Joint savings are smaller because mutable Perception and its optimizer
remain local; no actual Perception/Joint training checkpoint size is claimed.

- Full model, persistent buffers, optimizer and all execution payload fields of
  both legacy sources and new copies matched bitwise. Full standalone export and
  reload also matched; only that newly created scratch export was removed.
- Actual legacy vs split CUDA forward/backward predictions and all **36 Pose
  gradient tensors** matched exactly. No actual-model optimizer was created or
  stepped; original model state also remained unchanged after backward.
- The full related suite passed **154 tests**. Two additional tests were then
  added (signed-zero/NaN cache identity and split evidence-transfer coverage);
  the affected storage/provenance suites passed **28 tests**, including the
  previously counted tests. Total: **156 unique passing test cases**.
- CPU/CUDA synthetic fixtures cover six-stage state partitioning, uninterrupted
  vs resumed update/model/optimizer/RNG equality, pending Journal evaluation
  before another update, PASS/regression/convergence/cap decisions, Rollback,
  solver abort, Fine reset and cache invalidation. These fixture optimizer steps
  are tests, not experiments on the actual Knolling model.
- Failure injection covers before/after shared and manifest publication, partial
  serialization/full-disk errors, missing/corrupt blobs, modified local tensors,
  duplicate/missing partitions and incompatible shapes. Legacy import, split
  bundle relocation and standalone export are also checked.
- Preservation audit checked size, nanosecond mtime, inode and link count for
  **2,405 pre-existing artifact/data files** with no changes or missing files.
  Both actual source checkpoint SHA256 hashes were separately recomputed and
  unchanged. See `preservation.json`; this is not a claim to rehash all old data.

The reproducible real-state check is `scripts/validate_recovery_storage.py`
(`--grounding`, `--pose`, `--output NEW_DIRECTORY`, optional `--cuda`). It requires
the `knolling` environment, writes only new files, and performs zero optimizer
updates. Offline model tests use the existing Hugging Face cache.

## Remaining operational constraints

Shared objects are now dependencies of many artifacts: preserve them together
with retained checkpoints, Anchors and Journal bases. Relative common-store
references permit moving a split bundle with its structure intact; existing
integrated Journal/cache/Anchor absolute paths are unchanged, so arbitrary whole
run relocation still needs its original dependency handling. No garbage
collection, retention deletion or legacy migration is enabled.

The writer keeps a CPU snapshot of shared tensors for exact change detection
(roughly 528 MiB for these weights), with additional transient load/hash memory.
It avoids repeated disk serialization, not all GPU-to-CPU transfer/comparison.
Atomic staging requires room for a new shared blob on first publication and for
each mutable artifact. Power-loss/filesystem hardware failures were not tested;
tests inject process/serialization failures using the existing durable writer.
The earlier proposal's automatic disk reservation and retention planner remain
separate future work, not silently enabled by this storage change.

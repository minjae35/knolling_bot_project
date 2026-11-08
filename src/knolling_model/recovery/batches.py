"""Deterministic stage batch wiring, with a separate unique-source replay stream."""
from dataclasses import dataclass
import random

from .data import balanced_instruction_indices, unique_source_indices
from .stages import STAGES


@dataclass
class StageBatch:
    batch: object
    replay: object | None

    def loss_arguments(self):
        replay = None if self.replay is None else (self.replay.inputs, self.replay.supervision)
        return self.batch.inputs, self.batch.supervision, replay


def stage_batches(dataset, collator, stage, *, seed, epoch, batch_size=2):
    """Yield CPU batches only; no device transfer, optimizer, or file writes.

    Joint epochs cover both the complete instruction stream and the complete
    unique-source stream; the shorter stream cycles. This prevents partial-case
    repair subsets from silently replacing full detection replay.
    """
    if stage not in STAGES or batch_size < 1:
        raise ValueError('invalid stage or batch size')
    if not dataset.records or any(r['split'] != 'train' for r in dataset.records):
        raise ValueError('training batches require nonempty Train-only records')
    sources = unique_source_indices(dataset.records)
    random.Random(seed+epoch).shuffle(sources)
    indices = sources if stage == 'perception' else balanced_instruction_indices(dataset.records, seed, epoch)
    chunks = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    replay_chunks = [sources[i:i+batch_size] for i in range(0, len(sources), batch_size)]
    steps = max(len(chunks), len(replay_chunks)) if stage == 'joint' else len(chunks)
    for step in range(steps):
        batch = collator([dataset[i] for i in chunks[step % len(chunks)]])
        replay = collator([dataset[i] for i in replay_chunks[step % len(replay_chunks)]]) if stage == 'joint' else None
        yield StageBatch(batch, replay)

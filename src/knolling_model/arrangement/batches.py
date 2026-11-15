"""Deterministic stage batch wiring, with a separate unique-source replay stream."""
from dataclasses import dataclass
import random
import torch
from torch.utils.data import DataLoader

from .data import balanced_instruction_indices, unique_source_indices
from .stages import ALL_STAGES


@dataclass
class StageBatch:
    batch: object
    replay: object | None

    def to(self, device, non_blocking=False):
        return StageBatch(self.batch.to(device, non_blocking=non_blocking),
                          None if self.replay is None else self.replay.to(device, non_blocking=non_blocking))

    def loss_arguments(self):
        replay = None if self.replay is None else (self.replay.inputs, self.replay.supervision)
        return self.batch.inputs, self.batch.supervision, replay


def stage_batches(dataset, collator, stage, *, seed, epoch, batch_size=2, num_workers=0, pin_memory=False, uniform=False):
    """Yield CPU batches only; no device transfer, optimizer, or file writes.

    Joint epochs cover both the complete instruction stream and the complete
    unique-source stream; the shorter stream cycles. This prevents partial-case
    repair subsets from silently replacing full detection replay.
    """
    if stage not in ALL_STAGES or batch_size < 1:
        raise ValueError('invalid stage or batch size')
    if not dataset.records or any(r['split'] != 'train' for r in dataset.records):
        raise ValueError('training batches require nonempty Train-only records')
    sources = unique_source_indices(dataset.records)
    random.Random(seed+epoch).shuffle(sources)
    indices = sources if stage == 'perception' else balanced_instruction_indices(dataset.records, seed, epoch)
    if uniform and stage != 'perception':
        indices = list(range(len(dataset.records)))
        random.Random(seed+100003*epoch).shuffle(indices)
    chunks = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    replay_chunks = [sources[i:i+batch_size] for i in range(0, len(sources), batch_size)]
    steps = max(len(chunks), len(replay_chunks)) if stage == 'joint' else len(chunks)
    def loader(groups, offset):
        # A private generator avoids changing model/DN RNG on iterator creation
        # or resume, including when num_workers is zero.
        generator = torch.Generator().manual_seed(seed+epoch+offset)
        return DataLoader(dataset, batch_sampler=[groups[i % len(groups)] for i in range(steps)],
                          collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
                          generator=generator)
    replay = iter(loader(replay_chunks, 100003)) if stage == 'joint' else None
    for batch in loader(chunks, 0):
        yield StageBatch(batch, next(replay) if replay is not None else None)

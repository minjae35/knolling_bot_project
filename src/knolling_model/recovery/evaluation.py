"""Strict inference adapter; overflow is a failure, never GT-count truncation."""
import torch

from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch, summarize
from .model import CapacityError, legacy_evaluation_view


@torch.no_grad()
def evaluate_recovery(model, batches):
    """Read-only free rollout using unchanged S0 geometry/semantics and coverage.

    Per-image processing makes capacity failures explicit without discarding
    other images in a batch. This intentionally favors audit clarity over speed.
    """
    if model.threshold != .5:
        raise ValueError('legacy evaluation fixes the foreground threshold at 0.5')
    was_training = model.training
    model.eval()
    rows, records = [], []
    try:
        for batch in batches:
            for b, meta in enumerate(batch.metadata):
                one = Batch({k: v[b:b+1] for k, v in batch.inputs.items()}, [batch.supervision[b]], [meta])
                visual = model.perception(one.inputs['images'])
                try:
                    output = model.layout_from_perception(visual, one.inputs['input_ids'], one.inputs['attention_mask'])
                    view = legacy_evaluation_view(output)
                    overflow = False
                except CapacityError:
                    # S0 truth counts are 2-4 (supported domain <= max_objects).
                    # Thus this path must fail count before any target pose metric.
                    if len(one.supervision[0]['source_xy']) > model.max_objects:
                        raise ValueError('evaluation truth exceeds the reviewed domain') from None
                    source = visual['source']
                    view = dict(source=source, targets=[dict(uv=source['uv'], angle=source['angle'])])
                    overflow = True
                result = evaluate_batch(view, one)[0]
                if overflow:
                    if result['count_correct']:
                        raise AssertionError('capacity failure cannot have correct in-domain count')
                    result['errors'].append('layout_capacity')
                rows.append(result)
                records.append(meta['record'])
        if not rows:
            raise ValueError('empty evaluation set')
        return summarize(rows, records), rows
    finally:
        model.train(was_training)

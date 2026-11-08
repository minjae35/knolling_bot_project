import torch
from torch import nn

from knolling_model.data import Batch
from knolling_model.recovery.evaluation import evaluate_recovery
from knolling_model.recovery.model import CapacityError


class OverflowFixture(nn.Module):
    threshold = .5
    max_objects = 4

    def perception(self, images):
        n = 7
        source = dict(uv=torch.ones(1, n, 2)*.5, angle=torch.zeros(1, n), size=torch.ones(1, n, 2)*.05,
                      objectness_logits=torch.full((1, n), 10.), color_logits=torch.zeros(1, n, 3),
                      shape_logits=torch.zeros(1, n, 3))
        return dict(source=source)

    def layout_from_perception(self, visual, input_ids, attention_mask):
        raise CapacityError([7], self.max_objects)


def test_overflow_counts_as_failure_instead_of_being_dropped(s0_dataset):
    sample = s0_dataset[0]
    batch = Batch(dict(images=sample['image'][None], input_ids=torch.tensor([[1]]),
                       attention_mask=torch.tensor([[True]])), [sample['supervision']], [sample['metadata']])
    model = OverflowFixture().train()
    summary, rows = evaluate_recovery(model, [batch])
    assert model.training
    assert summary['samples'] == 1 and summary['count_accuracy'] == 0 and not summary['s0_gate']
    assert summary['correspondence_coverage'] == 0
    assert summary['target_position_mm']['n'] == 0
    assert 'layout_capacity' in rows[0]['errors']

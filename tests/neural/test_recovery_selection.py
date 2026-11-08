from types import SimpleNamespace

import pytest
import torch
from transformers import RTDetrConfig

from knolling_model.recovery.contracts import detector_labels
from knolling_model.recovery.selection import select_queries, match_selected, SourceSelectionError
from test_recovery_perception import truth


def test_matching_cannot_substitute_a_better_below_threshold_query():
    t = truth()
    boxes = detector_labels([t])[0]['boxes'][[0, 1, 1]].clone()
    boxes[1, 0] += .0003/.6
    scores = torch.logit(torch.tensor([[.9, .50001, .49999]]))
    source = {'objectness_logits': scores, 'uv': torch.stack((boxes[:, 0], 1-boxes[:, 1]), -1)[None]}
    visual = dict(source=source, stages=[dict(boxes=boxes[None], logits=scores[..., None])])
    model = SimpleNamespace(threshold=.5, max_objects=8, perception=SimpleNamespace(
        detector=SimpleNamespace(config=RTDetrConfig(num_labels=1))))
    pred, gt = match_selected(model, visual, [t])[0]
    assert pred.tolist() == select_queries(source)[0].tolist() == [0, 1]
    assert gt.tolist() == [0, 1]
    source['objectness_logits'] = torch.full_like(scores, -10)
    with pytest.raises(SourceSelectionError, match='count'):
        match_selected(model, visual, [t])
    source['objectness_logits'] = scores
    source['uv'] = source['uv'] + .2
    with pytest.raises(SourceSelectionError, match='correspondence'):
        match_selected(model, visual, [t])

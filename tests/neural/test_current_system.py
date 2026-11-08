from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn

from knolling_model.current import CurrentSystem, load_current
from knolling_model.perception.adapter import oracle_frame
from knolling_model.recovery.neural_keep import attach
from knolling_model.recovery.relative_task import configure
from test_recovery_relative_task import setup
from test_recovery_readout import batch_for
from test_recovery_integrated import equal


class PerceptionFixture(nn.Module):
    def __init__(self, frames):
        super().__init__();self.frames=frames
    def predict(self, images, ids, threshold):return self.frames


def system():
    model,_,_,_=setup();attach(model);configure(model,orientation=False)
    model.threshold=.5;batch=batch_for(model)
    frames=[replace(oracle_frame(t,[str(j) for j in range(len(t['source_xy']))],str(i)),source_mode='predicted_rgb')
            for i,t in enumerate(batch.supervision)]
    tokenizer=lambda commands,**kwargs: batch.inputs
    return CurrentSystem(PerceptionFixture(frames),model,tokenizer)


def test_predicted_objects_reach_keep_layout_without_targets_or_count_filtering():
    s=system();before=deepcopy(s.arrangement.state_dict())
    out=s.predict(torch.zeros(2,3,384,576),['horizontal','vertical'],['0','1'])
    assert out['objects'].valid.sum(-1).tolist()==[2,3]
    assert out['layout']['xy'].shape==(2,3,2)
    assert torch.isfinite(out['layout']['xy']).all()
    assert all(r['source_mode']=='predicted_rgb' for r in out['provenance'])
    assert not out['layout']['xy'].requires_grad
    equal(s.arrangement.state_dict(),before)
    assert all(p.grad is None for p in s.arrangement.parameters())


def test_overflow_and_oracle_provenance_fail_closed():
    s=system();s.arrangement.max_objects=2
    with pytest.raises(ValueError,match='capacity'):s.predict(torch.zeros(2,3,384,576),['a','b'],['0','1'])
    s.arrangement.max_objects=8
    s.perception.frames[0]=replace(s.perception.frames[0],source_mode='oracle_gt')
    with pytest.raises(ValueError,match='provenance'):s.predict(torch.zeros(2,3,384,576),['a','b'],['0','1'])


def test_current_loader_rejects_tampered_checkpoint_before_loading(tmp_path):
    artifact=tmp_path/'fake.pt';artifact.write_bytes(b'not a checkpoint')
    manifest=tmp_path/'models.json'
    manifest.write_text(json.dumps(dict(schema='knolling_current_models_v1',arrangement=dict(checkpoint=str(artifact),sha256='0'*64))))
    with pytest.raises(ValueError,match='hash mismatch'):load_current(manifest)

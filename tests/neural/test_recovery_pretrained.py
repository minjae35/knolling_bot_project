"""Opt-in, cached real-model CPU shape/backward check; no optimizer or evaluation run."""
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_PRETRAINED_TESTS') != '1',
                    reason='opt-in cached real-model CPU smoke test')
def test_cached_r18_minilm_cpu_shapes_and_backward(s0_dataset):
    from knolling_model.data import Collator
    from knolling_model.recovery.data import RecoveryDataset
    from knolling_model.recovery.model import load_recovery
    from knolling_model.recovery.stages import configure_stage, stage_loss
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.manual_seed(1203)
    model, tokenizer = load_recovery()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    dataset = RecoveryDataset(s0_dataset)
    batch = Collator(tokenizer)([dataset[0]])
    configure_stage(model, 'perception')
    terms, output = stage_loss(model, batch.inputs, batch.supervision)
    assert output['perception']['source']['uv'].shape == (1, 300, 2)
    assert len(output['perception']['stages']) == 4
    assert len(output['perception']['dn_stages']) == 3
    assert all(torch.isfinite(v) for v in terms.values())
    sum(terms.values()).backward()
    gradients = [p.grad for p in model.perception.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    model.zero_grad(set_to_none=True)
    configure_stage(model, 'predicted_layout')
    terms, output = stage_loss(model, batch.inputs, batch.supervision)
    assert output['layout']['xy'].shape == (1, len(batch.supervision[0]['size']), 2)
    assert output['layout']['rollout'] == 'free_rollout'
    sum(terms.values()).backward()
    assert model.layout.text_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.perception.parameters())
    assert all(p.grad is None for p in model.text_encoder.parameters())

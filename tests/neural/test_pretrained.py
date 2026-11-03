import os
import pytest
import torch


@pytest.mark.skipif(os.environ.get('RUN_PRETRAINED_TESTS') != '1', reason='requires cached pinned pretrained assets')
def test_real_pretrained_offline_forward_backward_without_optimizer(s0_dataset, monkeypatch):
    import requests
    from knolling_model.model import load_pretrained
    from knolling_model.data import Collator
    from knolling_model.losses import S0Criterion
    def forbidden(*args, **kwargs):
        raise AssertionError('offline pretrained loading must not contact the network')
    monkeypatch.setattr(requests.Session, 'request', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    model, tokenizer = load_pretrained(local_files_only=True)
    collate = Collator(tokenizer)
    whole = collate([s0_dataset[i] for i in range(32)])
    assert whole.inputs['input_ids'].shape[1] <= 512
    batch = collate([s0_dataset[0], s0_dataset[1]])
    before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    losses, _ = S0Criterion()(model(**batch.inputs), batch.supervision)
    losses['loss'].backward()
    assert torch.isfinite(losses['loss'])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(torch.equal(p, before[n]) for n,p in model.named_parameters() if n in before)

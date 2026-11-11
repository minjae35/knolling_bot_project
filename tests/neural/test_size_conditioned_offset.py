from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import torch
from test_phase1_single_row import Reference, example
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.storage import CheckpointStore, load_artifact, tensor_digest
from knolling_model.experiments.neural_sequential import ScenePoseDecoder, SequentialPose, position_loss
from knolling_model.experiments.size_conditioned_offset import (
    EXTRA, FEATURE_SCALE, SizeConditionedDecoder, widen_decoder)


def _pair():
    torch.manual_seed(1203)
    original = SequentialPose(Reference()).eval()
    widened = deepcopy(original).eval()
    widen_decoder(widened.decoder)
    return original, widened


def test_zero_columns_preserve_the_original_decoder():
    original, widened = _pair()
    obj, goal, selected = example('cpu')
    assert widened.decoder.output[0].in_features == original.decoder.output[0].in_features + EXTRA
    assert torch.count_nonzero(widened.decoder.output[0].weight[:, -EXTRA:]) == 0
    assert torch.equal(widened.decoder.output[0].weight[:, :-EXTRA], original.decoder.output[0].weight)
    assert torch.equal(widened(obj, goal, selected)['xy'], original(obj, goal, selected)['xy'])
    order = torch.tensor([[2, 0, 1]])
    assert torch.equal(widened.diagnose(obj, goal, selected, order)['xy'],
                       original.diagnose(obj, goal, selected, order)['xy'])
    assert torch.equal(widened.diagnose(obj, goal, selected, order, obj.xy)['xy'],
                       original.diagnose(obj, goal, selected, order, obj.xy)['xy'])
    text = Path('src/knolling_model/experiments/neural_sequential.py').read_text()
    assert 'previous_size' not in text
    assert 'sizes' not in text.split('class ScenePoseDecoder', 1)[1].split('class SequentialPose', 1)[0]


def test_permutation_padding_and_direct_size_slots():
    _, widened = _pair()
    obj, goal, selected = example('cpu', 2)
    with torch.no_grad():
        widened.decoder.output[0].weight[:, -EXTRA:].normal_(0, .05)
    out = widened(obj, goal, selected)
    perm_index = torch.tensor([2, 0, 1, 3, 4])
    perm = Objects(**{f.name: getattr(obj, f.name)[:, perm_index] for f in fields(obj)})
    changed = widened(perm, goal, selected[:, perm_index])
    torch.testing.assert_close(changed['xy'], out['xy'][:, perm_index], atol=1e-6, rtol=0)
    plain, _, _ = example('cpu')
    order = torch.tensor([[0, 2, 1]])
    seen = []

    def capture(_module, inputs):
        seen.append(inputs[0].detach().clone())

    handle = widened.decoder.output.register_forward_pre_hook(capture)
    widened.decoder(plain, goal, selected[:, :3], out['move_mask'][:, :3], order)
    handle.remove()
    assert len(seen) == 3
    scale = seen[0].new_tensor(FEATURE_SCALE)
    assert torch.equal(seen[0][0, -EXTRA:-2], torch.zeros(2))
    assert torch.allclose(seen[0][0, -2:], plain.size[0, 0] / scale)
    assert torch.allclose(seen[1][0, -EXTRA:-2], plain.size[0, 0] / scale)
    assert torch.allclose(seen[1][0, -2:], plain.size[0, 2] / scale)


def test_new_columns_have_a_nonzero_size_gradient_without_unit_gain():
    torch.manual_seed(5)
    _, widened = _pair()
    obj, goal, selected = example('cpu')
    obj = Objects(**{f.name: getattr(obj, f.name).detach().clone() for f in fields(obj)})
    obj.size = obj.size.requires_grad_(True)
    out = widened(obj, goal, selected)
    position_loss(out, out['xy'].detach() + torch.tensor([.004, -.002])).backward()
    new_grad = widened.decoder.output[0].weight.grad[:, -EXTRA:]
    assert torch.isfinite(new_grad).all() and new_grad.abs().sum() > 0
    assert torch.isfinite(obj.size.grad).all() and obj.size.grad.abs().sum() > 0
    widened.zero_grad(set_to_none=True)
    with torch.no_grad():
        # Wiring probe only. Training leaves these columns at zero.
        widened.decoder.output[0].weight[:, -EXTRA:].normal_(0, 1)
    order = torch.tensor([[0, 1, 2]])
    move = out['move_mask']
    step = 1e-3
    def at(size):
        probe = Objects(**{f.name: getattr(obj, f.name).detach() for f in fields(obj)} | {'size': size})
        return widened.decoder(probe, goal, selected, move, order)[0, 1].sum()
    base_size = obj.size.detach()
    def bump(sign):
        size = base_size.clone()
        size[0, 1] += sign * step
        return at(size)
    gain = float((bump(1) - bump(-1)).detach() / (2 * step))
    assert gain != 0 and torch.isfinite(torch.tensor(gain))
    value = at(obj.size)
    numeric = torch.autograd.grad(value, obj.size)[0][0, 1].sum()
    torch.testing.assert_close(numeric, torch.tensor(gain), rtol=1e-3, atol=1e-4)


def test_checkpoint_round_trip(tmp_path):
    _, widened = _pair()
    obj, goal, selected = example('cpu')
    order = torch.tensor([[2, 0, 1]])
    opt = torch.optim.AdamW([p for p in widened.parameters() if p.requires_grad], lr=1e-4)
    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        out = model(obj, goal, selected)
        loss = position_loss(out, obj.xy) + model.selector_loss(obj, goal, selected, out['move_mask'], order)
        loss.backward()
        optimizer.step()
    step(widened, opt)
    path = tmp_path / 'checkpoint.pt'
    CheckpointStore(tmp_path / 'shared').save_checkpoint(
        path, dict(stage='oracle_layout', model=widened.state_dict(), optimizer=opt.state_dict()))
    restored = SequentialPose(Reference()).eval()
    widen_decoder(restored.decoder)
    raw = load_artifact(path)
    restored.load_state_dict(raw['model'], strict=True)
    other = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-4)
    other.load_state_dict(raw['optimizer'])
    assert isinstance(restored.decoder, SizeConditionedDecoder)
    assert torch.equal(widened(obj, goal, selected)['xy'], restored(obj, goal, selected)['xy'])
    step(widened, opt)
    step(restored, other)
    assert tensor_digest(widened.state_dict()) == tensor_digest(restored.state_dict())
    for left, right in zip(opt.state.values(), other.state.values()):
        for key in left:
            assert torch.equal(left[key], right[key])


def test_head_input_gradient_is_open_and_not_forced_to_one():
    torch.manual_seed(9)
    _, widened = _pair()
    obj, goal, selected = example('cpu')
    out = widened(obj, goal, selected)
    order = torch.tensor([[0, 1, 2]])
    captured = {}

    def pre(_module, inputs):
        captured['values'] = inputs[0]

    handle = widened.decoder.output.register_forward_pre_hook(pre)
    widened.decoder(obj, goal, selected, out['move_mask'], order)
    handle.remove()
    def slot_grad():
        slot = captured['values'][:, -EXTRA:].detach().requires_grad_(True)
        body = captured['values'].detach().clone()
        body[:, -EXTRA:] = slot
        delta = widened.decoder.output(body)[0].sum()
        return torch.autograd.grad(delta, slot)[0]

    assert torch.count_nonzero(slot_grad()) == 0
    with torch.no_grad():
        widened.decoder.output[0].weight[:, -EXTRA:].normal_(0, .05)
    opened = slot_grad()
    assert torch.isfinite(opened).all() and opened.abs().sum() > 0

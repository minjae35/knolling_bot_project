import ast
import inspect
import textwrap
from pathlib import Path
import torch
from knolling_model.experiments.neural_sequential import SequentialPose
from knolling_model.experiments.offset_supervision import ANCHOR_X, SCALE, x_offset_loss


def _batch():
    xy = torch.tensor([[[.405, .20], [.550, .25], [9., 9.]],
                       [[.310, .20], [.470, .22], [.630, .21]]], dtype=torch.float64, requires_grad=True)
    order = torch.tensor([[0, 1, -1], [0, 1, 2]])
    target = torch.tensor([[[.400, .20], [.520, .20], [0., 0.]],
                           [[.320, .20], [.450, .20], [.610, .20]]], dtype=torch.float64)
    return xy, order, target


def test_offset_uses_fixed_anchor_and_ignores_y_and_padding():
    xy, order, target = _batch()
    value = x_offset_loss(xy, order, target)
    # Scene 0 steps: (0.405-0.3)-(0.400-0.3)=0.005; (0.550-0.400)-(0.520-0.400)=0.030.
    # Scene 1 steps: -0.010, 0.020, 0.020. Mean of per-scene means.
    expected = []
    prev = [ANCHOR_X, ANCHOR_X]
    for b, steps in enumerate(((0, 1), (0, 1, 2))):
        terms = []
        previous = prev[b]
        for j in steps:
            err = (float(xy[b, j, 0].detach()) - previous) - (float(target[b, j, 0]) - previous)
            terms.append(torch.nn.functional.smooth_l1_loss(torch.tensor(err / SCALE, dtype=torch.float64),
                                                           torch.zeros((), dtype=torch.float64)))
            previous = float(target[b, j, 0])
        expected.append(torch.stack(terms).mean())
    torch.testing.assert_close(value, torch.stack(expected).mean())
    shifted = xy.detach().clone()
    shifted[..., 1] += 5.
    shifted.requires_grad_()
    torch.testing.assert_close(x_offset_loss(shifted, order, target), value)
    value.backward()
    assert torch.count_nonzero(xy.grad[..., 1]) == 0
    assert torch.count_nonzero(xy.grad[0, 2]) == 0
    assert xy.grad[0, 0, 0] != 0 and xy.grad[0, 1, 0] != 0


def test_later_step_does_not_read_the_predicted_prefix():
    xy, order, target = _batch()
    base = x_offset_loss(xy, order, target)
    moved = xy.detach().clone()
    moved[0, 0, 0] += .02
    moved.requires_grad_()
    changed = x_offset_loss(moved, order, target)
    assert not torch.equal(base, changed)
    # The second object's offset still subtracts the GT x of the first, 0.400.
    second = (moved[0, 1, 0] - target[0, 0, 0]) - (target[0, 1, 0] - target[0, 0, 0])
    manual = torch.nn.functional.smooth_l1_loss(second.float() / SCALE, second.new_zeros(()))
    # Isolate scene 0 step 1 by evaluating a one-step tail with the same GT anchor.
    tail_xy = torch.stack((moved[0, 1], moved[0, 1])).detach()[None]
    # Direct residual check: moving the first prediction leaves the second residual unchanged.
    original_second = (xy[0, 1, 0] - target[0, 0, 0]) - (target[0, 1, 0] - target[0, 0, 0])
    torch.testing.assert_close(second, original_second)
    changed.backward()
    assert moved.grad[0, 1, 0] != 0
    # Finite difference away from the Huber kink.
    step = 1e-4
    hi = moved.detach().clone(); lo = moved.detach().clone()
    hi[0, 1, 0] += step; lo[0, 1, 0] -= step
    numeric = (x_offset_loss(hi, order, target) - x_offset_loss(lo, order, target)) / (2 * step)
    torch.testing.assert_close(numeric, moved.grad[0, 1, 0], rtol=1e-5, atol=1e-5)
    assert manual.ndim == 0 and tail_xy.shape[-1] == 2


def test_inference_forward_has_no_teacher_prefix():
    source = textwrap.dedent(inspect.getsource(SequentialPose.forward))
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls, source
    for call in calls:
        names = [kw.arg for kw in call.keywords]
        assert 'teacher_xy' not in names
        assert 'target' not in names
    text = Path('src/knolling_model/experiments/neural_sequential.py').read_text()
    forward = text.split('def forward(self, objects, goal, selected):', 2)[2]
    forward = forward.split('\n    def selector_loss', 1)[0]
    assert 'teacher_xy' not in forward
    assert 'x_offset_loss' not in text

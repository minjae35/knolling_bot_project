"""Save and reload must reproduce the same row-placement predictions."""
import torch

from knolling_model.experiments.relation_pose import NeuralRelationPose
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement, step_placement_loss


def _batch():
    torch.manual_seed(4)
    head = NeuralRelationPose(12)
    head.requires_grad_(False)
    features = torch.randn(2, 4, 12)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    goal = torch.zeros(2, 13)
    goal[:, 6] = 1
    row = torch.tensor([[0, 0, 1, -1], [0, 1, 1, 0]])
    slot = torch.tensor([[0, 1, 0, -1], [0, 0, 1, 1]])
    widths = torch.tensor([[.04, .06, .05, 0.], [.03, .04, .05, .06]])
    heights = torch.tensor([[.02, .03, .04, 0.], [.02, .05, .03, .04]])
    target = torch.zeros(2, 4, 2)
    target[..., 0] = torch.tensor([[.10, .18, .12, 0.], [.11, .20, .28, .19]])
    target[..., 1] = torch.tensor([[.20, .20, .08, 0.], [.22, .10, .10, .22]])
    angle = torch.zeros(2, 4)
    return head.hidden(features).detach(), valid, goal, widths, heights, row, slot, target, angle


def test_reloaded_checkpoint_matches_predictions_optimizer_and_step(tmp_path):
    hidden, valid, goal, widths, heights, row, slot, target, angle = _batch()
    torch.manual_seed(9)
    model = NeuralRowPlacement()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    step_placement_loss(output, target, angle, valid, row, slot)['loss'].backward()
    optimizer.step()
    config = dict(supervision='step_dx', lr=1e-3, update_name='test')
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 17, config)
    with torch.no_grad():
        original = model(hidden, valid, goal, widths, heights, row, slot, False)
    restored = NeuralRowPlacement()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0.)
    state = load_checkpoint(path, restored, restored_optimizer)
    with torch.no_grad():
        again = restored(hidden, valid, goal, widths, heights, row, slot, False)
    torch.testing.assert_close(again['xy'], original['xy'])
    torch.testing.assert_close(again['delta_x'], original['delta_x'])
    torch.testing.assert_close(again['angle'], original['angle'])
    torch.testing.assert_close(again['row_logits'], original['row_logits'])
    assert state['update'] == 17
    assert state['config'] == config
    assert state['optimizer']['param_groups'][0]['lr'] == 1e-3
    left = optimizer.state_dict()['state']
    right = restored_optimizer.state_dict()['state']
    assert left.keys() == right.keys()
    for key, value in left.items():
        torch.testing.assert_close(right[key]['exp_avg'], value['exp_avg'])
        torch.testing.assert_close(right[key]['exp_avg_sq'], value['exp_avg_sq'])

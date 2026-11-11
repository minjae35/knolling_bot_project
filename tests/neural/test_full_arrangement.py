"""Variable-length batches stay on the existing relation head."""
import torch
from knolling_model.experiments.full_arrangement import collate
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss


def scene(count, y):
    generator = torch.Generator().manual_seed(count)
    return dict(features=torch.randn(count, 12, generator=generator),
                xy=torch.rand(count, 2, generator=generator), size=torch.full((count, 2), .02),
                angle=torch.zeros(count), color=torch.tensor([[1., 0., 0.]]).expand(count, 3).contiguous(),
                shape=torch.tensor([[1., 0., 0.]]).expand(count, 3).contiguous(),
                valid=torch.ones(count, dtype=torch.bool),
                query_index=torch.arange(count), goal=torch.zeros(13),
                target_xy=torch.stack((torch.linspace(.1, .1+.04*(count-1), count), torch.full((count,), y)), 1),
                target_angle=torch.zeros(count), order=torch.arange(count))


def test_padding_does_not_change_the_relation_loss():
    items = [scene(2, .30), scene(4, .30), scene(6, .22), scene(8, .14)]
    batch = collate(items, 'cpu')
    assert batch['valid'].tolist() == [[1, 1, 0, 0, 0, 0, 0, 0], [1]*4+[0]*4, [1]*6+[0]*2, [1]*8]
    head = NeuralRelationPose(12)

    def loss_of(perturb):
        features = batch['features'].clone()
        if perturb:
            features[~batch['valid']] = 3
        output = head(features, batch['valid'], batch['goal'], batch['order'])
        return relation_loss(output, batch['target_xy'], batch['target_angle'], batch['order'])['loss']

    torch.manual_seed(5)
    first = loss_of(False)
    torch.manual_seed(5)
    second = loss_of(True)
    torch.testing.assert_close(first, second)
    output = head(batch['features'], batch['valid'], batch['goal'], batch['order'])
    relation_loss(output, batch['target_xy'], batch['target_angle'], batch['order'])['loss'].backward()
    assert head.relation[-1].weight.grad.abs().sum() > 0


def test_a_row_break_is_one_neighbor_delta():
    upper = torch.tensor([[.10, .30], [.22, .30]])
    lower = torch.tensor([[.10, .16]])
    points = torch.cat((upper, lower))
    delta = points[1:]-points[:-1]
    assert abs(float(delta[0, 1])) < 1e-6
    assert float(delta[1, 1]) < -.10
    rebuilt = torch.cat((points[:1], points[:1]+torch.cumsum(delta, 0)))
    torch.testing.assert_close(rebuilt, points)

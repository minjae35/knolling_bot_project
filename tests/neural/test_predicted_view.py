"""Predicted-view rendering keeps source assets and refuses an invented match."""
import inspect

import torch

from knolling_dataset.schema import Asset, Pose, Scene, scene_dict
from knolling_model.experiments.predicted_view import associate_source_assets, scene_at_predicted_poses


def _asset(name):
    return Asset(name, 'fam', 'box', .060, .040, .010,
                 ((-.030, -.020), (.030, -.020), (.030, .020), (-.030, .020)))


def _scene():
    objects = (
        _object('a', .15, .20),
        _object('b', .40, .20),
    )
    return Scene('fam', 7, 'validation', objects)


def _object(name, x, y):
    from knolling_dataset.schema import ObjectState
    return ObjectState(name, _asset(name), 'red' if name == 'a' else 'green', Pose(x, y, 0.))


def _rows(values, width):
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1).expand(1, width, -1).contiguous()


def _visual(centers):
    width = len(centers)
    uv = torch.tensor([[x/.6, y/.4] for x, y in centers], dtype=torch.float32)[None]
    return dict(source=dict(
        uv=uv, size=_rows([.060, .040], width), angle=torch.zeros(1, width),
        objectness_logits=torch.full((1, width), 8.)))


def _packed(centers, poses):
    from knolling_model.arrangement.contracts import Objects
    count = len(centers)
    return dict(
        xy=torch.tensor(poses, dtype=torch.float32)[None],
        angle=torch.zeros(1, count),
    ), Objects(
        xy=torch.tensor(centers, dtype=torch.float32)[None],
        size=_rows([.060, .040], count),
        angle=torch.zeros(1, count),
        color=_rows([1., 0., 0.], count),
        shape=_rows([1., 0., 0.], count),
        valid=torch.ones(1, count, dtype=torch.bool),
        query_index=torch.arange(count)[None],
    ).validate()


def test_association_uses_source_assets_and_predicted_poses():
    assert "record['target']" not in inspect.getsource(associate_source_assets)
    assert 'target_xy' not in inspect.getsource(associate_source_assets)
    record = dict(source=scene_dict(_scene()))
    output, objects = _packed([(.15, .20), (.40, .20)], [(.22, .30), (.48, .12)])
    found = associate_source_assets(record, _visual([(.15, .20), (.40, .20)]), output, objects)
    assert [item['object_id'] for item in found['matched']] == ['a', 'b']
    assert found['omitted'] == []
    view = scene_at_predicted_poses(found)
    placed = {obj.object_id: obj for obj in view.objects}
    assert placed['a'].asset.asset_id == 'a'
    assert placed['a'].source_pose == Pose(.15, .20, 0.)
    assert torch.allclose(torch.tensor([placed['a'].target_pose.x_m, placed['a'].target_pose.y_m]),
                          torch.tensor([.22, .30]))
    assert torch.allclose(torch.tensor([placed['b'].target_pose.x_m, placed['b'].target_pose.y_m]),
                          torch.tensor([.48, .12]))
    assert view.seed == 7


def test_an_unmatched_asset_is_omitted_instead_of_reassigned():
    record = dict(source=scene_dict(_scene()))
    output, objects = _packed([(.15, .20), (.05, .05)], [(.22, .30), (.50, .30)])
    found = associate_source_assets(record, _visual([(.15, .20), (.05, .05)]), output, objects)
    assert [item['object_id'] for item in found['matched']] == ['a']
    assert found['omitted'] == ['b']
    try:
        scene_at_predicted_poses(found)
    except ValueError as error:
        assert '2 to 8' in str(error)
    else:
        raise AssertionError('one correspondence must not grow into a full scene')

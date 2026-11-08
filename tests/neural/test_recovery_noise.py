import pytest
import torch

from knolling_model.recovery.contracts import oracle_objects
from knolling_model.recovery.noise import perturb_sources
from test_recovery_perception import truth


def test_noise_is_explicit_reproducible_and_does_not_modify_labels():
    labels = truth()
    objects = oracle_objects([labels])
    original = objects.xy.clone()
    options = dict(xy_std_m=.001, size_std_m=.0005, angle_std_rad=.01)
    a = perturb_sources(objects, **options, generator=torch.Generator().manual_seed(1))
    b = perturb_sources(objects, **options, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a.xy, b.xy)
    torch.testing.assert_close(objects.xy, original)
    torch.testing.assert_close(labels['source_xy'], original[0])
    torch.testing.assert_close(a.query_index, objects.query_index)
    assert (a.size[..., 0] >= a.size[..., 1]).all() and not torch.allclose(a.xy, original)
    with pytest.raises(ValueError, match='scales'):
        perturb_sources(objects, **{**options, 'xy_std_m': -1}, generator=torch.Generator())

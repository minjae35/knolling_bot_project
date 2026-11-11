"""Load the adopted checkpoints and predict target poses from RGB and text."""
import time

from knolling_model.experiments.inference import SCOPE, format_poses
from knolling_model.experiments.size_refinement import poses_from_detection
from score_end_to_end import load_bundle


def load(size_path=None):
    if size_path is None:
        return load_bundle()
    return load_bundle(size_path)


def predict(system, relation, row_model, head, image, text):
    """One BEV image and one instruction. Labels are not arguments."""
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('predict one image at a time')
    image = image.to(next(system.perception.parameters()).device)
    started = time.perf_counter()
    output, objects, _ = poses_from_detection(
        system, relation, row_model, image, [text], head)
    elapsed = time.perf_counter()-started
    valid = objects.valid[0].detach().cpu()
    poses = format_poses(
        output['xy'][0].detach().cpu(), output['angle'][0].detach().cpu(),
        objects.size[0].detach().cpu(), objects.color[0].detach().cpu(), valid)
    return dict(poses=poses, objects=len(poses), seconds=elapsed, scope=SCOPE)

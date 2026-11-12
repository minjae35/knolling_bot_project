"""RGB and language in, target poses out, for the adopted knolling checkpoints.

Supported tasks use 1 to 8 detected objects. A full command arranges every
object. Partial moves the goal encoder's selection and keeps the rest on the
detected source pose. No-op returns that source pose for every object.
Inference does not read a target pose, a teacher row, or an order.
"""
import math

import torch
from PIL import Image, ImageDraw

from knolling_dataset.schema import COLORS, HEIGHT_M, WIDTH_M
from knolling_model.experiments.full_arrangement import is_full

SCOPE = dict(
    supported='full, partial, and no-op arrangements of 1 to 8 objects',
    inputs=('bev_rgb', 'natural_language'),
    outputs=('x_m', 'y_m', 'theta_rad'),
    not_supported=(),
    inference_uses=(
        'rt_detr', 'size_refinement', 'goal_encoder', 'relation_head', 'hierarchical_arrangement',
        'constraint_decode', 'geometry_decoder'),
    inference_does_not_use=('ground_truth', 'teacher_row', 'teacher_order'),
)
CAPTION = 'Supported: full arrangement, partial selection, and no-op.'
UNSUPPORTED = 'UNSUPPORTED instruction.'
_PALETTE = {'red': (214, 70, 64), 'green': (64, 150, 86), 'blue': (55, 110, 196)}


def instruction_supported(record):
    """Full, partial, and no-op scenes of 1 to 8 objects."""
    instruction = record['instruction']
    identities = [obj['object_id'] for obj in record['source']['objects']]
    if not 1 <= len(identities) <= 8:
        return False
    if record['mode'] == 'full':
        return bool(is_full(record))
    if record['mode'] == 'no_op':
        return bool(instruction['keep_all'])
    if record['mode'] != 'partial' or instruction['keep_all']:
        return False
    selected, locked = set(instruction['selected_ids']), set(instruction['locked_ids'])
    return bool(selected and locked and selected.isdisjoint(locked) and selected | locked == set(identities))


def accepted_success(record, task_success):
    """An unsupported instruction cannot become a successful scene."""
    return bool(task_success) and instruction_supported(record)


def format_poses(xy, angle, size, color, valid):
    """One scene of packed tensors to pose dictionaries. No labels are read."""
    if xy.ndim != 2 or angle.ndim != 1 or size.ndim != 2 or color.ndim != 2:
        raise ValueError('format_poses expects a single scene')
    count = int(valid.sum())
    poses = []
    for index in range(count):
        poses.append(dict(
            x_m=float(xy[index, 0]), y_m=float(xy[index, 1]), theta_rad=float(angle[index]),
            width_m=float(size[index, 0]), height_m=float(size[index, 1]),
            color=COLORS[int(color[index].argmax())]))
    return poses


def render_comparison(image, poses, supported):
    """Input BEV on the left, predicted rectangles on the right."""
    photo = _to_image(image)
    width, height = photo.size
    layout = Image.new('RGB', (width, height), (244, 242, 236))
    draw = ImageDraw.Draw(layout)
    draw.rectangle((1, 1, width-2, height-2), outline=(180, 176, 168))
    for pose in poses:
        draw.polygon(_corners(pose, width, height), fill=_PALETTE[pose['color']], outline=(30, 30, 30))
    banner = UNSUPPORTED if not supported else CAPTION
    bar = 28
    canvas = Image.new('RGB', (width*2+8, height+bar), (255, 255, 255))
    canvas.paste(photo, (0, bar))
    canvas.paste(layout, (width+8, bar))
    ink = (160, 40, 32) if not supported else (40, 40, 40)
    ImageDraw.Draw(canvas).text((8, 6), banner, fill=ink)
    return canvas


def _to_image(image):
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('expected one CHW RGB image')
    if image.is_floating_point():
        array = image.detach().clamp(0, 1).mul(255).byte()
    else:
        array = image.detach().byte()
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode='RGB')


def _corners(pose, width, height):
    half_a, half_b = pose['width_m']/2, pose['height_m']/2
    cosine, sine = math.cos(pose['theta_rad']), math.sin(pose['theta_rad'])
    points = []
    for along, across in ((-half_a, -half_b), (half_a, -half_b), (half_a, half_b), (-half_a, half_b)):
        x = pose['x_m']+cosine*along-sine*across
        y = pose['y_m']+sine*along+cosine*across
        points.append((x/WIDTH_M*width, (1-y/HEIGHT_M)*height))
    return points

"""One validation scene each of full, partial, and no-op. Does not score the holdout."""
import json
import math
import os
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HOME', '/workspace/cache/huggingface')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from PIL import Image, ImageDraw, ImageFont
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable

from knolling_dataset.schema import COLORS, HEIGHT_M, WIDTH_M, angle_error
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import format_poses, instruction_supported
from knolling_model.experiments.partial_arrangement import HELD_BELOW
from knolling_model.experiments.size_refinement import poses_from_detection
from score_end_to_end import DATA, judge_scene, load_adopted

OUT = Path('/tmp/v2_validation_smoke')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PALETTE = {'red': (214, 70, 64), 'green': (64, 150, 86), 'blue': (55, 110, 196)}


def first_of_each(records):
    picked = {}
    for index, record in enumerate(records):
        mode = record['mode']
        if mode in picked or mode not in ('full', 'partial', 'no_op'):
            continue
        if not instruction_supported(record):
            continue
        picked[mode] = index
        if len(picked) == 3:
            break
    missing = {'full', 'partial', 'no_op'} - set(picked)
    if missing:
        raise RuntimeError(f'validation is missing a supported scene for {sorted(missing)}')
    return picked


def match(objects, truth):
    count = int(objects.valid[0].sum())
    distance = torch.cdist(objects.xy[0, :count].detach().cpu(), truth['source_xy'])
    used_query, used_gt, pairs = set(), set(), []
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs.append((query, gt))
    return count, pairs


def corners(pose, width, height):
    half_a, half_b = pose['width_m'] / 2, pose['height_m'] / 2
    cosine, sine = math.cos(pose['theta_rad']), math.sin(pose['theta_rad'])
    points = []
    for along, across in ((-half_a, -half_b), (half_a, -half_b), (half_a, half_b), (-half_a, half_b)):
        x = pose['x_m'] + cosine * along - sine * across
        y = pose['y_m'] + sine * along + cosine * across
        points.append((x / WIDTH_M * width, (1 - y / HEIGHT_M) * height))
    return points


def panel(photo, poses, instruction):
    width, height = photo.size
    layout = Image.new('RGB', (width, height), (244, 242, 236))
    draw = ImageDraw.Draw(layout)
    draw.rectangle((1, 1, width - 2, height - 2), outline=(180, 176, 168))
    for pose in poses:
        draw.polygon(corners(pose, width, height), fill=PALETTE[pose['color']], outline=(30, 30, 30))
    font = ImageFont.truetype(FONT, 16)
    bold = ImageFont.truetype(BOLD, 18)
    lines, current = [], ''
    for word in instruction.split():
        trial = word if not current else f'{current} {word}'
        if font.getlength(trial) <= width * 2 - 24:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    bar, gap, text_h = 36, 8, 28 + 22 * len(lines)
    canvas = Image.new('RGB', (width * 2 + gap, bar + height + text_h), (255, 255, 255))
    canvas.paste(photo, (0, bar))
    canvas.paste(layout, (width + gap, bar))
    pen = ImageDraw.Draw(canvas)
    pen.text((12, 8), 'Initial Scene', font=bold, fill=(20, 20, 20))
    pen.text((width + gap + 12, 8), 'Predicted Arrangement', font=bold, fill=(20, 20, 20))
    top = bar + height + 8
    for index, line in enumerate(lines):
        pen.text((12, top + index * 22), line, font=font, fill=(30, 30, 30))
    return canvas


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    data = ExpandedDataset(DATA, 'validation')
    chosen = first_of_each(data.records)
    system, relation, model, head, _saved = load_adopted()
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for mode in ('full', 'partial', 'no_op'):
        index = chosen[mode]
        record = data.records[index]
        image = data[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        text = record['instruction']['text']
        row = dict(
            mode=mode, split='validation', index=index, sample_id=record['sample_id'],
            family=record['scene_family_id'], instruction=text,
            selected_ids=list(record['instruction']['selected_ids']),
            locked_ids=list(record['instruction']['locked_ids']),
            keep_all=bool(record['instruction']['keep_all']),
            object_count=len(identities), pipeline_executed=False, placement_pass=None)
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [text], head)
        except Exception as exc:
            row['pipeline_error'] = f'{type(exc).__name__}: {exc}'
            report.append(row)
            continue
        row['pipeline_executed'] = True
        count = int(objects.valid[0].sum())
        poses = format_poses(
            output['xy'][0].detach().cpu(), output['angle'][0].detach().cpu(),
            objects.size[0].detach().cpu(), objects.color[0].detach().cpu(),
            objects.valid[0].detach().cpu())
        judged = judge_scene(
            output, objects, visual, record, identities, truth, 0, system.threshold, detail=False)
        detected, pairs = match(objects, truth)
        keep = float(output.get('keep_probability', torch.zeros(1))[0])
        gt_move = set() if record['instruction']['keep_all'] else set(record['instruction']['selected_ids'])
        objects_out = []
        selection_ok = []
        for query, gt in pairs:
            if output.get('moving_mask') is not None:
                pred_move = bool(output['moving_mask'][0, query])
            else:
                pred_move = keep <= 0.5 and float(output['selected'][0, query]) >= HELD_BELOW
            identity = identities[gt]
            delta = output['xy'][0, query].detach().cpu() - truth['source_xy'][gt]
            moved_mm = float(delta.norm()) * 1000
            moved_deg = math.degrees(angle_error(
                float(output['angle'][0, query].detach().cpu()), float(truth['source_angle'][gt])))
            should_move = identity in gt_move
            selection_ok.append(pred_move == should_move)
            objects_out.append(dict(
                object_id=identity, color=poses[query]['color'],
                detected_x_m=float(objects.xy[0, query].detach().cpu()[0]),
                detected_y_m=float(objects.xy[0, query].detach().cpu()[1]),
                detected_theta_rad=float(objects.angle[0, query].detach().cpu()),
                x_m=poses[query]['x_m'], y_m=poses[query]['y_m'], theta_rad=poses[query]['theta_rad'],
                theta_deg=math.degrees(poses[query]['theta_rad']),
                should_move=should_move, predicted_move=pred_move,
                shift_from_source_mm=moved_mm, turn_from_source_deg=moved_deg))
        photo = Image.open(data.root / record['source_render_dir'] / 'rgb.png').convert('RGB')
        image_path = OUT / f'{mode}.png'
        panel(photo, poses, text).save(image_path)
        row.update(
            route=str(output.get('route', '')),
            keep_probability=keep,
            detected_count=detected,
            detection_count_matches=detected == len(identities) == count,
            matched_to_source=len(pairs),
            selection_correct=len(pairs) == count == len(identities) and all(selection_ok),
            keep_decision=(keep > 0.5) == bool(record['instruction']['keep_all']),
            geometry=judged['geometry'],
            official_errors=judged['errors'],
            placement_pass=bool(judged['task']),
            objects=objects_out,
            image=str(image_path))
        report.append(row)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'objects'} | {
        'objects': row.get('objects', [])} for row in report], indent=2))


if __name__ == '__main__':
    main()

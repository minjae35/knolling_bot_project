"""Draw one validation full arrangement for 2, 4, 6, and 8 objects.

The right panel is a Blender render of the source assets at the predicted
poses. The left panel is the original BEV file. Dataset target poses are not
read. Unmatched assets are omitted.
"""
import json
from pathlib import Path

import run_phase1_pairwise_affine as runner
from PIL import Image, ImageDraw, ImageFont

from diagnose_row_transition import multi_row
from knolling_dataset.rendering import make_job, render_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import instruction_supported
from knolling_model.experiments.predicted_view import associate_source_assets, scene_at_predicted_poses
from knolling_model.experiments.size_refinement import poses_from_detection
from pipeline import load
from score_end_to_end import DATA, score_rgb
import run_relation_stable as stable

OUT = Path('artifacts/experiments/end_to_end/demos')
RENDER = Path('artifacts/experiments/end_to_end/renders')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def wrap(text, font, width):
    lines, current = [], ''
    for word in text.split():
        trial = word if not current else f'{current} {word}'
        if font.getlength(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def compose(before, after, instruction, count):
    """Paste the original BEV unchanged beside the rendered prediction."""
    if before.size != after.size or before.mode != 'RGB' or after.mode != 'RGB':
        raise ValueError('before and predicted renders must be the same RGB size')
    font = ImageFont.truetype(FONT, 18)
    bold = ImageFont.truetype(BOLD, 18)
    margin = 16
    lines = wrap(instruction, font, before.width*2+8-margin*2)
    line_height = 26
    bar = 32
    block = 44+line_height*len(lines)+16
    canvas = Image.new('RGB', (before.width*2+8, bar+before.height+block), (255, 255, 255))
    canvas.paste(before, (0, bar))
    canvas.paste(after, (before.width+8, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 6), f'Before, {count} objects', font=bold, fill=(20, 20, 20))
    draw.text((before.width+8+margin, 6), 'Predicted After', font=bold, fill=(20, 20, 20))
    top = bar+before.height+12
    draw.text((margin, top), 'Instruction', font=bold, fill=(20, 20, 20))
    for index, line in enumerate(lines):
        draw.text((margin, top+32+index*line_height), line, font=font, fill=(30, 30, 30))
    return canvas


def choose(records, count):
    return [index for index, record in enumerate(records)
            if multi_row(record) and instruction_supported(record)
            and len(record['source']['objects']) == count]


def main():
    runner.setup()
    data = ExpandedDataset(DATA, 'validation')
    system, relation, row_model, head, checkpoint = load()
    OUT.mkdir(parents=True, exist_ok=True)
    RENDER.mkdir(parents=True, exist_ok=True)
    pending, report = [], []
    for count in (2, 4, 6, 8):
        skipped = 0
        chosen = None
        for index in choose(data.records, count):
            record = data.records[index]
            item = data[index]
            image = item['image'][None].to(stable.DEVICE)
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, row_model, image, [record['instruction']['text']], head)
            except ValueError:
                skipped += 1
                continue
            association = associate_source_assets(record, visual, output, objects, system.threshold)
            try:
                view = scene_at_predicted_poses(association)
            except ValueError:
                skipped += 1
                continue
            folder = RENDER/f'count_{count}'
            job = make_job(view, 'target_pose', folder, record['sample_id'], render_seed=view.seed)
            scored = score_rgb(system, relation, row_model, head, data, [index])
            before = Image.open(data.root/record['source_render_dir']/'rgb.png').convert('RGB')
            chosen = dict(
                objects=count, index=index, sample_id=record['sample_id'],
                instruction=record['instruction']['text'], mode=record['mode'],
                predicted_objects=len(association['matched']),
                matched=[item['object_id'] for item in association['matched']],
                omitted=association['omitted'],
                task_success=bool(scored['task'] == 1),
                errors=[] if scored['task'] == 1 else sorted(scored['failure_counts']),
                detection_failures_skipped=skipped,
                checkpoint_update=checkpoint.get('update'),
                used_gt_target_pose=False)
            pending.append((count, before, job, chosen))
            break
        if chosen is None:
            raise SystemExit(f'no corresponded {count}-object validation scene')
    render_batch([item[2] for item in pending], RENDER/'work')
    for count, before, job, chosen in pending:
        after = Image.open(Path(job['output_dir'])/'rgb.png').convert('RGB')
        image = compose(before, after, chosen['instruction'], count)
        path = OUT/f'demo_{count}.png'
        image.save(path)
        changed = sum(abs(a-b) for a, b in zip(before.tobytes(), after.tobytes()))/len(before.tobytes())
        chosen['image'] = str(path)
        chosen['mean_abs_pixel_delta'] = changed
        report.append(chosen)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

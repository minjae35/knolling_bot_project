#!/usr/bin/env python3
"""One bounded full-76 split-pose run; no calibration, automatic retry or stages."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import torch
from tqdm import tqdm

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from knolling_model.runner import seed_everything
from knolling_model.data import S0Dataset, Collator
from knolling_model.recovery.dataset_v3 import RecoveryDatasetV3
from knolling_model.recovery.goal_labels import GoalDataset
from knolling_model.recovery.model import load_recovery
from knolling_model.recovery.runtime import FrozenTextCache, warm_text_cache
from knolling_model.recovery.contracts import oracle_objects
from knolling_model.recovery.stages import aligned_targets
from knolling_model.recovery.driver import cpu_copy, rng_state, save_json_exclusive, load_checkpoint
from knolling_model.recovery.storage import CheckpointStore, common_store, load_artifact, tensor_digest
from knolling_model.recovery.split_pose import load_model_state
from knolling_model.recovery.relative_task import SCHEMA, configure, pair_indices, losses, best_key, restore
from knolling_model.recovery.alternating import evaluate_alternating
from knolling_model.recovery.vertical_gap import POLICY as GAP_POLICY, supervision as gap_supervision, loss as gap_loss
from run_recovery_stages import register_terminal_run


def run(config_path, output, cpu_check=False):
    c = json.loads(Path(config_path).read_text())
    refine = c.get('task_refinement') == GAP_POLICY
    if (c['schema'] != SCHEMA or c['max_updates'] != 1200 or c['max_seconds'] != 1200
            or c['loss_weights'] != [1, 1, 1, 1, 1]
            or (c.get('task_refinement') is not None and not refine)
            or (refine and c.get('freeze_orientation') is not True)): raise ValueError('unapproved run contract')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); elapsed = lambda: time.monotonic()-start
    result = dict(status='FAIL', updates=0, cpu_check=cpu_check)
    best_path = None
    def log(event, **data):
        with (output/'history.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event, elapsed_seconds=elapsed(), **data), allow_nan=False)+'\n')
    try:
        if shutil.disk_usage(output).free < 3*1024**3: raise RuntimeError('less than 3 GiB free')
        source = Path(c['source'])
        if file_hash(source) != c['source_sha256']: raise ValueError('source checkpoint changed')
        raw = load_artifact(source)
        if refine:
            if (raw['schema'] != SCHEMA or raw['updates'] != 1200
                    or raw['config_sha256'] != digest(raw['config'])): raise ValueError('not pinned relative-task Best')
            donor = raw
            # Preserve the original dataset contract through the immutable donor chain.
            parent_path = Path(donor['config']['source'])
            if file_hash(parent_path) != donor['config']['source_sha256']: raise ValueError('dataset ancestor changed')
            parent = load_artifact(parent_path)
            if parent['dataset_fingerprint'] != donor['dataset_fingerprint']: raise ValueError('dataset ancestor mismatch')
            dc = parent['config']
        else:
            donor = load_checkpoint(source, raw['config'], raw['dataset_fingerprint'], raw['commit'])
            if donor['status'] != 'passed' or donor['cursor']['update'] != 2: raise ValueError('not validated split candidate')
            dc = donor['config']
        commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        device = 'cpu' if cpu_check else 'cuda'
        seed_everything(c['seed'], c['cpu_threads'])
        if not cpu_check:
            if '4090' not in torch.cuda.get_device_name(): raise RuntimeError('RTX 4090 required')
            register_terminal_run(output, c['max_updates'])
        d = GoalDataset(RecoveryDatasetV3(S0Dataset(dc['dataset'], dc['selection']), dc['data_v3']['path'], dc['data_v3']['sha256']))
        if len(d) != 76 or len(d.pairs) != 35 or d.fingerprint != donor['dataset_fingerprint']:
            raise ValueError('unexpected data/contrast suite')
        m, tok = load_recovery('configs/recovery_v4_review.json')
        if refine: restore(m, donor)
        else: load_model_state(m, donor)
        m.to(device)
        # Recollect references after device transfer.
        groups = {k:v for k,v in configure(m, orientation=not refine).items() if v}
        optimizers = {k: torch.optim.AdamW(v, lr=c['lr'], weight_decay=c['weight_decay'], foreach=False) for k,v in groups.items()}
        m.encode_text = FrozenTextCache(m.encode_text); col = Collator(tok)
        samples = [d[i] for i in range(len(d))]
        full = col(samples).to(device)
        batches = [col(samples[i:i+8]).to(device) for i in range(0, len(d), 8)]
        warm_text_cache(m, d, col, 8, device)
        objects = oracle_objects(full.supervision)
        indices = [torch.arange(len(t['source_xy']), device=device) for t in full.supervision]
        targets = aligned_targets(full.supervision, indices, objects)
        pairs = pair_indices(full.metadata, d.pairs, device)
        gap_labels = gap_supervision(full.metadata) if refine else []
        ids = [r['sample_id'] for r in d.records]
        def forward():
            return m.layout(objects, m.encode_text(full.inputs['input_ids'], full.inputs['attention_mask']), full.inputs['attention_mask'])
        def objective():
            out = forward(); terms = losses(out, targets, m.layout.orientation, pairs)
            if refine: terms['vertical_edge_gap_huber'] = gap_loss(out, objects, gap_labels)
            return out, terms
        def evaluate():
            return evaluate_alternating(m, batches, stage='oracle_layout', expected_ids=ids, pairs=d.pairs)
        store = CheckpointStore(common_store(output))
        def save(update, metrics):
            path = output/f'checkpoint-{update:06d}.pt'
            store.save_checkpoint(path, dict(schema=SCHEMA, stage='oracle_layout', config=c, config_sha256=digest(c),
                commit=commit, dataset_fingerprint=d.fingerprint, source_sha256=c['source_sha256'],
                model=m.state_dict(), optimizers={k:opt.state_dict() for k,opt in optimizers.items()},
                rng=rng_state(), updates=update, elapsed_seconds=elapsed(), metrics=metrics,
                batch_order=ids, pairs=d.pairs, best_checkpoint=str(best_path) if best_path else None))
            return path
        baseline, rows = evaluate()
        if refine and best_key(baseline)[:2] != (18,2): raise ValueError('baseline task metrics differ from pinned Best')
        def rank(metrics):
            key = best_key(metrics)
            if not refine: return key
            eligible = (key[0]>18 and key[1]>2 and
                        metrics['integrated_evidence']['angle_errors_deg']==baseline['integrated_evidence']['angle_errors_deg'])
            return (eligible, *key)
        save_json_exclusive(output/'baseline.json', dict(metrics=baseline, rows=rows))
        log('evaluation', update=0, metrics=baseline)
        best_metrics = deepcopy(baseline); best_path = save(0, baseline)
        def frozen_state():
            return {k:v for k,v in m.state_dict().items() if not k.startswith('layout.pose.') or
                    (refine and k.startswith('layout.pose.orientation_pose.'))}
        frozen = tensor_digest(frozen_state())
        if cpu_check:
            out, terms = objective()
            sum(terms.values()).backward()
            for key, ps in groups.items():
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in ps), key
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ps)
            assert all(p.grad is None for name,p in m.named_parameters() if not name.startswith('layout.pose.'))
            if refine: assert all(p.grad is None for p in m.layout.pose.orientation_pose.parameters())
            # No real optimizer step. Reload actual outputs exactly, including frozen Goal.
            saved = load_artifact(best_path); restore(m,saved)
            with torch.no_grad(): again = forward()
            for key in ('xy','angle','angle_logits','angle_residual'):
                torch.testing.assert_close(out[key],again[key],rtol=0,atol=0)
            assert not torch.cuda.is_initialized()
            result.update(status='PASS', loss={k:float(v.detach()) for k,v in terms.items()}, optimizer_steps=0)
        else:
            stop = 'update_budget'; last_evaluated = 0; bar = tqdm(total=c['max_updates'], desc='Full76 split-relative', unit='update')
            for update in range(1,c['max_updates']+1):
                if elapsed() >= c['max_seconds']-c['final_reserve_seconds']:
                    stop = 'time_budget'; break
                lr = c['min_lr']+(c['lr']-c['min_lr'])*.5*(1+math.cos(math.pi*(update-1)/c['max_updates']))
                for opt in optimizers.values():
                    opt.zero_grad(set_to_none=True)
                    for group in opt.param_groups: group['lr'] = lr
                _, terms = objective(); loss = sum(terms.values())
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward()
                norms = {k: float(torch.nn.utils.clip_grad_norm_(ps, c['gradient_clip'], error_if_nonfinite=True)) for k,ps in groups.items()}
                for opt in optimizers.values(): opt.step()
                if not all(torch.isfinite(p).all() for ps in groups.values() for p in ps): raise FloatingPointError('nonfinite model')
                result['updates'] = update
                log('update', update=update, lr=lr, loss=float(loss.detach()), losses={k:float(v.detach()) for k,v in terms.items()}, gradient_norms=norms)
                bar.update(1);bar.set_postfix(loss=f'{float(loss.detach()):.4f}', remaining_s=f'{c["max_seconds"]-elapsed():.0f}', refresh=False)
                if update%c['evaluate_every']==0 or update==c['max_updates']:
                    metrics, rows = evaluate(); last_evaluated = update; path = save(update,metrics)
                    save_json_exclusive(output/f'evaluation-{update:06d}.json', dict(metrics=metrics, rows=rows))
                    log('evaluation', update=update, metrics=metrics)
                    if rank(metrics)>rank(best_metrics): best_metrics, best_path = deepcopy(metrics),path
                    tqdm.write(f'update={update} validity={best_key(metrics)[0]}/76 contrasts={best_key(metrics)[1]}/35 xy_p95={metrics["target_position_p95_mm"]:.3f} angle_p95={metrics["target_angle_p95_deg"]:.3f}')
                    if best_key(metrics)[:2] == (76,35): stop='task_success';break
            bar.close()
            if result['updates'] != last_evaluated:
                metrics,rows=evaluate();path=save(result['updates'],metrics)
                save_json_exclusive(output/f'evaluation-{result["updates"]:06d}.json',dict(metrics=metrics,rows=rows))
                log('evaluation',update=result['updates'],metrics=metrics)
                if rank(metrics)>rank(best_metrics):best_metrics,best_path=deepcopy(metrics),path
            # Exact tensor reload plus independent actual-forward evaluation of Best.
            chosen=load_artifact(best_path);restore(m,chosen)
            if tensor_digest(m.state_dict())!=tensor_digest(chosen['model']):raise ValueError('reload mismatch')
            reloaded,reload_rows=evaluate()
            if reloaded != best_metrics: raise ValueError('reloaded evaluation differs')
            save_json_exclusive(output/'best-evaluation.json',dict(metrics=reloaded,rows=reload_rows))
            adopted = (rank(reloaded)[0] if refine else best_key(reloaded)[0]>4 and best_key(reloaded)[1]>0)
            result.update(status='PASS' if adopted else 'FAIL',
                stop_reason='memorization_gate' if stop=='task_success' else 'update_budget', termination=stop,
                s0_gate=best_key(reloaded)[:2]==(76,35), best_update=chosen['updates'], baseline=baseline,
                best_metrics=reloaded, reload_exact=True, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        assert file_hash(source)==c['source_sha256']
        assert frozen==tensor_digest(frozen_state())
        result.update(best_checkpoint=str(best_path.resolve()), original_unchanged=True, frozen_modules_unchanged=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', termination='error', best_checkpoint=str(best_path) if best_path else None)
    finally:
        result['elapsed_seconds']=elapsed()
        save_json_exclusive(output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','best_metrics')},indent=2),flush=True)
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu-check',action='store_true')
    a=p.parse_args();raise SystemExit(run(a.config,a.output,a.cpu_check))

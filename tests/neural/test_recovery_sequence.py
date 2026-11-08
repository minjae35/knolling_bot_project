import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('outcome,count,reason', [
    ('budget_exhausted',2,'update_budget'), ('rolled_back',2,'stage_failure'),
    ('early_stopped',2,'stage_failure'), ('passed',4,'memorization_gate')])
def test_stage_orchestrator_stops_without_retry_or_extension(tmp_path,outcome,count,reason):
    spec=importlib.util.spec_from_file_location('recovery_sequence','scripts/run_recovery_stages.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    def execute(command, check):
        stage=command[command.index('--stage')+1]
        path=Path(command[command.index('--output')+1])
        path.mkdir()
        status='passed' if stage=='oracle_layout' else outcome
        (path/'result.json').write_text(json.dumps(dict(status=status,checkpoint=str(path/'checkpoint.pt'))))
        calls.append(command)
        return SimpleNamespace(returncode=0 if status=='passed' else 2)
    result=module.run_stages('configs/recovery_gpu_initial.json',tmp_path/'approval',tmp_path/'sequence',execute=execute)
    assert len(calls)==count and '--parent-checkpoint' in calls[1]
    assert result['status']==('passed' if outcome=='passed' else 'stopped') and result['stop_reason']==reason
    assert result['automatic_extension'] is False


def test_extension_is_forwarded_once_and_cannot_admit_stage_two_without_pass(tmp_path):
    spec=importlib.util.spec_from_file_location('recovery_sequence','scripts/run_recovery_stages.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def execute(command,check):
        calls.append(command)
        path=Path(command[command.index('--output')+1]);path.mkdir()
        (path/'result.json').write_text(json.dumps(dict(status='early_stopped',checkpoint='unused')))
        return SimpleNamespace(returncode=2)
    result=module.run_stages('configs/recovery_gpu_initial.json','approval',tmp_path/'out',
                            extend_from='source.pt',execute=execute)
    assert len(calls)==1 and calls[0][-2:]==['--extend-from','source.pt']
    assert list(result['stages'])==['oracle_layout']


@pytest.mark.parametrize('status',['passed','budget_exhausted'])
def test_explicit_stage_one_scope_never_launches_perception(tmp_path,status):
    spec=importlib.util.spec_from_file_location('recovery_sequence','scripts/run_recovery_stages.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def execute(command,check):
        calls.append(command)
        path=Path(command[command.index('--output')+1]);path.mkdir()
        (path/'result.json').write_text(json.dumps(dict(status=status,checkpoint='unused')))
        return SimpleNamespace(returncode=0 if status=='passed' else 2)
    result=module.run_stages('configs/recovery_v3_gpu_stage1.json','approval',tmp_path/'out',
                            stop_after='oracle_layout',execute=execute)
    assert len(calls)==1 and result['requested_stages']==['oracle_layout']
    assert result['requested_stages_passed']==(status=='passed')
    assert result['s0_gate'] is False  # Stage 1 is not end-to-end system completion.

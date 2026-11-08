import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('outcome,count,reason', [
    ('budget_exhausted',2,'update_budget'), ('rolled_back',2,'stage_failure'), ('passed',4,'memorization_gate')])
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

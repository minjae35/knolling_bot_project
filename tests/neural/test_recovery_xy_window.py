"""Provisional exploration is never an accepted rollback anchor."""
from copy import deepcopy
import pytest
import torch
from knolling_model.arrangement import alternating as alt
from knolling_model.arrangement.integrated import load_integrated
from knolling_model.arrangement.svd_readout import SVDXYFitter
from knolling_model.arrangement.driver import read_config, authorize
from test_recovery_integrated import equal
from test_recovery_xy_diagnostic import xy_build
from test_recovery_alternating import Crash, diagnostic


def build_window(tmp_path, monkeypatch, success=True):
    c, source, make = xy_build(tmp_path, monkeypatch, 'cpu', 'fail')
    c['alternating_pose'] = deepcopy(alt.XY_WINDOW_POLICY)
    prior = torch.load(source, map_location='cpu', weights_only=False)
    prior['metrics']['integrated_evidence']['successful_scenes'] = ['0']
    monkeypatch.setattr(alt, 'load_source', lambda *a: deepcopy(prior))
    def measured(self):
        value = deepcopy(self.baseline)
        if self.cursor.update < 2 or not success:
            value['integrated_evidence']['successful_scenes'] = []
        return value
    monkeypatch.setattr(alt.AlternatingEngine, 'measured', measured)
    def diag(self):
        if not hasattr(self,'state') or 'reference_trial' not in self.state: return diagnostic()
        if self.cursor.update < 2 or not success: return diagnostic(rms=22.,hv=29.,angle=90.)
        return diagnostic(rms=22.,hv=29.)
    monkeypatch.setattr(alt.AlternatingEngine,'diagnostics',diag)
    def stall(*a,**kw): raise alt.SVDNumericalStall({'fixture':True})
    monkeypatch.setattr(SVDXYFitter,'step',stall)
    return c,source,make,prior


@pytest.mark.parametrize('success',[True,False])
@pytest.mark.parametrize('boundary',['candidate','feature','phase'])
def test_window_records_regression_and_resumes_without_promoting(tmp_path,monkeypatch,success,boundary):
    c,source,make,prior=build_window(tmp_path,monkeypatch,success)
    whole=make('whole').run('oracle_layout',parent=source)
    assert (whole['status'],whole['updates'])==(('passed',2) if success else ('rolled_back',10))
    paths=[]
    def crash(event,**r):
        if ((boundary=='candidate' and event=='candidate_evaluation' and r['update']==1)
          or (boundary=='feature' and event=='update' and r['update']==1)
          or (boundary=='phase' and event=='phase_transition' and r['update']==1)):
            pattern='readout-*.pt' if boundary=='candidate' else 'checkpoint-*.pt'
            paths.append(sorted((tmp_path/'part').glob(pattern))[-1]);raise Crash()
    with pytest.raises(Crash): make('part',observer=crash).run('oracle_layout',parent=source)
    resumed=make('resumed').run('oracle_layout',resume=paths[0])
    assert (resumed['status'],resumed['updates'])==(whole['status'],whole['updates'])
    a=load_integrated(whole['checkpoint'],c,'fixture','test');b=load_integrated(resumed['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor','metrics'):equal(a[k],b[k])
    equal(a['integrated']['candidate_history'],b['integrated']['candidate_history'])
    history=a['integrated']['candidate_history']
    assert history[0]['lost_scenes']==['0'] and not history[0]['passed']
    assert history[1]['lost_scenes']==['0'] and not history[1]['passed']
    assert a['integrated']['anchor_update']==0 and a['purpose']=='diagnostic'
    equal(a['baseline'],prior['metrics'])
    if not success:
        equal(a['model'],prior['model'])
        equal(a['optimizer'],a['integrated']['anchor_optimizer'])
        equal(a['rng'],a['integrated']['anchor_rng'])
    with pytest.raises(ValueError,match='terminal'):make('again').run('oracle_layout',resume=resumed['checkpoint'])


def test_window_deadline_rejects_unfinished_candidate(tmp_path,monkeypatch):
    c,source,make,prior=build_window(tmp_path,monkeypatch);now=[0.]
    def expire(event,**r):
        if event=='update':now[0]=120.
    end=make('timeout',observer=expire,clock=lambda:now[0]).run('oracle_layout',parent=source)
    assert end['status']=='rolled_back' and end['updates']==1
    s=load_integrated(end['checkpoint'],c,'fixture','test');equal(s['model'],prior['model'])


def test_window_nonfinite_is_not_provisional(tmp_path,monkeypatch):
    c,source,make,prior=build_window(tmp_path,monkeypatch)
    original=alt.AlternatingEngine.diagnostics
    monkeypatch.setattr(alt.AlternatingEngine,'diagnostics',lambda self:dict(original(self),**({'xy_svd_rms_mm':float('nan')} if self.cursor.update else {})))
    end=make('nan').run('oracle_layout',parent=source)
    assert end['status']=='rolled_back' and end['updates']==1 and 'nonfinite' in end['error']
    s=load_integrated(end['checkpoint'],c,'fixture','test');equal(s['model'],prior['model'])


def test_window_requires_separate_approval():
    c=read_config('configs/recovery_v4_xy_window_review.json');alt.validate_config(c)
    assert c['alternating_pose']==alt.XY_WINDOW_POLICY and not c['allow_training']
    with pytest.raises(PermissionError):authorize(c,{},'oracle_layout','data','commit')

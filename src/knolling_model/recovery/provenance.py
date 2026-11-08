"""Explicit evidence transfer; never an exhausted-run resume or a budget grant."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess

import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_FILES = {
    'model': ['knolling_model/recovery/'+p+'.py' for p in
              ('model','goal_encoder','goal_joint_layout','contracts','layout','joint_layout','goal_labels')],
    'loss': ['knolling_model/recovery/stages.py','knolling_model/losses.py','knolling_model/coordinates.py'],
    'teacher': ['knolling_dataset/'+p+'.py' for p in ('teacher_v3','teacher_continuous','teacher_balanced','teacher_legacy','schema','validator')],
    'evaluator': ['knolling_model/recovery/'+p+'.py' for p in ('evaluation','evaluation_v3','evaluation_v4')]
                 + ['knolling_model/evaluation.py'],
}
# Only reporting fields are ignored. They do not feed any gate or prediction.
REPORT_FIELDS = {'micro_predictions', 'integrated_evidence', 'breakdown'}
ALLOWED_CONFIG = {'integrated_pose','micro_precision','execution','allow_training','budget_status'}


class ContractAST(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module == 'family_manifest' and [n.name for n in node.names] == ['breakdown']:
            return None
        return node

    def visit_Assign(self,node):
        if len(node.targets)==1:
            t=node.targets[0]
            if (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='result'
                    and isinstance(t.slice,ast.Constant) and t.slice.value in REPORT_FIELDS):
                return None
        return self.generic_visit(node)

    def visit_If(self,node):
        node=self.generic_visit(node)
        return node if node.body else None


def contract_hashes(commit=None):
    result={}
    for group,files in CONTRACT_FILES.items():
        trees=[]
        for file in files:
            path='src/'+file
            source=(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT,text=True)
                    if commit else (ROOT/path).read_text())
            tree=ast.parse(source)
            if file.endswith('/evaluation_v4.py'):tree=ContractAST().visit(tree)
            trees.append([path,ast.dump(tree,include_attributes=False)])
        result[group]=digest(trees)
    return result


def load_evidence_transfer(path, config, fingerprint, grant):
    """A receipt pins BOTH code semantics and exactly which config fields differ."""
    from .driver import load_checkpoint
    if file_hash(path)!=grant.get('sha256'):raise ValueError('transfer checkpoint hash mismatch')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    old=raw.get('config',{})
    if raw.get('commit')!=grant.get('commit') or digest(old)!=grant.get('config_sha256'):
        raise ValueError('transfer source contract mismatch')
    changes={k for k in set(config)|set(old) if config.get(k)!=old.get(k)}
    if (not changes<=ALLOWED_CONFIG or changes!=set(grant.get('allowed_config_changes',[]))
            or grant.get('contracts')!=contract_hashes() or grant['contracts']!=contract_hashes(grant['commit'])):
        raise ValueError('transfer semantic/config contract mismatch')
    saved=load_checkpoint(path,old,fingerprint,grant['commit'])
    if saved.get('status')!='passed':raise ValueError('only passed evidence can transfer')
    return saved

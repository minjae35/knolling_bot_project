"""Explicit policy dispatch. Legacy default preserves existing Python callers."""
from .teacher_legacy import TeacherResult, _templates, _translations
from .policies import LEGACY, BALANCED, policy_config


def solve(scene, instruction, search_budget=20000, *, policy=LEGACY):
    policy_config(policy)
    if policy == LEGACY:
        from .teacher_legacy import solve as implementation
    else:
        from .teacher_balanced import solve as implementation
    return implementation(scene,instruction,search_budget)

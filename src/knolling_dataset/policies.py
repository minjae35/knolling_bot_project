"""Teacher version is data provenance, not inferred from installed code."""
import json
from pathlib import Path
from .schema import digest

LEGACY = "legacy_v1_2"
BALANCED = "balanced_rows_v1.3"
LEGACY_MANIFEST = "b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27"


def policy_config(policy):
    if policy not in (LEGACY,BALANCED):
        raise ValueError(f"unknown teacher policy: {policy}")
    return {"teacher_policy":policy,"translation_step_m":.005,"search_budget":20000,
            "comparison_decimals":12,"template_rule":"sequential_k_1_4" if policy == LEGACY else "sequential_balanced_k_1_N_plus_observed",
            "full_selection":"rows_area_width" if policy == LEGACY else "shape_area_rows_key",
            "partial_selection":"region_center_first" if policy == LEGACY else "global_move_shape_area_rows_key_y_x"}


def policy_metadata(policy, search_budget=20000):
    config = dict(policy_config(policy),search_budget=search_budget)
    return {"teacher_policy":policy,"teacher_config":config,"teacher_config_hash":digest(config)}


def resolve_dataset_policy(root, *, registered_legacy_hash=None):
    """Read-only registration: a known manifest hash is required for old data."""
    import hashlib
    root = Path(root)
    provenance = json.loads((root/"provenance.json").read_text())
    policy = provenance.get("teacher_policy")
    if policy:
        policy_config(policy)
        return policy
    actual = hashlib.sha256((root/"manifest.jsonl").read_bytes()).hexdigest()
    if registered_legacy_hash == actual:
        return LEGACY
    raise ValueError("unversioned dataset requires explicit legacy manifest registration")


def require_single_policy(policies):
    values = set(policies)
    if len(values) != 1:
        raise ValueError("mixed or empty teacher policy set")
    policy = next(iter(values))
    policy_config(policy)
    return policy

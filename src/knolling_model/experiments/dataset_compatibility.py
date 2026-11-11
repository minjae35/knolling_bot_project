"""One audited provenance transition; never ignore a schema or dataset hash."""
from knolling_dataset.schema import digest
from knolling_dataset.teacher_v3 import provenance

# Original immutable 20k annotations, and complete provenance objects (all
# Teacher files, validator, schema, configuration and their aggregate hashes).
ANNOTATION_SHA256 = '3e44cc3b623efa56d86a638d1f7fed51c0b4fd2e397a17ed72274e77d5509a04'
BEFORE_PROVENANCE = 'ff4f3460ab9ad66968f580fd376e2dfb4d27a1d9ae52df6f88cc0cfdb0520eea'
AFTER_PROVENANCE = '12c3cd00b5cc7f73209fe6167bebdc4a78abbf968c0e8fb7c4c076c707b8b3dc'
TRANSITION = 'aspect_ratio_1p3_four_ulp_v1'


def verify_expanded_teacher(recorded, annotation_sha256):
    """Return an auditable match type or reject before constructing samples.

    The sole legacy transition adds four ULPs to the dimensionless 1.3 aspect
    ratio comparison. No source coordinates, polygons, targets, gap/collision
    tolerances or Teacher algorithm changed. Historical records are not edited.
    A future source/config change needs a separate review, even if only comments.
    """
    current = provenance()
    if recorded == current:
        return 'exact'
    if (annotation_sha256 == ANNOTATION_SHA256
            and digest(recorded) == BEFORE_PROVENANCE
            and digest(current) == AFTER_PROVENANCE):
        return TRANSITION
    raise ValueError('expanded dataset Teacher/schema provenance mismatch: no audited transition')

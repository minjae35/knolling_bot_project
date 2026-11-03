# Dataset storage audit — 2026-11-03

Storage decision is pending user approval. No dataset was deleted, moved, added
to Git/LFS, or uploaded. Both legacy datasets received additional verified local
backups outside this checkout. No history rewrite is needed or recommended.

## Inventory

Logical byte sizes count every regular file, including local `.DS_Store` files.
Allocated disk usage differs because of filesystem allocation. All three contain
100 scene families and 130 samples. No symlinks were found in these trees.

| Dataset under `artifacts/` | Files | Logical bytes | MiB | Allocated usage |
| --- | ---: | ---: | ---: | ---: |
| `pilot_v1_2` | 1,041 | 77,739,767 | 74.14 | 77 MiB |
| `pilot_v1_3` | 1,614 | 131,037,402 | 124.97 | 138 MiB |
| `pilot_v1_3_en` (active) | 1,750 | 135,406,512 | 129.13 | 144 MiB |

The active dataset includes 81,326,671 bytes of `qc/`, 20,258,153 bytes of
`samples/`, 15,552,857 bytes of `scenes/`, 8,470,052 bytes of `legacy_targets/`,
4,201,770 bytes of `language_parent_annotations/`, 4,170,206 bytes of
`parent_annotations/`, plus manifests, provenance, sidecars and work metadata.
QC is about 60% of its size. This is a complete review/provenance artifact, not
just the minimum neural input payload. Preserve it intact for the initial transfer.

Largest individual files (bytes):

| Dataset | Relative file | Bytes |
| --- | --- | ---: |
| v1.2 | `work/render_jobs.json` | 1,064,939 |
| v1.2 | `qc/contact_sheet.png` | 928,860 |
| v1.2 | `qc/scene_0069_primary_source.png` | 203,527 |
| v1.2 | `qc/scene_0020_counterfactual_source.png` | 197,830 |
| v1.2 | `qc/scene_0020_primary_source.png` | 197,830 |
| v1.3 | `qc/migration_contact_sheet.png` | 1,146,313 |
| v1.3 | `qc/contact_sheet.png` | 911,409 |
| v1.3 | `qc/index.html` | 485,843 |
| v1.3 | `work/render_jobs.json` | 310,113 |
| v1.3 | `qc/scene_0069_primary_source.png` | 203,527 |
| v1.3 English | `qc/migration_contact_sheet.png` | 1,146,313 |
| v1.3 English | `qc/contact_sheet.png` | 911,409 |
| v1.3 English | `qc/index.html` | 318,815 |
| v1.3 English | `work/render_jobs.json` | 310,113 |
| v1.3 English | `qc/scene_0069_primary_source.png` | 203,527 |

## Legacy dependencies

| Consumer | Legacy dependency / consequence |
| --- | --- |
| `tests/test_pilot.py` | `PILOT_DATASET` selects the original v1.2 deliverable; tests exact legacy annotation reproduction and `COMPLETE.json`. |
| `tests/test_qc_v13.py` | `PILOT_V13_DATASET` selects the original v1.3 deliverable and its historical QC layout. Do not point this test at English QC. |
| `tests/test_language_migration.py` | Full migration regression needs `RUN_LANGUAGE_MIGRATION_TESTS=1` and the original `PILOT_V13_DATASET`; English-only acceptance uses `PILOT_EN_DATASET`. |
| `tests/test_migration.py` | Rendered integration generates temporary legacy fixtures; it does not require either archived on-disk pilot. This expensive integration is opt-in. |
| `migration.py`, `language_migration.py` | Reproducing each migration needs its parent dataset; routine S0 loading does not rerun migration. |
| `audit_v13.py`, `language_migration.py` audit | Active English audit reads copied `parent_*`, `parent_annotations/`, `language_parent_*`, source hashes and migration sidecars **inside the English directory**. Historical absolute parent-directory values are provenance, not a live path dependency. |
| `knolling_model/data.py`, `tests/neural/conftest.py` | S0 uses the English pilot and pinned selection. It does not load either external legacy directory. |
| README, `pilot-report.md`, `v1.3-pilot-report.md`, `v1.3-implementation.md`, `english-only-report.md`, specification | Historical commands, comparison results and provenance reference legacy versions. Retain those records; use explicit archival paths when running old acceptance tests. |

Removing the two external legacy folders would not remove the provenance copies
inside the active dataset, but would prevent their on-disk acceptance tests and
parent-based migration reproduction until restored. No removal is authorized.

## Verified backups

Location on the original Mac:

`/Users/minjae/Desktop/knolling_bot_project_backups/20261103T074433Z/`

| Archive | Bytes | Verified files |
| --- | ---: | ---: |
| `pilot_v1_2.tar.gz` | 72,772,287 | 1,041 |
| `pilot_v1_3.tar.gz` | 122,359,000 | 1,614 |

Archive SHA-256:

```text
c8650f8b3619b8b732b3d61ee68c9ac8e14ce13373627cb4bd7e559f1cdaed11  pilot_v1_2.tar.gz
4a19a4c73d547b443f9db5aefc6c425a40e074999a44f002232f0220d6608920  pilot_v1_3.tar.gz
```

Each companion `*.verification.json` records every relative path and SHA-256.
Verification read every archived member and compared the complete file mapping
against the originals, then rehashed the originals to confirm no concurrent
change. This verifies archive contents without overwriting or extracting over
the source. Any future restore should extract into an empty directory and compare
every restored file with that inventory before using it.

These backups share the original disk: they protect against an accidental folder
deletion, not disk loss. Before any future deletion decision, copy archives and
verification files to a separate disk or private remote storage and recheck their
hashes. Such remote storage changes still require approval.

## GitHub recommendation

The repository is private and datasets are currently excluded by `artifacts/`.
No dataset file is tracked. The existing Git object directory was about 2.5 MiB
at inspection, before this workflow documentation was added.

The English pilot **can fit in ordinary Git**: its largest individual file is
1.09 MiB. GitHub warns above 50 MiB and blocks files above 100 MiB; these limits
apply per file, not to the sum of the dataset. However, binary image revisions
would accumulate in Git history and slow future clones. See
[GitHub file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Recommended for this pilot: keep code/configurations/specifications in GitHub,
retain the complete active dataset locally, and transfer one SHA-256-verified
archive to vast.ai using SSH/SCP. This requires no additional storage service,
LFS setup or dataset pruning. Commands are in [the vast.ai runbook](vast-ai-s0.md).
The active transfer archive has **not** been created or uploaded in this task.

Git LFS is a viable alternative if a single clone-plus-LFS-fetch workflow is
preferred. It stores pointers in Git and the binary payload separately; it needs
Git LFS on both machines and sufficient account storage/download allowance.
Git LFS was not installed on this Mac during inspection. If approved, use a
normal additive commit with narrowly scoped tracking for the active dataset's
binary files and exclusions for work logs/OS metadata. Do not use `git lfs migrate`
or rewrite history. See [Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

No `.gitattributes`, dataset ignore exception, data upload, or dataset deletion
was made. Generated QC, pretrained weights, environments, credentials and training
outputs should remain outside committed source files.

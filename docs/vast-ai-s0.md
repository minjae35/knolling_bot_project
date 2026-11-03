# Clone, verify and prepare S0 on vast.ai

This runbook does not provision an instance or authorize a dataset upload.
Dataset storage/transfer still needs the user's approval. The preparation task
does not run training. Keep the legacy pilots and their verified backups intact.

## Readiness and scope

The committed S0 data adapter selects 32 Train samples from 25 scene families:
16 full, 8 partial, 8 no-op, with 2–4 objects each. Its configuration pins the
English manifest to:

```text
8000428a04522f50cf37567d85d442ac94d1f8a87121747d4243e33974320d06
```

At the start of this preparation, `93594b4` supplied the adapter/configuration
and its six tests. A model and losses were being developed concurrently in the
same checkout. **A complete, reviewed training CLI was not yet available at that
checkpoint.** Do not interpret environment setup or a successful data audit as
proof that S0 training is implemented. The final readiness notes below must be
checked against the exact code revision before paying for GPU runtime.

S0 success means the specification's 32-sample memorization gate, not
generalization: count accuracy 100%, source/target center and size P95 errors
at most 1 mm, orientation P95 at most 2 degrees. Report safety and instruction
failures separately. Maximum budget is 2,000 optimizer updates, FP32, fixed
learning rates, Detection + Target loss only. Do not replace the training CLI
with ad hoc dataset mutation or Teacher regeneration.

## 1. Pin code and install dependencies

Use a Linux x86_64 instance/container with Python **3.12**, `venv`, Git, SSH,
and a working NVIDIA driver visible through `nvidia-smi`. Ubuntu 24.04 is a
straightforward Python 3.12 base. Choose an SSH-capable template; record its
image identifier/digest and GPU model for reproducibility. Do not assume the
template's preinstalled Python or torch matches this project. Blender is not
needed to read this pilot or run S0.

The repository is private. Use an existing authorized SSH identity or a
repository-scoped, read-only GitHub credential supplied interactively. Do not
put a token in a clone URL, command argument, environment file, notebook or Git
configuration. A temporary read-only deploy key is another option; revoke it
after use. Credentials are not included in this repository.

```sh
mkdir -p /workspace
cd /workspace
git -c credential.helper= clone https://github.com/minjae35/knolling_bot_project.git
cd knolling_bot_project
# Replace with the full pushed commit selected for the run, not a moving branch.
git checkout --detach FULL_APPROVED_COMMIT_SHA
git status --short
bash scripts/setup_vast.sh
source .venv/bin/activate
```

`setup_vast.sh` refuses non-Linux/non-x86_64 machines, creates an isolated `.venv`,
installs torch **2.9.1+cu128** and torchvision **0.24.1+cu128** from the official
PyTorch index, then the pinned `requirements-neural-lock.txt` dependencies and
this package. It runs `pip check` and a tiny CUDA matrix operation, not training.
It records the resolved Linux dependencies, driver and code revision under
ignored `artifacts/environment/`. It never installs or changes a host driver.

Do **not** install `requirements-lock.txt` into this neural environment as well:
that is the historical dataset-only environment and has different NumPy/Pillow/
Shapely pins. Preserve it for the old regression environment.

The selected PyTorch pairing is documented in
[official installation instructions](https://pytorch.org/get-started/previous-versions/).
Pinned non-CUDA dependency versions were checked for available Linux wheels.
The complete CUDA installation must still be verified on the actual instance;
macOS tests do not establish NVIDIA compatibility or GPU memory requirements.
The checked-in neural freeze is not a hash-locked Linux/container image. Retain
the resolved Linux freeze and image digest with the run; exact numerical
reproduction across GPU/driver versions is not guaranteed.

If setup fails, stop. Check Python 3.12, architecture, free disk, network access,
and NVIDIA driver/device exposure. Do not fix a driver problem by silently
unpinning torch. Re-run the script only in the intended isolated environment.

## 2. Transfer the complete English pilot — after storage approval only

The following commands are instructions, not actions already performed.
Only the English pilot is needed on the GPU instance. Leave both external legacy
directories on the Mac for regression/provenance; keep the provenance copies
inside the English directory.

On the Mac, create a fresh transfer directory and archive without changing the
source dataset. No overwrite or deletion is needed:

```sh
cd /Users/minjae/Desktop/knolling_bot_project
TRANSFER_DIR="$(mktemp -d /Users/minjae/Desktop/knolling-transfer.XXXXXX)"
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' \
  -czf "$TRANSFER_DIR/pilot_v1_3_en.tar.gz" -C artifacts pilot_v1_3_en
cd "$TRANSFER_DIR"
shasum -a 256 pilot_v1_3_en.tar.gz > pilot_v1_3_en.tar.gz.sha256
```

Use the **exact SSH host, port and user** shown by vast.ai's Connect control.
Verify the SSH host identity before transferring. These variable values are
placeholders; they do not identify or create an instance:

```sh
VAST_SSH_HOST='HOST_FROM_CONNECT'
VAST_SSH_PORT='PORT_FROM_CONNECT'
VAST_SSH_USER='USER_FROM_CONNECT'
ssh -p "$VAST_SSH_PORT" "$VAST_SSH_USER@$VAST_SSH_HOST" 'mkdir -p /workspace/incoming'
scp -P "$VAST_SSH_PORT" pilot_v1_3_en.tar.gz pilot_v1_3_en.tar.gz.sha256 \
  "$VAST_SSH_USER@$VAST_SSH_HOST:/workspace/incoming/"
```

Vast supports standard SSH/SCP transfers; see its
[official FAQ](https://console.vast.ai/faq/). If its connection uses extra SSH
options, carry those options over to SCP as appropriate; use uppercase `-P` for
SCP's port. A checksum detects transfer corruption; use the checksum produced on
the trusted source Mac.

On the instance:

```sh
cd /workspace/incoming
sha256sum -c pilot_v1_3_en.tar.gz.sha256
cd /workspace/knolling_bot_project
mkdir -p artifacts
# Stop if data already exists. Never extract over a prior pilot.
test ! -e artifacts/pilot_v1_3_en && \
  tar -xzf /workspace/incoming/pilot_v1_3_en.tar.gz -C artifacts
```

Only extract the archive you created and hash-verified. Keep the source archive
and checksum until the full verification below passes. No training-only pruning
is needed for a 129 MiB pilot. GitHub LFS remains an alternative requiring a
separate approved storage change; this runbook does not install or enable it.

## 3. Verify data and the selected implementation

From the repository root with its neural environment active:

```sh
python scripts/verify_s0_data.py
python -m pytest -q tests/neural/test_data.py
python -m pip check
git status --short
```

The verification is read-only. It audits all 130 samples/100 families, image and
mask hashes/projections, object identities, splits, geometry, language, and the
embedded migration provenance. It then checks the pinned 32-sample selection,
source RGB shapes and source-target joins through the actual S0 adapter.
Expected output is `valid: true`, `samples: 130`, `scene_families: 100`,
`s0_samples: 32`, and the manifest SHA above. Do not edit that SHA to bypass a
mismatch; check the transferred version.

The legacy external directories and Blender are not needed for these checks.
Neural tests may otherwise skip when torch/data is absent, so a green pytest
summary alone is insufficient: the explicit verifier must pass and the six data
tests must run, not skip. Full old-pilot acceptance remains documented in README.

Model weights must use `configs/pretrained.json`: the text encoder's repository
revision is pinned, and the visual backbone uses `IMAGENET1K_V1`. Download weights
only through the reviewed model loader after that implementation is committed;
keep caches outside Git (for example `HF_HOME=/workspace/cache/huggingface` and
`TORCH_HOME=/workspace/cache/torch`). Neither weights nor tokens belong in Git.
Set these cache variables consistently for download and training.

## 4. S0 launch gate

The runnable S0 command is pending completion/verification of the concurrently
developed training entry point. This workflow intentionally does not invent a
module name or launch an unreviewed trainer. Before an actual run, the pinned
revision must include:

- Integrated model, source-only Hungarian matching and Detection + Target losses.
- A documented CLI/config for the selected 32 samples, CUDA device, seed, output
  directory, effective batch 16, and maximum 2,000 optimizer updates.
- Evaluation of all selected examples, including counterfactual instructions,
  plus checkpoints with optimizer/RNG/config/data revision and working resume.
- Passing model, loss, identity, gradient-flow and checkpoint/resume checks.

Until that gate is resolved, the environment/data preparation is usable but S0
is **not claimed to be launch-ready**. Run output must go under ignored
`artifacts/` or `runs/`, never into tracked source. Before stopping/deleting a
future instance, copy checkpoints, metrics, logs, resolved environment and the
code/data identifiers back to persistent storage and verify their checksums.

## Preparation validation on the Mac

The repository package was installed into the existing neural `.venv` without
replacing its pinned dependencies. `pip check` passed. The read-only English
verifier passed for 130 samples / 100 families and all 32 S0 selections; the six
data-adapter tests passed. Dataset tests with all three pilot paths enabled
passed **73 tests**, with **5 opt-in rendering/migration tests skipped**. No
Blender render, dataset regeneration or S0 training run was performed.

The setup script passed shell syntax checking and correctly refused macOS
before creating an environment or installing packages. Linux Python 3.12 wheel
availability and a cross-platform dependency dry run were checked; these do not
exercise Linux-only CUDA dependencies or a real NVIDIA device. The actual GPU
setup and S0 training result remain unverified until run on the selected host.

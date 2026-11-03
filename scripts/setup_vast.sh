#!/usr/bin/env bash
# Run explicitly on a Linux x86_64 NVIDIA instance. Installs; never trains.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This setup targets Linux x86_64; it will not modify this machine." >&2
  exit 2
fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
command -v nvidia-smi >/dev/null
nvidia-smi

python3.12 -m venv .venv
PYTHON=.venv/bin/python
# Use the explicit official CUDA index, not the container's preinstalled torch.
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1' 'torchvision==0.24.1'
"$PYTHON" -m pip install --index-url https://pypi.org/simple -r requirements-neural-lock.txt
"$PYTHON" -m pip install --no-deps --no-build-isolation -e .
"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import torch, torchvision
assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert torchvision.__version__ == '0.24.1+cu128', torchvision.__version__
assert torch.cuda.is_available(), 'NVIDIA GPU/driver is not available to PyTorch'
x = torch.ones(2, 2, device='cuda')
assert (x @ x).sum().item() == 8
torch.cuda.synchronize()
print('CUDA verified:', torch.cuda.get_device_name(0))
PY
mkdir -p artifacts/environment
"$PYTHON" -m pip freeze > artifacts/environment/requirements-linux-resolved.txt
nvidia-smi > artifacts/environment/nvidia-smi.txt
git rev-parse HEAD > artifacts/environment/code-revision.txt
echo 'Environment installed. Next: verify the transferred dataset; see docs/vast-ai-s0.md.'

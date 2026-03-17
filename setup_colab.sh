#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APT_PACKAGES=(
  python3-pip
  python3-dev
  git
  openssh-client
  ca-certificates
  curl
  bash
  libopenjp2-7-dev
  libopenjp2-tools
  openslide-tools
  libgeos-dev
  libgl1
  libglib2.0-0
  texlive-latex-extra
)

print_step() {
  printf '\n[%s] %s\n' "setup_colab" "$1"
}

print_step "Installing system packages"
apt-get update -qq
apt-get install -y -qq "${APT_PACKAGES[@]}"

if ! command -v uv >/dev/null 2>&1; then
  print_step "Installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.cargo/bin:$PATH"

print_step "Installing Python 3.12 via uv"
uv python install 3.12

print_step "Syncing project dependencies from pyproject.toml"
uv sync --python 3.12

if [ ! -f .env ] && [ -f .env_example ]; then
  print_step "Creating .env from .env_example"
  cp .env_example .env
fi

print_step "Verifying critical imports"
uv run --python 3.12 python - <<'PY'
from __future__ import annotations

import cv2
import openslide
import segmentation_models_pytorch as smp
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
print(f"openslide={openslide.__library_version__}")
print(f"opencv={cv2.__version__}")
print(f"smp={smp.__version__}")
PY

print_step "Setup complete"
printf '%s\n' "Next steps:"
printf '  1. Edit .env with your dataset and model paths.\n'
printf '  2. Mount Google Drive if your zip file or models live there.\n'
printf '  3. Run scripts with: uv run --python 3.12 python <script>.py\n'
printf '  4. Example: uv run --python 3.12 python 1_artifact_detection.py\n'

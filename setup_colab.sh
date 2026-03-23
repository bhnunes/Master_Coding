#!/bin/bash

set -euo pipefail

# Check for argument
MODE="${1:-default}"
PYTHON_VERSION="3.12"
EXTRA_ARGS=""

if [ "$MODE" == "grandqc" ]; then
    echo "!!! Configuring for GrandQC (Legacy Python 3.10) !!!"
    PYTHON_VERSION="3.10"
    # Sync ONLY the grandqc group and exclude default project dependencies to avoid conflicts
    EXTRA_ARGS="--group grandqc --no-default-groups"
else
    echo "--- Configuring for Master Project (Python 3.12) ---"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APT_PACKAGES=(
  python3-pip python3-dev git openssh-client ca-certificates
  curl bash libopenjp2-7-dev libopenjp2-tools openslide-tools
  libgeos-dev libgl1 libglib2.0-0 texlive-latex-extra
)

print_step() {
  printf '\n[%s] %s\n' "setup_env" "$1"
}

print_step "Installing system packages"
apt-get update -qq
apt-get install -y -qq "${APT_PACKAGES[@]}"

if ! command -v uv >/dev/null 2>&1; then
  print_step "Installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.cargo/bin:$PATH"

print_step "Installing Python $PYTHON_VERSION via uv"
uv python install "$PYTHON_VERSION"

print_step "Syncing dependencies (Python $PYTHON_VERSION)"
# This uses the specific group if grandqc is passed
uv sync --python "$PYTHON_VERSION" $EXTRA_ARGS

print_step "Verifying critical imports"
uv run --python "$PYTHON_VERSION" python - <<'PY'
import sys
import torch
try:
    import openslide
    print(f"Python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}")
    print(f"openslide={openslide.__library_version__}")
except ImportError as e:
    print(f"Import Error: {e}")
PY

print_step "Setup complete ($MODE)"
printf 'Run scripts with: uv run --python %s python <script>.py\n' "$PYTHON_VERSION"
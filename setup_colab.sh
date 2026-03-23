#!/bin/bash

set -euo pipefail

# Check for argument
MODE="${1:-default}"
PYTHON_VERSION="3.12"
EXTRA_ARGS=""

if [ "$MODE" == "grandqc" ]; then
    echo "!!! Configuring for GrandQC (Legacy Python 3.10) !!!"
    PYTHON_VERSION="3.10"
else
    echo "--- Configuring for Master Project (Python 3.12) ---"
    PYTHON_VERSION="3.12"
  
fi

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

print_step "Installing Python $PYTHON_VERSION via uv"
uv python install "$PYTHON_VERSION"

print_step "Installing dependencies (Python $PYTHON_VERSION)"

if [ "$MODE" == "grandqc" ]; then
    uv venv --python "$PYTHON_VERSION"
    # Force uv to install into this specific venv, bypassing the pyproject.toml constraints
    uv pip install --python .venv -r requirements-grandqc.txt
else
    uv sync --python "$PYTHON_VERSION"
fi

print_step "Verifying critical imports"

if [ "$MODE" == "grandqc" ]; then
    EXEC_CMD=".venv/bin/python"
else
    EXEC_CMD="uv run --python $PYTHON_VERSION python"
fi



print_step "Setup complete"
printf '%s\n' "Next steps:"
printf '  1. Edit .env with your dataset and model paths.\n'
printf '  2. Mount Google Drive if your zip file or models live there.\n'
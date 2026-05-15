#!/usr/bin/env bash

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_DEV="${SETUP_MACOS_INSTALL_DEV:-1}"

BREW_PACKAGES=(
  ca-certificates
  cmake
  geos
  git
  libomp
  openjpeg
  pkg-config
)

print_step() {
  printf '\n[%s] %s\n' "setup_macos" "$1"
}

fail() {
  printf 'setup_macos: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_macos() {
  local system_name
  system_name="$(uname -s)"
  if [ "$system_name" != "Darwin" ]; then
    fail "This script is for macOS. Detected: $system_name"
  fi
}

ensure_apple_silicon() {
  local machine
  machine="$(uname -m)"
  if [ "$machine" != "arm64" ]; then
    fail "This setup currently supports Apple Silicon Macs only. Detected: $machine"
  fi
}

activate_homebrew() {
  if command_exists brew; then
    return 0
  fi

  local candidate
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [ -x "$candidate" ]; then
      eval "$("$candidate" shellenv)"
      return 0
    fi
  done

  fail "Homebrew is required. Install it from https://brew.sh/ and rerun this script."
}

ensure_command_line_tools() {
  if xcode-select -p >/dev/null 2>&1; then
    return 0
  fi

  fail "Xcode Command Line Tools are required. Install them with: xcode-select --install"
}

install_brew_packages() {
  local missing=()
  local package

  for package in "${BREW_PACKAGES[@]}"; do
    if ! brew list --versions "$package" >/dev/null 2>&1; then
      missing+=("$package")
    fi
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    print_step "Homebrew packages already installed"
    return 0
  fi

  print_step "Installing Homebrew packages: ${missing[*]}"
  brew install "${missing[@]}"
}

ensure_uv() {
  if command_exists uv; then
    return 0
  fi

  print_step "Installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  if ! command_exists uv; then
    fail "uv installation finished, but uv is still not on PATH."
  fi
}

configure_native_library_paths() {
  local brew_prefix
  brew_prefix="$(brew --prefix)"

  export PATH="$brew_prefix/bin:$PATH"
  export PKG_CONFIG_PATH="$brew_prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CPPFLAGS="-I$brew_prefix/opt/libomp/include ${CPPFLAGS:-}"
  export LDFLAGS="-L$brew_prefix/opt/libomp/lib ${LDFLAGS:-}"
}

sync_python_dependencies() {
  local sync_args=(sync --python "$PYTHON_VERSION")

  if [ "$INSTALL_DEV" = "1" ] || [ "$INSTALL_DEV" = "true" ] || [ "$INSTALL_DEV" = "TRUE" ]; then
    sync_args+=(--group dev)
  fi

  print_step "Installing Python $PYTHON_VERSION via uv"
  uv python install "$PYTHON_VERSION"

  print_step "Syncing Python dependencies"
  if ! uv "${sync_args[@]}"; then
    cat >&2 <<'EOF'

Dependency sync failed.
On macOS, check that the project dependency lock has macOS-compatible wheels.
CUDA-only packages may need platform markers in pyproject.toml.
EOF
    exit 1
  fi
}

verify_imports() {
  print_step "Verifying Python imports"

  uv run --python "$PYTHON_VERSION" python - <<'PY'
import cv2
import openslide
import torch

print("openslide", getattr(openslide, "__library_version__", "loaded"))
print("cv2", cv2.__version__)
print("torch", torch.__version__)

mps_backend = getattr(torch.backends, "mps", None)
mps_available = bool(mps_backend and torch.backends.mps.is_available())
print("torch_mps_available", mps_available)
PY
}

main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir"

  ensure_macos
  ensure_apple_silicon
  ensure_command_line_tools
  activate_homebrew
  configure_native_library_paths
  install_brew_packages
  ensure_uv
  sync_python_dependencies
  verify_imports

  print_step "Setup complete"
  printf '%s\n' "Next steps:"
  printf '  1. Edit .env with your dataset, output, and model paths.\n'
  printf '  2. Run a phase with:\n'
  printf '     uv run --python %s python 1_artifact_detection.py\n' "$PYTHON_VERSION"
}

main "$@"

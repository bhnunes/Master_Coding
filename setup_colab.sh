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

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

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

is_colab_runtime() {
  [[ -n "${COLAB_RELEASE_TAG:-}" || -d /content || -d /usr/local/colab ]]
}

disable_nonessential_colab_apt_sources() {
  local mode="${SETUP_COLAB_DISABLE_THIRD_PARTY_APT:-auto}"

  case "$mode" in
    1 | true | TRUE | yes | YES)
      ;;
    0 | false | FALSE | no | NO)
      return 0
      ;;
    auto)
      if ! is_colab_runtime; then
        return 0
      fi
      ;;
    *)
      printf 'Invalid SETUP_COLAB_DISABLE_THIRD_PARTY_APT value: %s\n' "$mode" >&2
      printf 'Use auto, 1, true, yes, 0, false, or no.\n' >&2
      exit 2
      ;;
  esac

  local pattern='ppa\.launchpad(content)?\.net|r2u\.stat\.illinois\.edu/ubuntu'
  local changed=0
  local file

  for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
    [ -f "$file" ] || continue
    if grep -Eq "$pattern" "$file"; then
      cp -n "$file" "${file}.setup_colab.bak" 2>/dev/null || true
      sed -i -E "\#${pattern}#s#^[[:space:]]*([^#[:space:]])#\# disabled by setup_colab: \1#" "$file"
      changed=1
    fi
  done

  for file in /etc/apt/sources.list.d/*.sources; do
    [ -f "$file" ] || continue
    if grep -Eq "$pattern" "$file"; then
      cp -n "$file" "${file}.setup_colab.bak" 2>/dev/null || true
      mv "$file" "${file}.setup_colab.disabled"
      changed=1
    fi
  done

  rm -f /var/lib/apt/lists/*ppa.launchpad.net* \
    /var/lib/apt/lists/*ppa.launchpadcontent.net* \
    /var/lib/apt/lists/*r2u.stat.illinois.edu* \
    /var/lib/apt/lists/partial/*ppa.launchpad.net* \
    /var/lib/apt/lists/partial/*ppa.launchpadcontent.net* \
    /var/lib/apt/lists/partial/*r2u.stat.illinois.edu*

  if [ "$changed" -eq 1 ]; then
    print_step "Disabled nonessential third-party Colab apt sources"
  fi
}

APT_GET=(
  apt-get
  -o Acquire::Retries=3
  -o Acquire::http::Timeout=30
  -o Acquire::https::Timeout=30
)

print_step "Installing system packages"
disable_nonessential_colab_apt_sources
"${APT_GET[@]}" update -qq
"${APT_GET[@]}" install -y -qq "${APT_PACKAGES[@]}"

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

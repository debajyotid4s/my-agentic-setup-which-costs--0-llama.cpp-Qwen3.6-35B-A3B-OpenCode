#!/usr/bin/env bash
# 01-install-deps.sh
# System packages, Hugging Face CLI, OpenCode, plus sanity checks for the NVIDIA driver and CUDA.
# Targets Debian/Ubuntu. The NVIDIA driver and CUDA toolkit 12.x must already be installed
# (see SETUP.md, step 1). This script never installs CUDA for you.
#
# Optional environment variables:
#   SKIP_OPENCODE=1   do not install OpenCode
set -Eeuo pipefail

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v apt-get >/dev/null 2>&1 || die "apt-get not found. This script targets Debian/Ubuntu; install the equivalent packages manually."

say "Installing build dependencies"
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl wget pkg-config \
  libcurl4-openssl-dev python3 python3-pip pipx pciutils

say "Checking the NVIDIA driver and GPU"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install the NVIDIA driver first (SETUP.md, step 1)."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "Checking the CUDA toolkit (must be 12.x; CUDA 13 dropped Volta / V100)"
NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
fi
[[ -n "$NVCC" ]] || die "nvcc not found. Install CUDA toolkit 12.x (not 13) and add its bin directory to PATH."
CUDA_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]\+\)\..*/\1/p' | head -n1)"
echo "nvcc: $NVCC (CUDA major version: ${CUDA_MAJOR:-unknown})"
if [[ "${CUDA_MAJOR:-0}" != "12" ]]; then
  die "CUDA toolkit major version is '${CUDA_MAJOR:-unknown}', but 12.x is required for the V100."
fi

say "Installing the Hugging Face CLI (via pipx)"
pipx ensurepath >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
if command -v hf >/dev/null 2>&1; then
  echo "hf already installed: $(command -v hf)"
else
  pipx install "huggingface_hub[cli]" || pipx install huggingface_hub
fi
command -v hf >/dev/null 2>&1 || warn "'hf' not on PATH yet. Open a new shell (or run: export PATH=\"\$HOME/.local/bin:\$PATH\")."

if [[ "${SKIP_OPENCODE:-0}" != "1" ]]; then
  say "Installing OpenCode"
  if command -v opencode >/dev/null 2>&1; then
    echo "opencode already installed: $(opencode --version 2>/dev/null || echo 'version unknown')"
  else
    # Official install script from opencode.ai/docs (installs OpenCode 1).
    curl -fsSL https://opencode.ai/install | bash
  fi
  warn "This bundle was written against OpenCode 1 config. Record your version with: opencode --version"
fi

say "Done"
echo "Next: scripts/02-build-llama-cpp.sh"

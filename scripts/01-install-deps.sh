#!/usr/bin/env bash
# 01-install-deps.sh
# System packages, Hugging Face CLI, OpenCode, plus sanity checks for the NVIDIA driver and CUDA.
# Targets Debian/Ubuntu. The NVIDIA driver and a compatible CUDA toolkit must already be installed.
# CUDA 12.x is required for Volta (V100); CUDA 12.x or 13.x for Turing and newer.
# This script never installs CUDA for you.
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
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install the NVIDIA driver first."
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)"
GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
echo "GPU: $GPU_NAME | Driver: $GPU_DRIVER"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "Checking the CUDA toolkit"
NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
fi
[[ -n "$NVCC" ]] || die "nvcc not found. Install a CUDA toolkit and add its bin directory to PATH."
CUDA_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]\+\)\..*/\1/p' | head -n1)"
echo "nvcc: $NVCC (CUDA major version: ${CUDA_MAJOR:-unknown})"

# CUDA compatibility: some CUDA 13 distributions dropped Volta (sm_70) support.
# If your GPU is Volta (V100, TX2, Quadro RTX 4000/5000/6000), use CUDA 12.x.
# For Turing (sm_75) and newer, CUDA 12.x or 13.x both work.
CUDA_OK=1
case "$GPU_NAME" in
  *V100*|*Quadro*RTX*4000*|*Quadro*RTX*5000*|*Quadro*RTX*6000*|*Tesla*V100*)
    if [[ "${CUDA_MAJOR:-0}" != "12" ]]; then
      CUDA_OK=0
      echo "WARN: $GPU_NAME is Volta (sm_70). CUDA 13 may not support it." >&2
      echo "      Install CUDA 12.x toolkit, or rebuild llama.cpp with CMAKE_CUDA_ARCHITECTURES=70 and a CUDA 13 build that includes sm_70." >&2
    fi
    ;;
esac

if [[ "$CUDA_OK" -eq 1 ]]; then
  case "${CUDA_MAJOR:-0}" in
    12|13) echo "CUDA $CUDA_MAJOR: OK for this GPU." ;;
    *)
      CUDA_OK=0
      echo "WARN: CUDA $CUDA_MAJOR may not be supported. Use CUDA 12.x or 13.x." >&2
      ;;
  esac
fi

[[ "$CUDA_OK" -eq 1 ]] || warn "CUDA/GPU compatibility check failed. See above warnings."

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

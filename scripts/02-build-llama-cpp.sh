#!/usr/bin/env bash
# 02-build-llama-cpp.sh
# Builds llama.cpp from source for the Tesla V100 (Volta, compute capability 7.0).
# Prebuilt llama.cpp binaries do not include Volta kernels, so a source build is required.
#
# Optional environment variables:
#   LLAMA_DIR   where to clone/build   (default: $HOME/llama.cpp)
#   LLAMA_REF   tag or commit to pin   (default: current default branch)
set -Eeuo pipefail

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_REF="${LLAMA_REF:-}"

NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
fi
[[ -n "$NVCC" ]] || die "nvcc not found. Install CUDA toolkit 12.x first (SETUP.md, step 1)."
CUDA_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]\+\)\..*/\1/p' | head -n1)"
[[ "${CUDA_MAJOR:-0}" == "12" ]] || die "CUDA major version is '${CUDA_MAJOR:-unknown}'. The V100 needs 12.x (CUDA 13 dropped Volta)."
export CUDACXX="$NVCC"

say "Fetching llama.cpp into $LLAMA_DIR"
if [[ -d "$LLAMA_DIR/.git" ]]; then
  git -C "$LLAMA_DIR" fetch --tags origin
  git -C "$LLAMA_DIR" pull --ff-only || warn "Could not fast-forward (detached HEAD or local changes?). Building what is checked out."
else
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if [[ -n "$LLAMA_REF" ]]; then
  git -C "$LLAMA_DIR" checkout "$LLAMA_REF"
fi
echo "Building commit: $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"

say "Configuring (CUDA on, architecture 70 = V100)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release

say "Compiling (this can take several minutes)"
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-server llama-cli llama-bench

SERVER="$LLAMA_DIR/build/bin/llama-server"
[[ -x "$SERVER" ]] || die "Build finished but $SERVER is missing."

say "Build OK. Flags available in this build:"
HELP="$("$SERVER" --help 2>&1 || true)"
for f in --n-cpu-moe --fit --flash-attn --jinja --ctx-checkpoints \
         --checkpoint-min-step --checkpoint-every-n-tokens \
         --reasoning-budget --chat-template-kwargs; do
  if grep -q -- "$f" <<<"$HELP"; then
    printf '  [x] %s\n' "$f"
  else
    printf '  [ ] %s   (not in this build)\n' "$f"
  fi
done
echo
echo "Next: scripts/03-download-model.sh"

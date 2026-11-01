#!/usr/bin/env bash
# 03-download-model.sh
# Downloads Qwen3.6-35B-A3B (Unsloth dynamic GGUF, UD-IQ4_XS, about 17.7 GB).
# Always download a fresh copy: Unsloth updated the chat template (developer-role support),
# which OpenCode needs.
#
# Optional environment variables:
#   MODEL_REPO   Hugging Face repo   (default: unsloth/Qwen3.6-35B-A3B-GGUF)
#   MODEL_QUANT  quant tag           (default: UD-IQ4_XS)
#   MODEL_DIR    destination folder  (default: $HOME/models/qwen3.6)
set -Eeuo pipefail

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
command -v hf >/dev/null 2>&1 || die "'hf' not found. Run scripts/01-install-deps.sh first (or open a new shell)."

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-IQ4_XS}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6}"

mkdir -p "$MODEL_DIR"
AVAIL_GB="$(df -BG --output=avail "$MODEL_DIR" | tail -n1 | tr -dc '0-9')"
if [[ "${AVAIL_GB:-0}" -lt 25 ]]; then
  die "Only ${AVAIL_GB:-0} GB free in $MODEL_DIR; need roughly 25 GB. Free up space or set MODEL_DIR."
fi

say "Downloading ${MODEL_REPO} (*${MODEL_QUANT}*) to ${MODEL_DIR}"
hf download "$MODEL_REPO" --include "*${MODEL_QUANT}*" --local-dir "$MODEL_DIR"

say "Downloaded files"
shopt -s nullglob
FILES=("$MODEL_DIR"/*"${MODEL_QUANT}"*.gguf)
[[ ${#FILES[@]} -gt 0 ]] || die "No *${MODEL_QUANT}*.gguf found in $MODEL_DIR. Check the repo's file list."
ls -lh "${FILES[@]}"

echo
echo "llm-up.sh expects: $MODEL_DIR/Qwen3.6-35B-A3B-${MODEL_QUANT}.gguf"
echo "If the filename differs, set LLM_MODEL=/full/path/to/file.gguf when starting the server."
echo "Next: scripts/04-install-opencode-config.sh"

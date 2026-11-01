#!/usr/bin/env bash
# llm-up.sh
# Starts llama-server for Qwen3.6-35B-A3B (UD-IQ4_XS) on a Tesla V100 16 GB.
#
# Defaults are the values this setup ran with. Override any of them per run, for example:
#   LLM_NCMOE=16 ./llm-up.sh
#   LLM_CTX=131072 LLM_THREADS=3 ./llm-up.sh
#
# Tunables (environment variables):
#   LLM_MODEL            path to the .gguf        (default: ~/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf)
#   LLM_SERVER           path to llama-server     (default: ~/llama.cpp/build/bin/llama-server)
#   LLM_HOST / LLM_PORT  bind address             (default: 127.0.0.1 / 8080)
#   LLM_CTX              context window, tokens   (default: 262144)
#   LLM_NCMOE            layers whose experts stay in system RAM (default: 18). LOWER = faster, uses more VRAM.
#   LLM_THREADS          decode threads           (default: physical CPU cores)
#   LLM_THREADS_BATCH    prefill threads          (default: all logical CPUs)
#   LLM_BATCH / LLM_UBATCH   batch sizes          (default: 2048 / 512)
#   LLM_KV_K / LLM_KV_V  KV-cache types           (default: q8_0 / q8_0)
# Optional extras (off by default):
#   LLM_CHECKPOINTS=32          enable prompt-cache checkpoints (flag names are auto-detected per build)
#   LLM_CHECKPOINT_STEP=2048    checkpoint spacing in tokens
#   LLM_PRESERVE_THINKING=1     pass preserve_thinking=true to the chat template (A/B test it)
#   LLM_REASONING_BUDGET=4096   cap thinking tokens (trade-off: lower quality on hard problems)
set -Eeuo pipefail

MODEL="${LLM_MODEL:-$HOME/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
SERVER="${LLM_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8080}"

CTX="${LLM_CTX:-262144}"
NCMOE="${LLM_NCMOE:-18}"
BATCH="${LLM_BATCH:-2048}"
UBATCH="${LLM_UBATCH:-512}"
KV_K="${LLM_KV_K:-q8_0}"
KV_V="${LLM_KV_V:-q8_0}"

# Threads: physical cores for decode (-t), all logical CPUs for batch/prefill (-tb).
LOGICAL="$(nproc --all)"
PHYS="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l || true)"
if [[ -z "${PHYS:-}" || "$PHYS" -lt 1 ]]; then
  PHYS=$(( LOGICAL > 1 ? LOGICAL / 2 : 1 ))
fi
THREADS="${LLM_THREADS:-$PHYS}"
THREADS_BATCH="${LLM_THREADS_BATCH:-$LOGICAL}"

# Verify required files
if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server executable not found: $SERVER" >&2
  echo "       Run scripts/02-build-llama-cpp.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Model file not found: $MODEL" >&2
  echo "       Run scripts/03-download-model.sh, or set LLM_MODEL=/path/to/file.gguf" >&2
  exit 1
fi
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q "[:.]${PORT}[[:space:]]"; then
  echo "ERROR: port ${PORT} is already in use (is another llama-server running?)." >&2
  exit 1
fi

# Optional extras
EXTRA=()
if [[ -n "${LLM_CHECKPOINTS:-}" ]]; then
  HELP="$("$SERVER" --help 2>&1 || true)"
  if grep -q -- '--ctx-checkpoints' <<<"$HELP"; then
    EXTRA+=(--ctx-checkpoints "$LLM_CHECKPOINTS")
  fi
  if grep -q -- '--checkpoint-min-step' <<<"$HELP"; then
    EXTRA+=(--checkpoint-min-step "${LLM_CHECKPOINT_STEP:-2048}")
  elif grep -q -- '--checkpoint-every-n-tokens' <<<"$HELP"; then
    EXTRA+=(--checkpoint-every-n-tokens "${LLM_CHECKPOINT_STEP:-1024}")
  fi
fi
if [[ "${LLM_PRESERVE_THINKING:-0}" == "1" ]]; then
  EXTRA+=(--chat-template-kwargs '{"preserve_thinking": true}')
fi
if [[ -n "${LLM_REASONING_BUDGET:-}" ]]; then
  EXTRA+=(--reasoning-budget "$LLM_REASONING_BUDGET")
fi

echo "Starting Qwen3.6-35B-A3B..."
echo "Model:    $MODEL"
echo "Context:  $CTX tokens | KV cache: K=$KV_K V=$KV_V"
echo "Offload:  --n-cpu-moe $NCMOE (experts of this many layers stay in system RAM)"
echo "Threads:  decode=$THREADS batch=$THREADS_BATCH (physical cores=$PHYS, logical=$LOGICAL)"
echo "Batch:    $BATCH | Micro-batch: $UBATCH"
echo "Extras:   ${EXTRA[*]:-none}"
echo "Server:   http://$HOST:$PORT"

exec "$SERVER" \
  -m "$MODEL" \
  --alias qwen3.6-35b-a3b \
  -ngl 99 \
  --n-cpu-moe "$NCMOE" \
  -fit off \
  -c "$CTX" \
  -np 1 \
  -fa on \
  -ctk "$KV_K" \
  -ctv "$KV_V" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  --jinja \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  -t "$THREADS" \
  -tb "$THREADS_BATCH" \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  --host "$HOST" \
  --port "$PORT"

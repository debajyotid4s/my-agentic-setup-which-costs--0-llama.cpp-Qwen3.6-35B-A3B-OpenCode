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
#   LLM_CHECKPOINTS             enable prompt-cache checkpoints (default: 16, auto-detected flag names)
#   LLM_CHECKPOINT_STEP=2048    checkpoint spacing in tokens
#   LLM_REASONING_BUDGET=4096   cap thinking tokens (trade-off: lower quality on hard problems)
# Warm start / KV cache persistence:
#   LLM_PROMPT_CACHE=1          save KV cache to disk, restore on restart (saves re-prefill after idle)
#   LLM_MLOCK=1                 lock model/cache in RAM (prevents OS page eviction, uses more RAM)
# Memory safety (prevents OOM kills / forced reboots):
#   LLM_MAX_MEM=12G             cap process memory (e.g. 8G, 12G, 14G). Uses systemd-run --scope.
#   LLM_SWAP_CHECK=1            warn if no swap is configured (recommended: at least 8 GB)
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

# Memory cap — prevents OOM reboots. Override with LLM_MAX_MEM=14G for more headroom.
LLM_MAX_MEM="${LLM_MAX_MEM-12G}"

# Threads: physical cores for decode (-t), all logical CPUs for batch/prefill (-tb).
LOGICAL="$(nproc --all)"
PHYS="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l || true)"
if [[ -z "${PHYS:-}" || "$PHYS" -lt 1 ]]; then
  PHYS=$(( LOGICAL > 1 ? LOGICAL / 2 : 1 ))
fi
THREADS="${LLM_THREADS:-$PHYS}"
THREADS_BATCH="${LLM_THREADS_BATCH:-$LOGICAL}"

# ── Memory safety checks ─────────────────────────────────────────────────────
# Prevent OOM-induced reboots by capping process memory and warning about missing swap.

# Port check must happen BEFORE systemd-run path (which skips the normal path's port check)
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q "[:.]${PORT}[[:space:]]"; then
  echo "ERROR: port ${PORT} is already in use (is another llama-server running?)." >&2
  exit 1
fi

if [[ "${LLM_SWAP_CHECK:-0}" == "1" ]]; then
  SWAP_TOTAL=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  if [[ "${SWAP_TOTAL:-0}" -lt 8000000 ]]; then
    echo "WARN: swap is ${SWAP_TOTAL:-0} kB (0 if none). Recommend at least 8 GB."
    echo "     Without swap, OOM kills can force a reboot. Create one:"
    echo "       fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
  fi
fi

# ── Optional extras (built BEFORE launcher selection so flags reach the server) ──
EXTRA=()
# disable idle timeout — prevents KV cache eviction after inactivity
EXTRA+=(--timeout 0)
# preserve_thinking improves KV cache reuse across turns (Qwen recommendation)
EXTRA+=(--chat-template-kwargs '{"preserve_thinking": true}')
# prompt-cache persistence — saves/restores KV cache to disk to avoid full re-prefill after idle
if [[ -n "${LLM_PROMPT_CACHE:-}" ]]; then
  # Validate that this llama-server build supports --prompt-cache (some builds only have it in llama-cli)
  HELP="$("$SERVER" --help 2>&1 || true)"
  if ! grep -q -- '--prompt-cache' <<<"$HELP"; then
    echo "WARN: this llama-server build does not support --prompt-cache (may be in llama-cli only)." >&2
    echo "      Set LLM_PROMPT_CACHE= to disable." >&2
  else
    CACHE_DIR="$HOME/.llama-cache"
    mkdir -p "$CACHE_DIR"
    CACHE_FILE="$CACHE_DIR/qwen3.6-35b-a3b-kvcache.bin"
    EXTRA+=(--prompt-cache "$CACHE_FILE" --prompt-cache-all)
    echo "Prompt cache: $CACHE_FILE (saves KV cache between restarts)"
  fi
fi
# mlock — keeps model weights and KV cache in RAM, prevents OS from swapping them out
# WARNING: with MemoryMax cap, mlock can exceed the cap and trigger OOM. Do NOT enable with LLM_MAX_MEM set.
if [[ -n "${LLM_MLOCK:-}" ]]; then
  if [[ -n "${LLM_MAX_MEM:-}" ]]; then
    echo "WARN: LLM_MLOCK=1 with LLM_MAX_MEM=${LLM_MAX_MEM} is dangerous — model (~17.7 GB) exceeds the cap, will OOM." >&2
    echo "      Disable LLM_MAX_MEM (set LLM_MAX_MEM=) before using LLM_MLOCK=1." >&2
  fi
  EXTRA+=(--mlock)
  echo "mlock: enabled (keeps cache in RAM)"
fi
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
if [[ -n "${LLM_REASONING_BUDGET:-}" ]]; then
  EXTRA+=(--reasoning-budget "$LLM_REASONING_BUDGET")
fi

# ── Verify required files ─────────────────────────────────────────────────────
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

# ── Memory cap: build optional systemd-run prefix ─────────────────────────────
LAUNCH=()
if [[ -n "${LLM_MAX_MEM:-}" ]]; then
  mem_str="$LLM_MAX_MEM"
  mem_val=""
  mem_unit="G"
  if [[ "$mem_str" =~ ^([0-9]+)[Gg]$ ]]; then
    mem_val="${BASH_REMATCH[1]}"
    mem_bytes=$(( mem_val * 1073741824 ))
  elif [[ "$mem_str" =~ ^([0-9]+)[Mm]$ ]]; then
    mem_val="${BASH_REMATCH[1]}"
    mem_bytes=$(( mem_val * 1048576 ))
    mem_unit="M"
  else
    echo "WARN: unrecognized LLM_MAX_MEM='$mem_str' (use e.g. 12G); starting without a cap" >&2
    mem_bytes=""
  fi
  if [[ -n "$mem_bytes" ]]; then
    command -v systemd-run >/dev/null 2>&1 || { echo "ERROR: systemd-run not found; set LLM_MAX_MEM= to start without a cap." >&2; exit 1; }
    echo "Memory cap: ${mem_val}${mem_unit} (MemoryMax), MemoryHigh at 90%"
    LAUNCH=(systemd-run --scope --unit=llama-server -p "MemoryMax=${mem_bytes}" -p "MemoryHigh=$(( mem_bytes * 90 / 100 ))")
  fi
fi

echo "Starting Qwen3.6-35B-A3B..."
echo "Model:    $MODEL"
echo "Context:  $CTX tokens | KV cache: K=$KV_K V=$KV_V"
echo "Offload:  --n-cpu-moe $NCMOE (experts of this many layers stay in system RAM)"
echo "Threads:  decode=$THREADS batch=$THREADS_BATCH (physical cores=$PHYS, logical=$LOGICAL)"
echo "Batch:    $BATCH | Micro-batch: $UBATCH"
echo "Extras:   ${EXTRA[*]:-none}"
echo "Server:   http://$HOST:$PORT"

exec "${LAUNCH[@]+"${LAUNCH[@]}"}" "$SERVER" \
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

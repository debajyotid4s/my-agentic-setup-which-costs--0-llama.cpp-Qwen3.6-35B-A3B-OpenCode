#!/usr/bin/env bash
# llm-up.sh
# Start llama-server for Qwen3.6-35B-A3B (UD-IQ4_XS) on the user's V100 setup.
#
# The script checks the installed llama-server's own --help output before adding
# optional flags. Required performance/runtime flags fail early with a useful
# message if the binary is too old or was built without that option.
#
# Examples:
#   LLM_NCMOE=16 ./llm-up.sh
#   LLM_CTX=131072 LLM_THREADS=3 ./llm-up.sh
#   LLM_MAX_MEM= LLM_MLOCK=1 ./llm-up.sh   # disable cgroup cap before enabling mlock
#   LLM_REASONING_BUDGET=0 ./llm-up.sh     # unlimited reasoning (omit the budget flag)
#
# Tunables (environment variables):
#   LLM_MODEL                 GGUF model path
#   LLM_SERVER                llama-server executable
#   LLM_HOST / LLM_PORT       bind address / port
#   LLM_CTX                   context window (default: 262144)
#   LLM_NCMOE                 first N MoE layers kept in system RAM (default: 18)
#   LLM_THREADS               generation threads (default: physical CPU cores)
#   LLM_THREADS_BATCH         prompt/batch threads (default: all logical CPUs)
#   LLM_BATCH / LLM_UBATCH    logical / physical batch sizes (2048 / 512)
#   LLM_KV_K / LLM_KV_V       KV cache types (q8_0 / q8_0)
#   LLM_REASONING_BUDGET      reasoning token cap (4096); 0 omits the cap
#   LLM_REASONING_BUDGET_MESSAGE  message inserted when the cap is reached
#   LLM_PRESENCE_PENALTY      presence penalty (default: 0.5)
#   LLM_CHECKPOINTS           optional context checkpoints count
#   LLM_CHECKPOINT_STEP       optional checkpoint spacing (default: 2048)
#   LLM_CACHE_RAM             server prompt-cache RAM limit in MiB (unset = server default)
#   LLM_CACHE_PROMPT          1/0 to explicitly enable/disable prompt caching, if supported
#   LLM_CACHE_IDLE_SLOTS      set to 1 to retain idle slots in prompt cache, if supported
#   LLM_PROMPT_CACHE          legacy variable; disk prompt-cache files are not supported by llama-server
#   LLM_MLOCK                 set to 1 to request mlock (do not combine with a MemoryMax cap)
#   LLM_MAX_MEM               cgroup memory cap (default: 12G); set empty to disable
#   LLM_SWAP_CHECK            set to 1 to warn when swap is below 8 GB
#   LLM_LOG_FILE              log path (default: ~/llm-server.log); set empty to disable tee logging
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
THREADS="${LLM_THREADS:-}"
THREADS_BATCH="${LLM_THREADS_BATCH:-}"

REASONING_BUDGET="${LLM_REASONING_BUDGET:-4096}"
REASONING_MESSAGE="${LLM_REASONING_BUDGET_MESSAGE:-Reasoning budget reached. Stop analysing; run one concrete command or test for your leading hypothesis, then continue.}"
PRESENCE_PENALTY="${LLM_PRESENCE_PENALTY:-0.5}"

# Empty explicitly disables the cap; unset uses the 12G default.
LLM_MAX_MEM="${LLM_MAX_MEM-12G}"
LLM_LOG_FILE="${LLM_LOG_FILE-$HOME/llm-server.log}"

# ── Basic validation ─────────────────────────────────────────────────────────
if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server executable not found or not executable: $SERVER" >&2
  echo "       Build it with: cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_BUILD_SERVER=ON" >&2
  echo "                     cmake --build ~/llama.cpp/build --config Release --parallel" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model file not found: $MODEL" >&2
  exit 1
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "ERROR: LLM_PORT must be an integer from 1 to 65535 (got '$PORT')." >&2
  exit 1
fi

# Query the exact binary that this script will launch. Do not assume that the
# source checkout and the installed executable are at the same revision.
SERVER_VERSION="$("$SERVER" --version 2>&1 || true)"
SERVER_HELP="$("$SERVER" --help 2>&1 || true)"
if [[ -z "$SERVER_HELP" ]]; then
  echo "ERROR: could not read help output from $SERVER" >&2
  exit 1
fi

has_option() {
  local option="$1"
  # Match a complete option token, not a longer option with the same prefix.
  grep -Eq -- "(^|[[:space:],])${option}([[:space:],=]|$)" <<<"$SERVER_HELP"
}

require_option() {
  local option="$1"
  if ! has_option "$option"; then
    echo "ERROR: this llama-server binary does not advertise required option '$option'." >&2
    echo "       Version: ${SERVER_VERSION:-unknown}" >&2
    echo "       Binary:  $SERVER" >&2
    echo "       Check '$SERVER --help' or rebuild/update llama.cpp." >&2
    exit 1
  fi
}

# These are the options the launch configuration actually depends on.
REQUIRED_OPTIONS=(
  --model --alias --n-gpu-layers --n-cpu-moe --fit --ctx-size --parallel
  --flash-attn --cache-type-k --cache-type-v --batch-size --ubatch-size
  --jinja --temp --top-p --top-k --min-p --presence-penalty
  --threads --threads-batch --host --port
)
for option in "${REQUIRED_OPTIONS[@]}"; do
  require_option "$option"
done

# ── CPU thread defaults ──────────────────────────────────────────────────────
LOGICAL="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 1)"
PHYS="$(lscpu -p=CORE,SOCKET 2>/dev/null | awk -F, '!/^#/ { key=$1 "," $2; if (!seen[key]++) n++ } END { print n+0 }')"
if [[ ! "$PHYS" =~ ^[0-9]+$ ]] || (( PHYS < 1 )); then
  PHYS=$(( LOGICAL > 1 ? LOGICAL / 2 : 1 ))
fi
THREADS="${THREADS:-$PHYS}"
THREADS_BATCH="${THREADS_BATCH:-$LOGICAL}"

# ── Port and swap checks ─────────────────────────────────────────────────────
if command -v ss >/dev/null 2>&1; then
  if ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
    echo "ERROR: TCP port $PORT is already listening. Stop the existing server or choose another LLM_PORT." >&2
    exit 1
  fi
elif command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: TCP port $PORT is already listening. Stop the existing server or choose another LLM_PORT." >&2
    exit 1
  fi
else
  echo "WARN: neither ss nor lsof is installed; cannot pre-check whether port $PORT is in use." >&2
fi

if [[ "${LLM_SWAP_CHECK:-0}" == "1" ]]; then
  SWAP_TOTAL="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  if [[ "${SWAP_TOTAL:-0}" -lt 8000000 ]]; then
    echo "WARN: swap is ${SWAP_TOTAL:-0} kB (0 if none). Recommend at least 8 GB." >&2
    echo "      Example (requires root): fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile" >&2
  fi
fi

# ── Optional server arguments ────────────────────────────────────────────────
EXTRA=()

# Qwen chat-template option. Keep it only when the installed server supports it.
if has_option --chat-template-kwargs; then
  EXTRA+=(--chat-template-kwargs '{"preserve_thinking": true}')
else
  echo "WARN: $SERVER does not advertise --chat-template-kwargs; preserve_thinking will not be set." >&2
fi

# The old --prompt-cache FNAME / --prompt-cache-all switches are llama-cli
# options, not llama-server disk persistence. Current server builds use an
# in-memory cache instead. Preserve old env compatibility but explain the change.
if [[ -n "${LLM_PROMPT_CACHE:-}" ]]; then
  echo "NOTE: LLM_PROMPT_CACHE is a legacy setting. llama-server does not restore a disk KV-cache file; use LLM_CACHE_RAM (MiB) for its in-memory prompt cache." >&2
fi

if [[ -n "${LLM_CACHE_RAM:-}" ]]; then
  if has_option --cache-ram; then
    EXTRA+=(--cache-ram "$LLM_CACHE_RAM")
  else
    echo "WARN: this server does not advertise --cache-ram; ignoring LLM_CACHE_RAM." >&2
  fi
fi

case "${LLM_CACHE_PROMPT:-}" in
  1|true|TRUE|yes|YES)
    if has_option --cache-prompt; then EXTRA+=(--cache-prompt); else echo "WARN: --cache-prompt is unsupported; leaving the server default." >&2; fi
    ;;
  0|false|FALSE|no|NO)
    if has_option --no-cache-prompt; then EXTRA+=(--no-cache-prompt); else echo "WARN: --no-cache-prompt is unsupported; leaving the server default." >&2; fi
    ;;
esac

if [[ "${LLM_CACHE_IDLE_SLOTS:-0}" == "1" ]]; then
  if has_option --cache-idle-slots; then
    EXTRA+=(--cache-idle-slots)
  else
    echo "WARN: this server does not advertise --cache-idle-slots; ignoring LLM_CACHE_IDLE_SLOTS." >&2
  fi
fi

if [[ -n "${LLM_CHECKPOINTS:-}" ]]; then
  if has_option --ctx-checkpoints; then
    EXTRA+=(--ctx-checkpoints "$LLM_CHECKPOINTS")
  else
    echo "WARN: context checkpoints requested, but --ctx-checkpoints is unsupported by this server." >&2
  fi
  if has_option --checkpoint-min-step; then
    EXTRA+=(--checkpoint-min-step "${LLM_CHECKPOINT_STEP:-2048}")
  elif has_option --checkpoint-every-n-tokens; then
    EXTRA+=(--checkpoint-every-n-tokens "${LLM_CHECKPOINT_STEP:-1024}")
  fi
fi

# 0 means no cap in this launcher (do NOT pass --reasoning-budget 0: llama.cpp
# interprets that as ending reasoning immediately). Positive values cap tokens.
if [[ "$REASONING_BUDGET" != "0" ]]; then
  require_option --reasoning-budget
  EXTRA+=(--reasoning-budget "$REASONING_BUDGET")
  if [[ "$REASONING_BUDGET" != "-1" ]]; then
    if has_option --reasoning-budget-message; then
      EXTRA+=(--reasoning-budget-message "$REASONING_MESSAGE")
    else
      echo "WARN: --reasoning-budget-message is unsupported; the token budget will still be applied without the custom message." >&2
    fi
  fi
fi

# mlock is deliberately opt-in. A fixed MemoryMax limit plus mlock can make
# this model fail under the cgroup cap, so reject that combination.
if [[ "${LLM_MLOCK:-0}" == "1" ]]; then
  if [[ -n "$LLM_MAX_MEM" ]]; then
    echo "ERROR: LLM_MLOCK=1 cannot be combined with LLM_MAX_MEM=$LLM_MAX_MEM." >&2
    echo "       Disable the cap explicitly with LLM_MAX_MEM= before enabling mlock." >&2
    exit 1
  fi
  if has_option --mlock; then
    EXTRA+=(--mlock)
  else
    echo "ERROR: LLM_MLOCK=1 was requested, but this server does not advertise --mlock." >&2
    exit 1
  fi
fi

# ── Memory cap / systemd scope ────────────────────────────────────────────────
LAUNCH=()
if [[ -n "$LLM_MAX_MEM" ]]; then
  mem_str="$LLM_MAX_MEM"
  mem_bytes=""
  mem_val=""
  mem_unit=""
  if [[ "$mem_str" =~ ^([0-9]+)[Gg]$ ]]; then
    mem_val="${BASH_REMATCH[1]}"
    mem_unit="G"
    mem_bytes=$(( mem_val * 1073741824 ))
  elif [[ "$mem_str" =~ ^([0-9]+)[Mm]$ ]]; then
    mem_val="${BASH_REMATCH[1]}"
    mem_unit="M"
    mem_bytes=$(( mem_val * 1048576 ))
  else
    echo "ERROR: invalid LLM_MAX_MEM='$mem_str'. Use a whole number such as 12G or 12288M, or set it empty to disable the cap." >&2
    exit 1
  fi

  if (( mem_bytes < 1 )); then
    echo "ERROR: LLM_MAX_MEM must be greater than zero." >&2
    exit 1
  fi
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: systemd-run is unavailable; set LLM_MAX_MEM= to run without a memory cap." >&2
    exit 1
  fi

  # A fixed --unit=llama-server collides with a previously loaded
  # llama-server.scope. Unique scope names allow repeated starts without trying
  # to stop or kill an unrelated server; the port check above catches live use.
  SCOPE_UNIT="llama-server-${UID}-${BASHPID}"
  echo "Memory cap: ${mem_val}${mem_unit} (MemoryMax), MemoryHigh at 90%"
  LAUNCH=(systemd-run --scope --quiet --unit="$SCOPE_UNIT" \
    -p "MemoryMax=${mem_bytes}" \
    -p "MemoryHigh=$(( mem_bytes * 90 / 100 ))")
else
  echo "Memory cap: disabled (LLM_MAX_MEM is empty)"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo "llama-server version: ${SERVER_VERSION:-unknown}"
echo "Binary:   $SERVER"
echo "Starting Qwen3.6-35B-A3B..."
echo "Model:    $MODEL"
echo "Context:  $CTX tokens | KV cache: K=$KV_K V=$KV_V"
echo "Offload:  --n-cpu-moe $NCMOE (first N MoE layers kept in system RAM)"
echo "Threads:  decode=$THREADS batch=$THREADS_BATCH (physical=$PHYS, logical=$LOGICAL)"
echo "Batch:    $BATCH | Micro-batch: $UBATCH"
echo "Extras:   ${EXTRA[*]:-none}"
echo "Server:   http://$HOST:$PORT"

SERVER_ARGS=(
  --model "$MODEL"
  --alias qwen3.6-35b-a3b
  --n-gpu-layers 99
  --n-cpu-moe "$NCMOE"
  --fit off
  --ctx-size "$CTX"
  --parallel 1
  --flash-attn on
  --cache-type-k "$KV_K"
  --cache-type-v "$KV_V"
  --batch-size "$BATCH"
  --ubatch-size "$UBATCH"
  --jinja
  --temp 0.6
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty "$PRESENCE_PENALTY"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  "${EXTRA[@]}"
  --host "$HOST"
  --port "$PORT"
)

run_server() {
  "${LAUNCH[@]}" "$SERVER" "${SERVER_ARGS[@]}"
}

if [[ -n "$LLM_LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LLM_LOG_FILE")"
  echo "Logging to $LLM_LOG_FILE"
  run_server 2>&1 | tee -a "$LLM_LOG_FILE"
else
  run_server
fi

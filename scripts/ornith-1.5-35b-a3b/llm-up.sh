#!/usr/bin/env bash
# Launch Ornith-1.5-35B-A3B-IQ4_XS with llama.cpp on:
#   NVIDIA Tesla V100 16 GB VRAM + 16 GB system RAM
# Target: one agent session, 256K context, deferred compaction at 75% (~197K tokens).
#
# Hardware/runtime notes:
# - This is a Qwen3.5-MoE model: 40 text layers, 256 routed experts,
#   8 experts active/token, 30 linear-attention layers and 10 full-attention
#   layers. The model advertises a native 262144-token context.
# - IQ4_XS is about 19.28 GB on disk, so it cannot be fully resident in 16 GB
#   VRAM. --n-cpu-moe keeps the first N layers' routed expert weights in RAM.
# - Start at 18 CPU-MoE layers. This is a tuning point, NOT a guaranteed fit;
#   check the startup log and both RAM/VRAM. For CUDA OOM, first try reducing
#   MOE_CACHE_MIB (1024 -> 512) or raise NCMOE to 19/20 if host RAM has room.
#   Lower NCMOE to 16/17 only if host RAM is pressured AND VRAM can absorb it.
# - The full 256K Q8_0 KV cache for the 10 full-attention layers is roughly 2.66 GiB,
#   excluding runtime buffers. Linear-attention recurrent state is additional
#   but comparatively small. Batch buffers, MoE cache and mmproj also use VRAM.
# - Keep mmap enabled (llama.cpp default). Do NOT use --mlock or a restrictive
#   systemd MemoryMax on a 16 GB RAM machine. Swap is not a speed solution.
# - V100 is compute capability 7.0. Build llama.cpp with CUDA Toolkit 12.9 or
#   earlier and CMAKE_CUDA_ARCHITECTURES=70; CUDA 13.0 dropped offline support
#   for Volta. Use a checkout containing llama.cpp commit 77e2db8 or newer: it
#   adds the SM70 Flash-Attention config for Dk=256, DV=256 used by this model.
#
# Suggested build (run from a checkout updated to current upstream master):
#   git -C "$HOME/llama.cpp" pull --ff-only
#   cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" \
#     -DGGML_CUDA=ON \
#     -DCMAKE_CUDA_ARCHITECTURES=70 \
#     -DGGML_CUDA_FA_QUANTS=q8_0-q8_0 \
#     -DLLAMA_BUILD_SERVER=ON
#   cmake --build "$HOME/llama.cpp/build" --config Release \
#     --target llama-server --parallel
#
# Examples:
#   ./llm-up.sh
#   LLM_NCMOE=19 ./llm-up.sh
#   LLM_NCMOE=20 LLM_MOE_CACHE_MIB=512 ./llm-up.sh  # reduce VRAM on OOM
#   LLM_NCMOE=16 ./llm-up.sh                       # only if RAM is tight and VRAM fits
#   LLM_CTX=65536 ./llm-up.sh
#   LLM_MOE_CACHE_MIB=0 ./llm-up.sh
#   LLM_USE_MTP=1 ./llm-up.sh     # optional; consumes extra runtime/VRAM
#   LLM_MMPROJ=/path/to/mmproj-Ornith-1.5-35B-A3B-f16.gguf ./llm-up.sh
#
# Environment overrides:
#   LLM_MODEL, LLM_SERVER, LLM_HOST, LLM_PORT
#   LLM_CTX=131072, LLM_NCMOE=18, LLM_N_GPU_LAYERS=auto
#   LLM_FIT_TARGET=1024 (MiB per GPU), LLM_MOE_CACHE_MIB=1024
#   LLM_CACHE_RAM=1536 (MiB), LLM_BATCH=512, LLM_UBATCH=128
#   LLM_KV_K=q8_0, LLM_KV_V=q8_0
#   LLM_THREADS, LLM_THREADS_BATCH
#   LLM_TEMP=0.6, LLM_TOP_P=0.95, LLM_TOP_K=20, LLM_MIN_P=0.0
#   LLM_COMPACTION_AT=196608 (client threshold; deferred-compaction targets 75% of LLM_CTX)
#   LLM_MMPROJ= optional image projector; LLM_USE_MTP=0/1
#   LLM_LOG_FILE (empty disables tee logging)

set -Eeuo pipefail

MODEL="${LLM_MODEL:-/home/themachinist/models/ornith1.5/Ornith-1.5-35B-A3B-IQ4_XS.gguf}"
SERVER="${LLM_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8080}"

CTX="${LLM_CTX:-262144}"
NCMOE="${LLM_NCMOE:-18}"
N_GPU_LAYERS="${LLM_N_GPU_LAYERS:-auto}"
FIT="${LLM_FIT:-off}"
FIT_TARGET="${LLM_FIT_TARGET:-1536}"
MOE_CACHE_MIB="${LLM_MOE_CACHE_MIB:-1024}"
CACHE_RAM="${LLM_CACHE_RAM:-1536}"
BATCH="${LLM_BATCH:-2048}"
UBATCH="${LLM_UBATCH:-512}"
OFFLOAD_MIN_BATCH="${LLM_OFFLOAD_MIN_BATCH:-}"
CTX_CHECKPOINTS="${LLM_CTX_CHECKPOINTS:-}"
CHECKPOINT_EVERY="${LLM_CHECKPOINT_EVERY:-}"
KV_K="${LLM_KV_K:-q8_0}"
KV_V="${LLM_KV_V:-q8_0}"
THREADS="${LLM_THREADS:-}"
THREADS_BATCH="${LLM_THREADS_BATCH:-}"

TEMP="${LLM_TEMP:-0.6}"
TOP_P="${LLM_TOP_P:-0.95}"
TOP_K="${LLM_TOP_K:-20}"
MIN_P="${LLM_MIN_P:-0.0}"
COMPACTION_AT="${LLM_COMPACTION_AT:-196608}"
MMPROJ="${LLM_MMPROJ:-}"
USE_MTP="${LLM_USE_MTP:-0}"
THINKING="${LLM_THINKING:-0}"
LOG_FILE="${LLM_LOG_FILE-$HOME/llm-server.log}"

fail() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN: $*" >&2; }

# ── Basic validation ─────────────────────────────────────────────────────────
[[ -x "$SERVER" ]] || fail "llama-server not executable: $SERVER
Build with CUDA support for sm_70 (see build instructions at the top of this file)."
[[ -f "$MODEL" ]] || fail "Model not found: $MODEL"

[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) \
  || fail "LLM_PORT must be an integer from 1 to 65535 (got '$PORT')."

for name in CTX NCMOE FIT_TARGET MOE_CACHE_MIB CACHE_RAM BATCH UBATCH TOP_K COMPACTION_AT; do
  value="${!name}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a non-negative integer (got '$value')."
done
(( CTX >= 1 && CTX <= 262144 )) || fail "CTX must be between 1 and the model's 262144 native context."
(( NCMOE <= 40 )) || fail "NCMOE cannot exceed this model's 40 text layers."
(( BATCH >= 1 && UBATCH >= 1 && UBATCH <= BATCH )) || fail "Require BATCH >= UBATCH >= 1."
(( FIT_TARGET >= 256 )) || fail "FIT_TARGET should be at least 256 MiB."
[[ "$USE_MTP" == "0" || "$USE_MTP" == "1" ]] || fail "LLM_USE_MTP must be 0 or 1."
[[ -z "$OFFLOAD_MIN_BATCH" || "$OFFLOAD_MIN_BATCH" =~ ^[0-9]+$ ]] \
  || fail "LLM_OFFLOAD_MIN_BATCH must be a non-negative integer when set (got '$OFFLOAD_MIN_BATCH')."
for name in CTX_CHECKPOINTS CHECKPOINT_EVERY; do
  value="${!name}"
  [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] \
    || fail "LLM_${name} must be a positive integer when set (got '$value'); 0/-1 would disable checkpoints and force full prompt re-processing on this hybrid model."
done
if [[ -n "$CTX_CHECKPOINTS" ]] && (( CTX_CHECKPOINTS * 64 + CACHE_RAM > 4096 )); then
  warn "LLM_CTX_CHECKPOINTS=$CTX_CHECKPOINTS needs roughly $(( CTX_CHECKPOINTS * 64 )) MiB of host RAM (about 64 MiB each) on top of --cache-ram $CACHE_RAM MiB. On a 16 GB host that also holds CPU-resident experts, this risks paging."
fi
[[ "$THINKING" == "0" || "$THINKING" == "1" ]] || fail "LLM_THINKING must be 0 or 1."
if [[ "$THINKING" == "1" ]]; then REASONING_MODE=on; THINK_KW=true; else REASONING_MODE=off; THINK_KW=false; fi
[[ "$FIT" == "on" || "$FIT" == "off" ]] || fail "LLM_FIT must be 'on' or 'off' (got '$FIT')."
if [[ "$FIT" == "on" ]] && (( NCMOE > 0 )); then
  warn "LLM_FIT=on with LLM_NCMOE=$NCMOE: llama.cpp aborts auto-fit when --n-cpu-moe already sets tensor overrides, so fit would do nothing. Use LLM_FIT=off (default) or set LLM_NCMOE=0 to let fit place the experts."
fi

case "$KV_K" in q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1|f16|bf16|f8_e4m3) ;; *) fail "Unsupported LLM_KV_K: $KV_K" ;; esac
case "$KV_V" in q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1|f16|bf16|f8_e4m3) ;; *) fail "Unsupported LLM_KV_V: $KV_V" ;; esac

# Warn if the GGUF appears incomplete. HF lists this exact IQ4_XS file as 19.28 GB.
MODEL_BYTES="$(stat -c '%s' "$MODEL" 2>/dev/null || echo 0)"
MODEL_GIB="$(awk -v n="$MODEL_BYTES" 'BEGIN { if (n>0) printf "%.2f", n/1073741824; else print "unknown" }')"
if (( MODEL_BYTES > 0 && MODEL_BYTES < 18500000000 )); then
  warn "Model file is ${MODEL_GIB} GiB; expected approximately 17.96 GiB (19.28 GB). Verify the download."
fi

# CPU thread defaults: physical cores for decode; logical CPUs for prompt processing.
LOGICAL="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 1)"
PHYS="$(lscpu -p=CORE,SOCKET 2>/dev/null | awk -F, '!/^#/ { k=$1 "," $2; if (!seen[k]++) n++ } END { print n+0 }')"
if [[ ! "$PHYS" =~ ^[0-9]+$ ]] || (( PHYS < 1 )); then
  PHYS=$(( LOGICAL > 1 ? LOGICAL / 2 : 1 ))
fi
THREADS="${THREADS:-$PHYS}"
THREADS_BATCH="${THREADS_BATCH:-$LOGICAL}"
[[ "$THREADS" =~ ^[0-9]+$ ]] && (( THREADS >= 1 )) || fail "LLM_THREADS must be a positive integer."
[[ "$THREADS_BATCH" =~ ^[0-9]+$ ]] && (( THREADS_BATCH >= 1 )) || fail "LLM_THREADS_BATCH must be a positive integer."

# ── RAM / GPU checks ─────────────────────────────────────────────────────────
if [[ -r /proc/meminfo ]]; then
  MEM_AVAILABLE_KB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ "${MEM_AVAILABLE_KB:-0}" =~ ^[0-9]+$ ]] && (( MEM_AVAILABLE_KB < 8000000 )); then
    warn "MemAvailable is below 8 GiB before model load. Close other applications; CPU-resident experts need host RAM."
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -n "$GPU_INFO" ]]; then
    echo "GPU:       $GPU_INFO"
    grep -qi 'V100' <<<"$GPU_INFO" || warn "Detected GPU is not named V100; this setup is tuned for Tesla V100 16 GB."
  else
    warn "nvidia-smi could not query a GPU; check the NVIDIA driver."
  fi
else
  warn "nvidia-smi is unavailable; cannot verify GPU state."
fi

# Query the exact llama-server binary being launched.
SERVER_VERSION="$("$SERVER" --version 2>&1 || true)"
SERVER_HELP="$("$SERVER" --help 2>&1 || true)"
[[ -n "$SERVER_HELP" ]] || fail "Could not read help output from $SERVER"

has_option() {
  local opt="$1"
  grep -Eq -- "(^|[[:space:],])${opt}([[:space:],=]|$)" <<<"$SERVER_HELP"
}
require_option() {
  local opt="$1"
  has_option "$opt" || fail "This llama-server does not advertise '$opt'. Version: ${SERVER_VERSION:-unknown}. Update llama.cpp."
}

REQUIRED_OPTIONS=(
  --model --alias --n-gpu-layers --n-cpu-moe --fit --fit-target --ctx-size
  --parallel --flash-attn --cache-type-k --cache-type-v --batch-size --ubatch-size
  --jinja --reasoning --temp --top-p --top-k --min-p --threads --threads-batch
  --host --port
)
for opt in "${REQUIRED_OPTIONS[@]}"; do require_option "$opt"; done

# Confirm this exact binary sees a CUDA device. A driver-visible GPU alone is not enough.
if has_option --list-devices; then
  DEVICE_LIST="$("$SERVER" --list-devices 2>&1 || true)"
  echo "llama.cpp devices:"
  printf '%s\n' "$DEVICE_LIST"
  grep -Eiq 'CUDA' <<<"$DEVICE_LIST" || fail "llama-server does not list a CUDA device; rebuild with GGML_CUDA=ON and sm_70 support."
else
  warn "This build has no --list-devices option; CUDA offload cannot be verified before launch."
fi

# Port pre-check.
if command -v ss >/dev/null 2>&1; then
  if ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
    fail "TCP port $PORT is already listening. Stop that server or set LLM_PORT."
  fi
elif command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "TCP port $PORT is already listening. Stop that server or set LLM_PORT."
  fi
else
  warn "Neither ss nor lsof is installed; cannot pre-check whether port $PORT is in use."
fi

# ── Optional options ──────────────────────────────────────────────────────────
EXTRA=()

# This exact model template checks the boolean variable `enable_thinking`.
# --reasoning is the primary switch; the template kwarg reinforces it.
if has_option --chat-template-kwargs; then
  EXTRA+=(--chat-template-kwargs "{\"enable_thinking\":${THINK_KW}}")
else
  warn "--chat-template-kwargs is unavailable; --reasoning $REASONING_MODE remains in effect."
fi

# Cache CPU-resident experts in VRAM for decode-time reuse. Set 0 to disable.
if has_option --moe-cache-mib; then
  EXTRA+=(--moe-cache-mib "$MOE_CACHE_MIB")
else
  warn "--moe-cache-mib is unavailable; CPU expert GPU cache disabled."
fi

# Limit idle-slot/prompt-cache RAM; this is a host-RAM cap, not the live KV size.
if has_option --cache-ram; then
  EXTRA+=(--cache-ram "$CACHE_RAM")
else
  warn "--cache-ram is unavailable; llama.cpp's default prompt-cache RAM policy applies."
fi

# Hybrid (linear-attention) models cannot truncate their recurrent state, so prefix reuse in an
# agent loop depends on saved context checkpoints. Both options are opt-in: when unset the
# server's own defaults apply (32 checkpoints per slot, one every 8192 tokens). Each checkpoint
# is roughly 63 MiB on Qwen3.x-35B-A3B-class models, so more checkpoints cost host RAM.
if [[ -n "$CTX_CHECKPOINTS" ]]; then
  if has_option --ctx-checkpoints; then
    EXTRA+=(--ctx-checkpoints "$CTX_CHECKPOINTS")
  else
    warn "--ctx-checkpoints is unavailable; LLM_CTX_CHECKPOINTS ignored."
  fi
fi
if [[ -n "$CHECKPOINT_EVERY" ]]; then
  if has_option --checkpoint-every-n-tokens; then
    EXTRA+=(--checkpoint-every-n-tokens "$CHECKPOINT_EVERY")
  else
    warn "--checkpoint-every-n-tokens is unavailable; LLM_CHECKPOINT_EVERY ignored."
  fi
fi

# Explicitly avoid silent context shifting; the client should compact before the limit.
if has_option --no-context-shift; then
  EXTRA+=(--no-context-shift)
else
  warn "--no-context-shift is unavailable; verify the server's default context-shift behavior."
fi

# Optional MTP: the GGUF contains an MTP layer, but leave it off by default to
# preserve VRAM for the 128K context. Enable only after the base model is stable.
if [[ "$USE_MTP" == "1" ]]; then
  require_option --spec-type
  require_option --spec-draft-n-max
  EXTRA+=(--spec-type draft-mtp --spec-draft-n-max 2)
fi

# Optional vision projector. Text-only agent use does not need this; it consumes
# additional host/GPU memory. Use the repository's f16 projector on a V100.
if [[ -n "$MMPROJ" ]]; then
  [[ -f "$MMPROJ" ]] || fail "LLM_MMPROJ was set but file does not exist: $MMPROJ"
  require_option --mmproj
  EXTRA+=(--mmproj "$MMPROJ")
fi

# Prompt batches of at least this many tokens are processed on the GPU, which copies the
# CPU-resident experts over PCIe first. llama.cpp's default is 32. Leave unset unless
# benchmarking shows short agent tool-result prefills are faster on the CPU.
if [[ -n "$OFFLOAD_MIN_BATCH" ]]; then export GGML_OP_OFFLOAD_MIN_BATCH="$OFFLOAD_MIN_BATCH"; fi

if [[ -n "$LOG_FILE" ]]; then mkdir -p "$(dirname "$LOG_FILE")"; fi

# ── Launch summary ───────────────────────────────────────────────────────────
echo "llama-server: $SERVER_VERSION"
echo "Binary:      $SERVER"
echo "Model:       $MODEL ($MODEL_GIB GiB on disk)"
echo "Context:     $CTX tokens | one slot | KV K=$KV_K V=$KV_V"
echo "CPU MoE:     first $NCMOE of 40 layers kept in host RAM"
if [[ "$FIT" == "on" ]]; then
  echo "GPU layers:  $N_GPU_LAYERS | auto-fit ON, target ${FIT_TARGET} MiB"
else
  echo "GPU layers:  $N_GPU_LAYERS | auto-fit OFF (placement is manual: NCMOE=$NCMOE)"
fi
echo "MoE cache:   ${MOE_CACHE_MIB} MiB (if supported)"
echo "KV/cache:    K=$KV_K V=$KV_V; idle prompt cache capped at ${CACHE_RAM} MiB (if supported)"
echo "Checkpoints: max ${CTX_CHECKPOINTS:-server default} per slot | every ${CHECKPOINT_EVERY:-server default} tokens (~63 MiB host RAM each)"
echo "Batches:     batch=$BATCH | ubatch=$UBATCH | GPU-offload min batch=${OFFLOAD_MIN_BATCH:-default(32)}"
echo "If startup fails with CUDA OOM: LLM_UBATCH=256, or LLM_NCMOE=$((NCMOE+1)), or LLM_MOE_CACHE_MIB=512"
echo "Threads:     decode=$THREADS (physical=$PHYS) | batch=$THREADS_BATCH (logical=$LOGICAL)"
echo "Sampling:    temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P"
if [[ "$THINKING" == "1" ]]; then
  echo "Thinking:    ENABLED (--reasoning on + enable_thinking=true); reasoning tokens count toward the context"
else
  echo "Thinking:    disabled (--reasoning off + enable_thinking=false where supported)"
fi
echo "Compaction:  configure in the agent client at ~$COMPACTION_AT tokens; server does not summarize history"
echo "MTP:         $USE_MTP"
echo "Vision:      ${MMPROJ:-disabled}"
echo "Server:      http://$HOST:$PORT"
echo "RAM policy:  mmap/default loading; no mlock, swap policy, or cgroup MemoryMax applied"

# Expert placement. --n-cpu-moe is only passed when > 0 so that LLM_NCMOE=0 leaves
# tensor overrides unset and --fit (if enabled) is free to place experts itself.
PLACEMENT=()
(( NCMOE > 0 )) && PLACEMENT+=(--n-cpu-moe "$NCMOE")
[[ "$FIT" == "on" ]] && PLACEMENT+=(--fit-target "$FIT_TARGET")

SERVER_ARGS=(
  --model "$MODEL"
  --alias ornith-1.5-35b-a3b-iq4-xs
  --n-gpu-layers "$N_GPU_LAYERS"
  --fit "$FIT"
  --ctx-size "$CTX"
  --parallel 1
  --flash-attn on
  --cache-type-k "$KV_K"
  --cache-type-v "$KV_V"
  --batch-size "$BATCH"
  --ubatch-size "$UBATCH"
  --jinja
  --reasoning "$REASONING_MODE"
  --temp "$TEMP"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  --host "$HOST"
  --port "$PORT"
  ${PLACEMENT[@]+"${PLACEMENT[@]}"}
  "${EXTRA[@]}"
)

# exec replaces this shell with llama-server, so SIGTERM/SIGINT from a terminal, `kill`, or a
# supervisor such as systemd reach the server itself and it can shut down cleanly. With the old
# `server | tee` pipeline bash stayed as the parent: SIGTERM killed bash and left the server
# running as an orphan, and Ctrl-C killed tee before the server's shutdown messages were logged.
# Logging therefore uses a process substitution whose tee ignores INT/TERM (dispositions set to
# "ignored" survive exec); it exits on its own when the server closes the pipe.
if [[ -n "$LOG_FILE" ]]; then
  if : >>"$LOG_FILE" 2>/dev/null; then
    echo "Logging to $LOG_FILE"
    exec > >(trap '' INT TERM; exec tee -a -- "$LOG_FILE") 2>&1
  else
    warn "Cannot write to $LOG_FILE; server output goes to the terminal only."
  fi
fi
exec "$SERVER" "${SERVER_ARGS[@]}"
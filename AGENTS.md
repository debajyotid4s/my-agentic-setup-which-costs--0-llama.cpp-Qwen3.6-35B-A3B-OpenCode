# Local agentic coding with llama.cpp + OpenCode

Zero-API-cost coding agent — runs a local LLM via `llama-server` on your NVIDIA GPU, with OpenCode as the agent interface.

## What this repo does

Installs and configures `llama.cpp` + `llama-server` + OpenCode so the model runs on your GPU and OpenCode talks to it over `http://127.0.0.1:8080/v1`.

The global OpenCode config (`opencode.json`, `AGENTS.md`, `/handoff` command, compaction plugin) is in `config/` and deployed by `scripts/04-install-opencode-config.sh` to `~/.config/opencode/`.

## Setup order

```
01-install-deps.sh      → system packages, hf CLI, OpenCode (checks CUDA 12.x)
02-build-llama-cpp.sh   → builds llama.cpp (set CMAKE_CUDA_ARCHITECTURES for your GPU)
03-download-model.sh    → downloads a GGUF model (set MODEL_REPO / MODEL_QUANT)
04-install-opencode-config.sh  → deploys global config to ~/.config/opencode/
05-init-project.sh /path/to/project  → creates .agent/HANDOFF.md, SCRATCH.md, opencode.json
scripts/<model>/llm-up.sh  → starts llama-server (run in a dedicated terminal)
scripts/<model>/llm-stop.sh → graceful shutdown (SIGTERM → wait 30s → SIGKILL)
```

Override any tunable per-run: `LLM_NCMOE=16 ./scripts/ornith-1.5-35b-a3b/llm-up.sh`

## Pick your hardware

**CUDA arch** (set before building): RTX 20xx=75, RTX 30xx=86, RTX 40xx=89, V100/RTX4000-6000=70, GTX 10xx=61

**Model by VRAM**: 8GB→Q5_K_M (~10GB), 12GB→Q6_K (~12GB), 16GB→UD-IQ4_XS (~17.7GB) or Q8_0 (~22GB), 24GB→Q8_0, 48GB+→Llama 70B Q4_K_M (~38GB)

**System RAM**: 16GB=min for small models, 32GB=comfortable for 13B-30B, 64GB+=for 30B-70B

## Critical constraints

- **CUDA 12.x** — check `nvcc --version`. CUDA 13 may drop support for older GPUs.
- **OpenCode 2** — this bundle targets the V2 config and plugin API. `opencode --version` will tell you.
- **One inference slot** (`-np 1`) — all VRAM reserved for one long session.
- **`compaction.auto: false`** — the `deferred-compaction` plugin requests compaction at a safe point. Background compaction stays off because there is one slot.

## Key files

| Path | Purpose |
|---|---|
| `config/opencode.json` | Global V2 config: llama.cpp provider, deferred compaction, permissions |
| `config/AGENTS.global.md` | Global working rules — installed as `~/.config/opencode/AGENTS.md` |
| `config/commands/handoff.md` | The `/handoff` command definition |
| `config/plugins/deferred-compaction/` | V2 safe-point compaction controller |
| `config/project-templates/` | Copied into projects by `05-init-project.sh` |
| `scripts/qwen3.6-35b-a3b/llm-up.sh` | Qwen server launcher — all tunables are env vars |
| `scripts/ornith-1.5-35b-a3b/llm-up.sh` | Ornith server launcher — all tunables are env vars |
| `scripts/*/llm-stop.sh` | Graceful shutdown |
| `Dockerfile` | Multi-stage: build llama.cpp + slim runtime. Model mounted read-only from host |

## Tuning knobs (env vars on `llm-up.sh`)

| Variable | Default | Effect |
|---|---|---|
| `LLM_NCMOE` | 18 | Layers whose experts stay in system RAM (MoE models only). Lower = more VRAM used. |
| `LLM_CTX` | 262144 | Context window. 131072 = faster, less VRAM. |
| `LLM_THREADS` | physical cores | Decode threads. |
| `LLM_UBATCH` | 512 | Micro-batch size. Costs VRAM. |
| `LLM_CHECKPOINTS` | off | Prompt-cache checkpoints for cache reuse. |
| `LLM_PROMPT_CACHE` | off | Save KV cache to disk for warm restarts. |
| `LLM_MLOCK` | off | Lock model/cache in RAM. |
| `LLM_MAX_MEM` | 12G | Cap process memory to prevent OOM reboots. |

## Memory system (per-project)

- `.agent/HANDOFF.md` — long-term memory, auto-loaded. **Rewrite, never append.** Max 60 lines.
- `.agent/SCRATCH.md` — working journal, NOT auto-loaded. Append-only. Read with `tail -n 40`.
- `.agent/` is added to `.git/info/exclude` by `05-init-project.sh`.

Daily flow: work → `/handoff` → `/new`.

### Shutting down

Do NOT Ctrl+C (CUDA cleanup can hang):

```bash
./scripts/<model>/llm-stop.sh           # graceful (30s timeout, then SIGKILL)
./scripts/<model>/llm-stop.sh --force   # immediate kill
```

## Warm start (KV cache persistence)

After idle, if every turn re-reads the whole history:

```bash
LLM_PROMPT_CACHE=1 LLM_MLOCK=1 ./scripts/qwen3.6-35b-a3b/llm-up.sh
```

Trade-off: ~2–4 GB disk, ~5–15s cold start restore.

| Flag | Effect | Trade-off |
|---|---|---|
| `LLM_PROMPT_CACHE=1` | Saves KV cache to `~/.llama-cache/` | ~2–4 GB disk, ~5–15s cold start |
| `LLM_MLOCK=1` | Locks model + cache in RAM | Uses more system RAM |
| Both | Best warm-start: disk + RAM | Combined cost |

## Memory safety (prevents OOM reboots)

If your system restarts under load:

1. **Cap memory** — `LLM_MAX_MEM=10G ./scripts/qwen3.6-35b-a3b/llm-up.sh` (uses `systemd-run --scope MemoryMax=`).
2. **Add swap** — `sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab`
3. **Lower `LLM_NCMOE`** — fewer layers in system RAM (MoE models).
4. **Lower `LLM_CTX`** — `LLM_CTX=131072` saves ~1.3 GB VRAM.

Check swap: `LLM_SWAP_CHECK=1 ./scripts/qwen3.6-35b-a3b/llm-up.sh`

## Docker

```bash
docker build --build-arg CMAKE_CUDA_ARCHITECTURES=70 -t local-llm-opencode:latest .
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro local-llm-opencode:latest
docker stop llm-server     # graceful; docker kill to force
```

The image contains only `llama-server` (pinned `LLAMA_REF=b12375`); OpenCode runs on the host.
For the Ornith command and flag overrides see `README.md`. Append flags after the image name,
e.g. `--n-cpu-moe 16 -c 131072`.

## Quick verification

```bash
curl -s http://127.0.0.1:8080/health
nvidia-smi  # VRAM should stay below your card's total
grep -n "prompt eval time" ~/llm-server.log | tail -3
```

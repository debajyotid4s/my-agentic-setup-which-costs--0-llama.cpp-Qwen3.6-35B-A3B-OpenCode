# qwen-v100-opencode

A reproducible setup for running **Qwen3.6-35B-A3B** locally as a coding agent via **OpenCode** on a **Tesla V100 16 GB** (Volta, sm_70).

## What this repo does

Installs and configures `llama.cpp` + `llama-server` + OpenCode so the model runs on your GPU and OpenCode talks to it over `http://127.0.0.1:8080/v1`. Zero API cost.

The global OpenCode config (`opencode.json`, `AGENTS.md`, `/handoff` command, compaction plugin) is in `config/` and deployed by `scripts/04-install-opencode-config.sh` to `~/.config/opencode/`.

## Setup order (do not skip)

```
01-install-deps.sh      → system packages, hf CLI, OpenCode (checks CUDA 12.x)
02-build-llama-cpp.sh   → builds llama.cpp with sm_70 kernels (prebuilts lack Volta)
03-download-model.sh    → downloads UD-IQ4_XS (~17.7 GB) to ~/models/qwen3.6/
04-install-opencode-config.sh  → deploys global config to ~/.config/opencode/ (backs up existing)
05-init-project.sh /path/to/project  → creates .agent/HANDOFF.md, .agent/SCRATCH.md, project opencode.json
llm-up.sh               → starts llama-server (run in a dedicated terminal)
```

Override any tunable per-run: `LLM_NCMOE=16 ./scripts/llm-up.sh`

## Memory safety (prevents OOM reboots)

If your system restarts under load, the kernel OOM killer is hitting. Fix it:

1. **Cap process memory** — `LLM_MAX_MEM=10G ./scripts/llm-up.sh` (uses `systemd-run --scope MemoryMax=`). The server exits cleanly instead of crashing the system. Start low (8G–10G) and raise if it loads but truncates.
2. **Add swap** — creates a safety net so the OOM killer doesn't fire:
   ```bash
   sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   ```
3. **Lower `LLM_NCMOE`** — fewer layers in system RAM, more on GPU. Each step down frees ~0.37 GB system RAM. Try `LLM_NCMOE=14` if you have ≤16 GB total RAM.
4. **Lower `LLM_CTX`** — `LLM_CTX=131072` saves ~1.3 GB VRAM and proportionally less system RAM.

Check swap status: `LLM_SWAP_CHECK=1 ./scripts/llm-up.sh`

## Critical constraints

- **CUDA 12.x only** — CUDA 13 dropped Volta support. The install scripts check this.
- **OpenCode 1** — this bundle targets OpenCode 1 config. `opencode --version` will tell you. OpenCode 2 shares the binary but has different config.
- **One inference slot** (`-np 1`) — all VRAM reserved for one long session.
- **Model quant**: `UD-IQ4_XS` (not Q4_K_M). IQ4_XS keeps more experts on GPU, saving ~4 GB VRAM. Always download a fresh copy — Unsloth updated the chat template for developer-role support.
- **`prune: false`** in `compaction` — pruning edits old tool output, which invalidates the prompt cache and forces a full re-prefill.

## Key files

| Path | Purpose |
|---|---|
| `config/opencode.json` | Global provider config: llama.cpp provider, compaction (auto, no prune), agent temps |
| `config/AGENTS.global.md` | Global working rules — installed as `~/.config/opencode/AGENTS.md` |
| `config/commands/handoff.md` | The `/handoff` command definition |
| `config/plugins/better-compaction.ts` | Custom compaction prompt (experimental hook) |
| `config/project-templates/` | Copied into projects by `05-init-project.sh` |
| `scripts/llm-up.sh` | Server launcher — all tunables are env vars (see header) |
| `Dockerfile` | Multi-stage: build llama.cpp (sm_70) + slim runtime. Model mounted read-only from host |

## Tuning knobs (env vars on `llm-up.sh`)

| Variable | Default | Effect |
|---|---|---|
| `LLM_NCMOE` | 18 | Layers whose experts stay in system RAM. Lower = faster, uses more VRAM (~0.37 GB/lower). Tune to ~15 GiB used. |
| `LLM_CTX` | 262144 | Context window. 131072 = faster, less VRAM, ~90K compaction point. |
| `LLM_THREADS` | physical cores | Decode threads. Try 3 on a 4-core CPU. |
| `LLM_UBATCH` | 512 | Micro-batch size. Costs VRAM. |
| `LLM_CHECKPOINTS` | off | Prompt-cache checkpoints for cache reuse. Each ~60 MiB system RAM. |
| `LLM_REASONING_BUDGET` | off | Caps thinking tokens. |

## Memory system (per-project)

- `.agent/HANDOFF.md` — long-term memory, auto-loaded on every request. **Rewrite, never append.** Max 60 lines. Edit forces full re-prefill.
- `.agent/SCRATCH.md` — working journal, NOT auto-loaded. Append-only. Read with `tail -n 40`.
- `.agent/` is added to `.git/info/exclude` by `05-init-project.sh`.

Daily flow: work → `/handoff` → `/new`. The new session loads HANDOFF.md automatically.

### Shutting down

Do NOT Ctrl+C to stop the server (CUDA cleanup can hang):

```bash
./scripts/llm-stop.sh           # graceful (30s timeout, then SIGKILL)
./scripts/llm-stop.sh --force   # immediate kill
```

## KV cache reuse (the #1 perf issue)

**Warm start after idle (the #1 user complaint):** If every turn after an hour of idle re-reads the whole history:

1. **Disable idle timeout** — already done by default (`--timeout 0` in `llm-up.sh`). If you see this, the server was started without it.
2. **Enable prompt-cache persistence** — saves KV cache to disk, restores on restart (~5-15s cold start vs full re-prefill):
   ```bash
   LLM_PROMPT_CACHE=1 LLM_MLOCK=1 ./scripts/llm-up.sh
   ```
   Trade-off: ~2-4 GB disk, ~5-15s restore time on cold start instead of instant.
3. **Stay in one agent per session** — Build vs Plan have different tool defs → no prefix match.
4. **Run** `grep -n "forcing full prompt re-processing" ~/llm-server.log` — expect no matches.
5. If still broken: enable checkpoints (`LLM_CHECKPOINTS=16`), or lower compaction point to 128K profile

## Warm start flags (llm-up.sh)

| Flag | Effect | Trade-off |
|---|---|---|
| `LLM_PROMPT_CACHE=1` | Saves KV cache to `~/.llama-cache/`, restores on restart | ~2-4 GB disk, ~5-15s cold start |
| `LLM_MLOCK=1` | Locks model + cache in RAM, prevents OS page eviction | Uses more system RAM |
| Both together | Best warm-start: disk persistence + RAM residency | Combined disk + RAM cost |

## Quick verification

```bash
curl -s http://127.0.0.1:8080/health
nvidia-smi  # VRAM should stay below ~15 GiB
grep -n "prompt eval time" ~/llm-server.log | tail -3
```

## Docker

```bash
docker build -t qwen-v100-opencode:latest .
docker run --rm -it --gpus all -p 8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro \
  qwen-v100-opencode:latest
```

Override CMD flags by appending: `--n-cpu-moe 16 -c 131072`

**Warm-start in Docker** — mount the prompt-cache volume for KV cache persistence:
```bash
docker run --rm -it --gpus all -p 8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro \
  -v ~/llama-cache:/llama-cache \
  qwen-v100-opencode:latest \
  --prompt-cache /llama-cache/qwen3.6-35b-a3b-kvcache.bin --prompt-cache-all
```

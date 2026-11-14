# Local agentic coding with llama.cpp + OpenCode

Zero-API-cost coding agent: a local LLM served by `llama-server`, driven by **OpenCode v2.3.0**
(V2 config and plugin API). Two launchers are included: `qwen3.6-35b-a3b` and
`ornith-1.5-35b-a3b`. Target hardware: one NVIDIA GPU (developed on a 16 GB Tesla V100).

## Layout

```
scripts/
  01-install-deps.sh              packages, hf CLI, OpenCode
  02-build-llama-cpp.sh           llama.cpp build (set CMAKE_CUDA_ARCHITECTURES)
  03-download-model.sh            GGUF download (MODEL_REPO, MODEL_QUANT)
  04-install-opencode-config.sh   deploy config/ to ~/.config/opencode/
  05-init-project.sh <dir>        per-project .agent/ memory files
  <model>/llm-up.sh               start llama-server
  <model>/llm-stop.sh             graceful shutdown (SIGTERM -> 30s -> SIGKILL)
config/
  opencode.json                   -> ~/.config/opencode/opencode.json
  AGENTS.global.md                -> ~/.config/opencode/AGENTS.md
  commands/handoff.md             -> ~/.config/opencode/commands/handoff.md
  plugins/deferred-compaction/    -> ~/.config/opencode/plugins/deferred-compaction/
  project-templates/              copied into projects by 05-init-project.sh
```

## Hardware

| GPU | `CMAKE_CUDA_ARCHITECTURES` | CUDA |
|---|---|---|
| Tesla V100 / Quadro RTX 4000–6000 | `70` | **12.x only** (CUDA 13 dropped Volta) |
| GTX 10xx (Pascal) | `61` | 12.x only |
| RTX 20xx / T4 | `75` | 12.x or 13.x |
| RTX 30xx / A100 | `86` | 12.x or 13.x |
| RTX 40xx | `89` | 12.x or 13.x |
| RTX 50xx / H100 | `90` | 12.x or 13.x |

Models (MoE, hybrid CPU/GPU via `--n-cpu-moe`):

| Model | Quant | Disk | `LLM_NCMOE` start |
|---|---|---|---|
| Qwen3.6-35B-A3B | UD-IQ4_XS | ~17.7 GB | 18 |
| Ornith-1.5-35B-A3B | IQ4_XS | ~19.3 GB | 18 |

System RAM: 32 GB+ is comfortable; 16 GB needs a higher `LLM_NCMOE` and no `LLM_MAX_MEM`/`--mlock`.

## Setup

```bash
cd qwen-v100-opencode
chmod +x scripts/*.sh
./scripts/01-install-deps.sh                                  # SKIP_OPENCODE=1 to skip
CMAKE_CUDA_ARCHITECTURES=70 ./scripts/02-build-llama-cpp.sh   # LLAMA_REF=<tag> to pin
MODEL_REPO=<repo> MODEL_QUANT=UD-IQ4_XS ./scripts/03-download-model.sh
./scripts/04-install-opencode-config.sh
./scripts/05-init-project.sh /path/to/project
```

Start a server in its own terminal (do not Ctrl+C it):

```bash
./scripts/ornith-1.5-35b-a3b/llm-up.sh 2>&1 | tee -a ~/llm-server.log
curl -s http://127.0.0.1:8080/health
```

Then run `opencode` in the project directory. OpenCode installs its VS Code extension automatically.

Verify provider/model/context: `opencode api get /api/model/default`.

## Tuning

All knobs are environment variables on `llm-up.sh`.

| Variable | Default | Effect |
|---|---|---|
| `LLM_CTX` | `262144` | Context window; ~1.3 GB VRAM saved at `131072` |
| `LLM_NCMOE` | `18` | MoE layers whose experts stay in system RAM; lower = more VRAM used |
| `LLM_UBATCH` / `LLM_BATCH` | 128–512 | Micro-batch; costs VRAM, helps long prefill |
| `LLM_THREADS` | physical cores | Decode threads |
| `LLM_KV_K` / `LLM_KV_V` | `q8_0` / `q8_0` | KV cache quant; `q4_0` V saves VRAM |
| `LLM_COMPACTION_AT` | `209715` | Client threshold hint (80% of 256K) |
| `LLM_MOE_CACHE_MIB` | `1024` | Ornith only: drop to 512/0 on CUDA OOM |
| `LLM_TEMP` / `LLM_TOP_P` / `LLM_TOP_K` | `0.6` / `0.95` / `20` | Sampling |
| `LLM_MODEL` / `LLM_SERVER` / `LLM_PORT` | — | Override GGUF, binary, port |

Qwen launcher only (16 GB RAM machines should keep these off): `LLM_MAX_MEM` (cgroup cap),
`LLM_MLOCK`, `LLM_PROMPT_CACHE`, `LLM_SWAP_CHECK`, `LLM_REASONING_BUDGET`.
Mixed K/V types need `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

Never set OpenCode's model `limit.context` above the server's `LLM_CTX`.

## Compaction

`compaction.auto: false`. The `deferred-compaction` plugin compacts only at a safe point:
execution finished, no tool in flight, last step `stop`, session idle.

- Pending at **80%** of the window; proactive at **70%** after 10 min idle.
- Keeps ~15,000 recent tokens (`compaction.keep.tokens`); the session continues, no `/new`.
- One compaction at a time; manages the local `v100` provider only (`*` = all providers).
- `/compact` still works manually; `/handoff` + `/new` only for a deliberate reset.

See `config/plugins/deferred-compaction/README.md` for the full behaviour and tests.

## Memory safety and warm start

If the machine reboots under load, the OOM killer is at work:

```bash
LLM_MAX_MEM=10G ./scripts/qwen3.6-35b-a3b/llm-up.sh   # cap (qwen launcher)
LLM_CTX=131072 ./scripts/<model>/llm-up.sh            # less KV cache
```

Add 8 GB swap as a safety net, and raise `LLM_NCMOE` if VRAM is short.
`LLM_MLOCK=1` requires `LLM_MAX_MEM=` (empty) and enough RAM for the whole model.

Warm start (qwen launcher): `LLM_PROMPT_CACHE=1` saves the KV cache to disk and restores it on
restart (~5–15 s, ~2–4 GB disk). Verify with `grep "forcing full prompt re-processing" ~/llm-server.log`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available` | Wrong CUDA arch; rebuild with the right `CMAKE_CUDA_ARCHITECTURES` |
| CUDA OOM at load | Raise `LLM_NCMOE`, lower `LLM_CTX`/`LLM_UBATCH`/`LLM_MOE_CACHE_MIB` |
| Low GPU utilisation | Expected: experts are on CPU. Lower `LLM_NCMOE` |
| Every turn re-reads history | Prompt cache disabled or invalidated; do not edit `HANDOFF.md` mid-session |
| A long session stops compacting after a restart | State is in-memory: run `/compact` once on a resumed session near the limit |
| OpenCode request dies after ~5 min | `headerTimeout`/`chunkTimeout` must be `false` |
| Tool calls arrive as plain text | Ensure `--jinja`; re-download the GGUF |
| Port 8080 busy | `./scripts/<model>/llm-stop.sh`, or set `LLM_PORT` and update `baseURL` |
| `hf: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| `Bad Request` / repeated "go on" | Context overflow or damaged history: check `~/llm-server.log`, then `/undo`, `/compact` |
| Config ignored | `opencode --version` must be 2.x |
| Suspend hangs | NVIDIA UVM deadlock: `sudo systemctl mask nvidia-suspend.service` |

## Docker

The image ships only `llama-server`. OpenCode stays on the host and talks to
`http://127.0.0.1:8080/v1`. Models are mounted read-only; nothing is baked in.

```bash
docker build --build-arg CMAKE_CUDA_ARCHITECTURES=70 -t local-llm-opencode:latest .

# default CMD: Qwen3.6-35B-A3B, 256K
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro local-llm-opencode:latest

# Ornith-1.5-35B-A3B (same flags as its launcher)
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/ornith1.5:/models/ornith1.5:ro local-llm-opencode:latest \
  -m /models/ornith1.5/Ornith-1.5-35B-A3B-IQ4_XS.gguf \
  --alias ornith-1.5-35b-a3b-iq4-xs -ngl auto --n-cpu-moe 18 --fit off \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --jinja \
  --reasoning off --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --host 127.0.0.1 --port 8080

docker stop llm-server     # graceful; docker kill to force
```

`LLAMA_REF` is pinned to `b12375` (current release; 77e2db8 is the minimum for Ornith's
SM70 FA kernels). Mixed K/V types need `-DGGML_CUDA_FA_ALL_QUANTS=ON` at build time.
Memory cap on a 16 GB RAM host: add `--memory 12g` to `docker run`. The Ornith launcher
deliberately keeps mmap and applies no cgroup cap; the Qwen launcher defaults to
`LLM_MAX_MEM=12G`.

## Daily workflow

1. Start the server (dedicated terminal), then `opencode` in the project.
2. Work; the agent appends milestones to `.agent/SCRATCH.md`.
3. Compaction happens automatically at a safe point; the session continues.
4. Stop the server with `./scripts/<model>/llm-stop.sh` (never Ctrl+C).

Memory files per project: `.agent/HANDOFF.md` (long-term, auto-loaded, rewrite — max 60 lines)
and `.agent/SCRATCH.md` (journal, read with `tail -n 40`). `.agent/` is git-excluded.

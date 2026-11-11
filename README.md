# Local agentic coding with llama.cpp + OpenCode

**Zero-API-cost coding agent** — runs a local LLM via `llama-server` on your NVIDIA GPU, with OpenCode as the agent interface. A small memory layer (`AGENTS.md`, `HANDOFF.md`, `SCRATCH.md`) means you do not have to re-teach the model after compaction or a new session.

---

## What is in this repo

```
qwen-v100-opencode/
├── Dockerfile                            multi-stage: build llama.cpp + slim runtime
├── README.md                             this file
├── AGENTS.md                             quick reference for agents
├── scripts/
│   ├── 01-install-deps.sh                packages, Hugging Face CLI, OpenCode
│   ├── 02-build-llama-cpp.sh             builds llama.cpp (set CUDA arch via env var)
│   ├── 03-download-model.sh              downloads a model GGUF from HuggingFace
│   ├── 04-install-opencode-config.sh     installs global OpenCode files (with backups)
│   ├── 05-init-project.sh                prepares one project (.agent/ memory files)
│   ├── qwen3.6-35b-a3b/
│   │   ├── llm-up.sh                     starts Qwen3.6-35B-A3B server
│   │   └── llm-stop.sh                   graceful shutdown
│   └── ornith-1.5-35b-a3b/
│       ├── llm-up.sh                     starts Ornith-1.5-35B-A3B server
│       └── llm-stop.sh                   graceful shutdown
└── config/
    ├── opencode.json                     -> ~/.config/opencode/opencode.json
    ├── AGENTS.global.md                  -> ~/.config/opencode/AGENTS.md
    ├── commands/handoff.md               -> ~/.config/opencode/commands/handoff.md
    ├── plugins/better-compaction.ts      -> ~/.config/opencode/plugins/better-compaction.ts
    └── project-templates/                copied into each project by 05-init-project.sh
        ├── opencode.project.json
        ├── AGENTS.project.md
        └── HANDOFF.md
```

---

## 1. Pick your hardware path

The scripts are **hardware-agnostic** — you set a few variables based on your machine.

### Choose your CUDA compute capability

| Your GPU | Set before building |
|---|---|
| RTX 20xx (Turing) | `CMAKE_CUDA_ARCHITECTURES=75` |
| RTX 30xx (Ampere) | `CMAKE_CUDA_ARCHITECTURES=86` |
| RTX 40xx (Ada) | `CMAKE_CUDA_ARCHITECTURES=89` |
| RTX 50xx / H100 | `CMAKE_CUDA_ARCHITECTURES=90` |
| Tesla V100 / Quadro RTX 4000/5000/6000 | `CMAKE_CUDA_ARCHITECTURES=70` |
| Tesla T4 / Quadro T1000/T2000 | `CMAKE_CUDA_ARCHITECTURES=75` |
| GTX 10xx (Pascal) | `CMAKE_CUDA_ARCHITECTURES=61` |
| GTX 9xx / K80 | Not supported (need sm_50+, check llama.cpp docs) |

```bash
CMAKE_CUDA_ARCHITECTURES=70 ./scripts/02-build-llama-cpp.sh
```

### Pick a model

| Model | Quant | VRAM | --n-cpu-moe | KV Cache |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | UD-IQ4_XS | ~17.7 GB | 18 | K=q8_0 V=q8_0 |
| Ornith-1.5-35B-A3B | IQ3_XS | ~14.5 GB | 9 | K=q8_0 V=q4_0 |

Find GGUF files on HuggingFace: search `<model-name>-GGUF`. Popular repos: `bartowski/`, `QuantFactory/`, `unsloth/`.

### System RAM guidelines

| Total RAM | Comfortable with |
|---|---|
| 16 GB | Small models (≤8B), or hybrid models with high `--n-cpu-moe` |
| 32 GB | Medium models (13B–30B) |
| 64 GB+ | Large models (30B–70B) |

---

## 2. Step-by-step setup

### Step 1: Prerequisites

1. **NVIDIA driver** — check: `nvidia-smi` shows your GPU.
2. **CUDA toolkit** — check: `nvcc --version` shows a compatible version:

   | Your GPU (compute capability) | Recommended CUDA |
   |---|---|
   | Tesla V100 / Quadro RTX 4000/5000/6000 (Volta, sm_70) | **CUDA 12.x** (some CUDA 13 builds dropped Volta support) |
   | RTX 20xx / T4 (Turing, sm_75) | CUDA 12.x or 13.x |
   | RTX 30xx / A100 (Ampere, sm_80/86) | CUDA 12.x or 13.x |
   | RTX 40xx (Ada, sm_89) | CUDA 12.x or 13.x |
   | H100 (Hopper, sm_90) | CUDA 12.x or 13.x |
   | GTX 10xx (Pascal, sm_61) | CUDA 12.x only |

   If not found:
    ```bash
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
    ```
3. **Disk:** 25–50 GB free (model + llama.cpp build).
4. **OS:** Debian/Ubuntu assumed (scripts use `apt`). Adjust for your distro.

> **OpenCode 1 vs 2:** OpenCode 2 exists and shares the `opencode` binary. This bundle targets OpenCode 1 config. Check with `opencode --version`.

### Step 2: Install dependencies

```bash
cd qwen-v100-opencode
chmod +x scripts/*.sh
./scripts/01-install-deps.sh
```

Installs build tools, Hugging Face CLI (`hf`), OpenCode, and checks the NVIDIA driver + CUDA 12.x. Skip OpenCode with `SKIP_OPENCODE=1`.

### Step 3: Build llama.cpp

```bash
CMAKE_CUDA_ARCHITECTURES=70 ./scripts/02-build-llama-cpp.sh
```

Clones `llama.cpp` to `~/llama.cpp`, compiles with `-DGGML_CUDA=ON`, and prints which flags your build supports. Pin a version with `LLAMA_REF=<tag-or-commit>`.

**For reasoning-budget support**, use commit `70de1e28e909f3298d777dc9ead1d9718291925c` or newer (PR #20297).

### Step 4: Download a model

```bash
# Qwen3.6-35B-A3B
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF MODEL_QUANT=UD-IQ4_XS ./scripts/03-download-model.sh

# Ornith-1.5-35B-A3B
MODEL_REPO=<your-repo> MODEL_QUANT=IQ3_XS ./scripts/03-download-model.sh
```

Downloads to `~/models/<modelname>/`. The script searches for `*${MODEL_QUANT}*.gguf`. If the filename differs, set `LLM_MODEL=/full/path/to/file.gguf` when starting the server.

### Step 5: Install OpenCode config

```bash
./scripts/04-install-opencode-config.sh
```

Copies `opencode.json`, the global `AGENTS.md`, `/handoff` command, and compaction plugin into `~/.config/opencode/`. Existing files are backed up as `<file>.bak.<timestamp>`.

### Step 6: Start the model server

In a dedicated terminal:

**For Qwen3.6-35B-A3B:**
```bash
./scripts/qwen3.6-35b-a3b/llm-up.sh 2>&1 | tee -a ~/llm-server.log
```

**For Ornith-1.5-35B-A3B:**
```bash
./scripts/ornith-1.5-35b-a3b/llm-up.sh 2>&1 | tee -a ~/llm-server.log
```

Wait for "Starting <model-name>..." Then verify:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/props | python3 -m json.tool | grep n_ctx
```

### Step 7: VS Code

Run `opencode` in your project's terminal. OpenCode installs its VS Code extension automatically. Shortcuts: `Ctrl+Esc` opens it, `Alt+Ctrl+K` inserts file references.

### Step 8: Prepare each project

```bash
./scripts/05-init-project.sh /path/to/your/project
```

Creates `.agent/HANDOFF.md`, `.agent/SCRATCH.md`, `opencode.json` with instructions, and a skeleton `AGENTS.md`. Adds `.agent/` to `.git/info/exclude`.

---

## 3. Tuning for your hardware

All tunables are environment variables on `llm-up.sh`.

### `LLM_NCMOE`: layers in system RAM (hybrid models only)

**Qwen3.6-35B-A3B (UD-IQ4_XS):**
1. Start at **18** (default). This fits ~3.9 GB headroom for KV cache on a V100 16 GB.
2. Watch VRAM: `watch -n1 nvidia-smi --query-gpu=memory.used --format=csv`
3. Step **down** by 2 (16, 14, 12...). Each step moves ~0.37 GB experts onto GPU.
4. Stop at **~1–2 GB VRAM spare**. If it fails to load, go back up.

**Ornith-1.5-35B-A3B (IQ3_XS):**
1. Start at **9** (default). IQ3_XS uses ~59% less system RAM per token than IQ4_XS.
2. For 262142 context with K=q8_0/V=q4_0, 9 layers fits comfortably on 16 GB VRAM.
3. To maximize GPU experts: `LLM_NCMOE=5` (requires ~2 GB more VRAM).

> **Non-MoE models** (Llama, Mistral, etc.): ignore this flag.

### `LLM_CTX`: context window

Bigger context = more VRAM for KV cache, slower but longer conversations.

| Profile | Server | OpenCode limit | Approx. token capacity |
|---|---|---|---|
| Default | `LLM_CTX=262144` | context 262144, input 229376, reserved 98304 | ~130K |
| Faster | `LLM_CTX=131072` | context 131072, input 114688, reserved 49152 | ~90K |

The 128K profile saves ~1.3 GB VRAM. Never set OpenCode's limits above the server's `LLM_CTX`.

### `LLM_THREADS`: decode threads

Auto-detected as physical cores. On a 4-core CPU, try `LLM_THREADS=3` — leaving one core free for the GPU feeder thread sometimes helps.

### `LLM_UBATCH`: micro-batch size

Default 512. With experts on CPU, prefill is sensitive to this. For Ornith-1.5 with long prompts: `LLM_UBATCH=1024` (costs ~500 MiB VRAM, ~1.6 expert layers).

### KV cache types

| Model | Default K/V | Lower VRAM |
|---|---|---|
| Qwen3.6-35B-A3B | q8_0 / q8_0 | q8_0 / q4_0 |
| Ornith-1.5-35B-A3B | q8_0 / q4_0 | q4_0 / q4_0 |

Note: Mixed K/V types need CUDA build flag `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

---

## 4. Memory safety (prevents OOM reboots)

If your system restarts under load, the kernel OOM killer is hitting.

1. **Cap process memory** — `LLM_MAX_MEM=10G ./scripts/qwen3.6-35b-a3b/llm-up.sh` (uses `systemd-run --scope MemoryMax=`). Server exits cleanly instead of crashing. Start low (8G–10G).
2. **Add swap** — safety net so OOM killer doesn't fire:
   ```bash
   sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
3. **Lower `LLM_NCMOE`** — fewer layers in system RAM (MoE models only). Each step down frees ~0.37 GB system RAM.
4. **Lower `LLM_CTX`** — `LLM_CTX=131072` saves ~1.3 GB VRAM.

Check swap: `LLM_SWAP_CHECK=1 ./scripts/qwen3.6-35b-a3b/llm-up.sh`

---

## 4b. Context overflow prevention

"Bad Request" errors are usually caused by the prompt exceeding the context limit. This setup uses a layered approach:

1. **Compaction reserved space** — `opencode.json` sets `reserved: 98304` (96K tokens). Compaction triggers before the prompt gets too large.
2. **File size limits** — The agent rules (in `~/.config/opencode/AGENTS.md`) prevent dumping large files (>2 KB / 100 lines) into context. Always use `grep`, `head`, or `wc` first.
3. **Reasoning budget** — `LLM_REASONING_BUDGET=4096` (default) caps thinking tokens. Set higher for hard problems, lower to prevent budget exhaustion: `LLM_REASONING_BUDGET=2048`.
4. **Always keep a server log** — See Section 5.

Recovery on "Bad Request":
- Check `~/llm-server.log` for the error reason
- `/undo` the last turn (removes damaged history)
- `/compact` (forces context reduction)
- `/handoff` then `/new` (fresh session if needed)

## 5. Server logging

The server logs to stdout/stderr. To capture them:

```bash
./scripts/qwen3.6-35b-a3b/llm-up.sh 2>&1 | tee -a ~/llm-server.log
```

Or set `LLM_LOG_FILE` in the script (defaults to `~/llm-server.log`).

On "Bad Request", check the log:
```bash
grep -nE " 400$|exceed|n_prompt_tokens|prefill|Jinja|parse|error" ~/llm-server.log | tail -20
```

| Log message | Cause | Fix |
|---|---|---|
| `exceeds the available context size` | Context overflow | `/compact` or `/new` |
| `prefill is incompatible with enable_thinking` | OpenCode bug | `/new` (fresh session) |
| `Jinja` or `parse` errors | Damaged history | `/undo` the last turn, then `/compact` |

---

## 6. Warm start / KV cache persistence

**Cold start after idle:** If every turn after an hour of idle re-reads the whole history:

1. **Enable prompt-cache persistence** — saves KV cache to disk, restores on restart (~5–15s cold start vs full re-prefill):
    ```bash
    LLM_PROMPT_CACHE=1 ./scripts/qwen3.6-35b-a3b/llm-up.sh
    ```
    > **Note:** `--prompt-cache` and `--prompt-cache-all` may not be supported by your `llama-server` build. The script validates the flag at startup and warns if unsupported.
2. **Lock in RAM** — `LLM_MLOCK=1` keeps model weights and KV cache in RAM, preventing OS page eviction:
    > **WARNING:** Do NOT use `LLM_MLOCK=1` with `LLM_MAX_MEM` set. The model (~17.7 GB for Qwen, ~14.5 GB for Ornith) exceeds typical caps (8G–12G). Disable the cap first: `LLM_MAX_MEM= LLM_MLOCK=1 ./scripts/qwen3.6-35b-a3b/llm-up.sh`.
3. **Verify:** `grep -n "forcing full prompt re-processing" ~/llm-server.log` — expect no matches.

| Flag | Effect | Trade-off |
|---|---|---|
| `LLM_PROMPT_CACHE=1` | Saves KV cache to disk | ~2–4 GB disk, ~5–15s cold start |
| `LLM_MLOCK=1` | Locks model + cache in RAM | Uses more system RAM, **dangerous with MemoryMax cap** |
| Both | Best warm-start: disk + RAM | Combined cost |

---

## 7. Daily workflow

1. Terminal A: `./scripts/qwen3.6-35b-a3b/llm-up.sh` (or ornith-1.5-35b-a3b)
2. VS Code terminal: `opencode`
3. Work. The agent appends milestones to `.agent/SCRATCH.md`.
4. When done: run `/handoff`, then `/new`. New session loads `HANDOFF.md` automatically.
5. Auto-compaction triggers near the compaction point. Run `/compact` yourself anytime.

### Shutting down

Do NOT Ctrl+C (CUDA cleanup can hang):

```bash
./scripts/qwen3.6-35b-a3b/llm-stop.sh           # graceful
./scripts/ornith-1.5-35b-a3b/llm-stop.sh        # graceful
./scripts/*/llm-stop.sh --force   # immediate kill
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution` | Built wrong CUDA arch. Rebuild with correct `CMAKE_CUDA_ARCHITECTURES`. |
| CUDA out of memory at load | Raise `LLM_NCMOE` (MoE), lower `LLM_CTX`, or lower `LLM_UBATCH`. |
| Quantized KV fails | Use `LLM_KV_V=f16`, then fall back to `LLM_CTX=131072`. |
| Low GPU utilisation | Expected with experts on CPU. Lower `LLM_NCMOE`. |
| Every turn re-reads history | Section 5; keep `prune: false`; do not edit `HANDOFF.md` mid-session. |
| OpenCode request dies after ~5 min | `headerTimeout` / `chunkTimeout` must be `false` in provider options. |
| Tool calls as plain text | Confirm `--jinja` in server command, re-download fresh GGUF. |
| "port 8080 already in use" | `./scripts/*/llm-stop.sh`, or set `LLM_PORT` + update `baseURL`. |
| `hf: command not found` | `export PATH="$HOME/.local/bin:$PATH"`. |
| Config looks ignored | Check `opencode --version`. This bundle targets OpenCode 1. |
| System suspend hangs | NVIDIA UVM deadlock. `sudo systemctl mask nvidia-suspend.service`. |
| "Bad Request" / "go on" repeats | Context overflow or damaged history. Check `~/llm-server.log`. Use `/undo`, then `/compact`, then `/new`. |
| Reasoning budget not working | Update llama.cpp to commit `70de1e28e909f3298d777dc9ead1d9718291925c` or newer. |

---

## 9. Docker

Name your container so you can stop it easily:

```bash
docker build -t local-llm-opencode:latest .
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro \
  local-llm-opencode:latest
```

**Shutting down Docker:**
```bash
docker stop llm-server    # graceful (SIGTERM → 30s → SIGKILL)
docker kill llm-server    # immediate
```

The image is multi-stage: `build` stage compiles llama.cpp with the CUDA arch from the Dockerfile. The `runtime` stage is slim. Model is mounted read-only.

Override flags by appending:
```bash
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro \
  local-llm-opencode:latest \
  --n-cpu-moe 16 -c 131072
```

For your GPU arch, rebuild with the correct arch in the Dockerfile:
```bash
docker build --build-arg CMAKE_CUDA_ARCHITECTURES=70 -t local-llm-opencode:latest .
```

---

## 10. Memory system (per-project)

- `.agent/HANDOFF.md` — long-term memory, auto-loaded. **Rewrite, never append.** Max 60 lines.
- `.agent/SCRATCH.md` — working journal, NOT auto-loaded. Append-only. Read with `tail -n 40`.
- `.agent/` is added to `.git/info/exclude` by `05-init-project.sh`.

Daily flow: work → `/handoff` → `/new`. The new session loads `HANDOFF.md` automatically.

# Local agentic coding on a Tesla V100 16 GB
**llama.cpp + Qwen3.6-35B-A3B + OpenCode (terminal / VS Code)**

A fully local, no-API-cost coding agent. The model runs in `llama-server` on your GPU; OpenCode talks to it over an OpenAI-compatible endpoint. A small memory layer (`AGENTS.md`, `HANDOFF.md`, `SCRATCH.md`) means you do not have to re-teach the model after a compaction or a new session.

---

## 1. Hardware this setup was built and tuned on

| Part | Value | Notes |
|---|---|---|
| GPU | NVIDIA Tesla V100 **16 GB** (Volta, compute capability 7.0) | Desktop card. No bf16, no FP8. Needs CUDA **12.x** (CUDA 13 dropped Volta). |
| CPU | Intel Core i3-10100 | 4 cores / 8 threads, 3.6 GHz base, up to 4.3 GHz, 6 MiB L3, AVX2 (no AVX-512), 1 NUMA node |
| Memory controller (Intel spec for this CPU) | Dual-channel DDR4-2666 max, 41.6 GB/s theoretical | Real bandwidth depends on how many DIMMs you installed. |
| PCIe (Intel spec for this CPU) | PCIe 3.0, up to x16 | Actual link width is not recorded; check it (step 9). |
| System RAM | **not recorded** | Run `free -h`. About 16 GB is enough for this quant; 32 GB is comfortable. |
| OS | Linux desktop, **distro/version not recorded** | The scripts assume Debian/Ubuntu (`apt`). |
| NVIDIA driver / CUDA toolkit | **not recorded** | Record with `nvidia-smi` and `nvcc --version`. |
| OpenCode version | **not recorded** | This bundle was written against OpenCode 1 docs. Record with `opencode --version`. |

What was observed: after switching to `-t 4` (physical cores) and `--n-cpu-moe 18`, generation became noticeably faster. No tokens/s numbers were recorded; there is a table in section 11 to fill in.

---

## 2. What is in this folder

```
qwen-v100-opencode/
├── SETUP.md                              <- this file
├── scripts/
│   ├── 01-install-deps.sh                packages, Hugging Face CLI, OpenCode, CUDA/driver checks
│   ├── 02-build-llama-cpp.sh             builds llama.cpp for the V100 (sm_70)
│   ├── 03-download-model.sh              downloads Qwen3.6-35B-A3B UD-IQ4_XS (~17.7 GB)
│   ├── 04-install-opencode-config.sh     installs the global OpenCode files (with backups)
│   ├── 05-init-project.sh                prepares one project (.agent/ memory files)
│   └── llm-up.sh                         starts the model server
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

## 3. Why it is set up this way

| Choice | Reason |
|---|---|
| **Qwen3.6-35B-A3B**, not a bigger/denser model | Only about 3B parameters are active per token, so it is fast, and only 10 of its 40 layers keep a KV cache (2 KV heads, head dim 256), so long context is cheap: about 20 KiB/token in f16, about 10.6 KiB/token with q8/q8. |
| **UD-IQ4_XS** (17.7 GB) | The Q4_K_M file (22.1 GB) forces far more experts into system RAM. IQ4_XS keeps more on the GPU. |
| **`--n-cpu-moe 18`** | Keeps the experts of 18 of 40 layers in system RAM and the rest on the GPU. Lower = faster, but uses more VRAM. Tune it (section 5). |
| **`-t` = physical cores, `-tb` = logical** | Guidance for partial CPU offload is to use physical cores for decode, not hyperthreads. Generation got faster after this change together with the lower `--n-cpu-moe`; the two effects were not measured separately. |
| **`-fit off`** | So your manual `--n-cpu-moe` / `-c` values are used exactly as written. |
| **`-fa on -ctk q8_0 -ctv q8_0`** | Flash attention plus a quantized KV cache. This combination ran correctly on this V100. |
| **`-np 1`** | One slot. Keeps all VRAM for one long session. |
| **256K server context, compaction near 180K** | The server has about 80K tokens of headroom above the compaction point, so the compaction request itself fits. |
| **`prune: false`** | Pruning edits old tool output in the middle of the history, which (very likely) invalidates the server's prompt cache and costs a long re-prefill. It also has a reported bug where cleared reads led to edits from stale content. |
| **`headerTimeout` / `chunkTimeout` = false** | The default is 5 minutes. A long prefill on this setup can exceed that and abort the request. |
| **`small_model` on a remote free model** | Without it, OpenCode falls back to the main model for titles and would use your single local slot. |
| **HANDOFF.md auto-loaded, SCRATCH.md not** | `HANDOFF.md` is part of the prompt prefix. Editing it forces a full re-prefill, so it is rewritten rarely. The journal (`SCRATCH.md`) is appended to cheaply and read with `tail`. |
| **No MTP model files** | On 16 GB cards the MTP head costs 1-2 GB of VRAM and was a net loss for the 35B model. |

---

## 4. Step-by-step setup on a clean machine

### Step 1: Prerequisites (manual)

1. **NVIDIA driver** that supports the V100. Check: `nvidia-smi` shows the Tesla V100.
2. **CUDA toolkit 12.x**, not 13 (CUDA 13 dropped Volta). Check: `nvcc --version` says `release 12.x`. If `nvcc` is not found but `/usr/local/cuda/bin` exists, add it to your `PATH`:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   ```
3. **Disk:** about 25 GB free for the model, plus a few GB for the llama.cpp build.
4. **Do not use Ollama or LM Studio** for this setup. Newer Ollama releases dropped Volta and LM Studio does not support it. Use `llama-server` from the source build below.

### Step 2: Install dependencies

```bash
cd qwen-v100-opencode
chmod +x scripts/*.sh
./scripts/01-install-deps.sh
```
It installs build tools, the Hugging Face CLI (`hf`), and OpenCode, and it stops with a clear message if the driver or CUDA 12.x is missing. Use `SKIP_OPENCODE=1 ./scripts/01-install-deps.sh` to skip OpenCode.

> **OpenCode 1 vs 2:** OpenCode 2 now exists and shares the `opencode` command. The install script above installs OpenCode 1. The config in this bundle follows the OpenCode 1 docs; it has not been checked against OpenCode 2. If you use Homebrew, use `brew install anomalyco/tap/opencode` for V1.

### Step 3: Build llama.cpp for the V100

```bash
./scripts/02-build-llama-cpp.sh
```
This clones `llama.cpp` to `~/llama.cpp`, compiles with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`, and prints which tuning flags your build supports. Prebuilt binaries do not contain Volta kernels, so a source build is required. Pin a version with `LLAMA_REF=<tag-or-commit> ./scripts/02-build-llama-cpp.sh`.

### Step 4: Download the model

```bash
./scripts/03-download-model.sh
```
Downloads `UD-IQ4_XS` (about 17.7 GB) to `~/models/qwen3.6/`. Always use a fresh download: Unsloth updated the chat template for developer-role support, which OpenCode needs.

### Step 5: Install the OpenCode files

```bash
./scripts/04-install-opencode-config.sh
```
Copies `opencode.json`, the global `AGENTS.md`, the `/handoff` command and the compaction plugin into `~/.config/opencode/`. Any existing file that differs is backed up first as `<file>.bak.<timestamp>`.

### Step 6: Start the model server

In a dedicated terminal:

```bash
./scripts/llm-up.sh 2>&1 | tee -a ~/llm-server.log
```
Wait for the "server is listening" line. Then, in another terminal:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/props | python3 -m json.tool | grep n_ctx      # expect 262144
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":200}'
```

### Step 7: VS Code

Open VS Code's integrated terminal in your project and run `opencode`. OpenCode installs its VS Code extension automatically when it runs in that terminal. Useful shortcuts: `Ctrl+Esc` opens it, `Alt+Ctrl+K` inserts file references, and your current selection is shared automatically.

### Step 8: Prepare each project (once per project)

```bash
./scripts/05-init-project.sh /path/to/your/project
```
This creates, without overwriting anything that exists:
- `.agent/HANDOFF.md` (long-term memory, auto-loaded)
- `.agent/SCRATCH.md` (working journal, not auto-loaded)
- `opencode.json` with `"instructions": [".agent/HANDOFF.md"]`
- an `AGENTS.md` skeleton (only if you have none)

It also adds `.agent/` to `.git/info/exclude`, so your notes stay out of git. Then run `opencode` in the project, run `/init` once, and fill in the run/test commands and architecture notes in `AGENTS.md`. Commit that file.

### Step 9: Verify (do all four once)

1. **VRAM:** with the server running and a long prompt processed, `nvidia-smi` should stay below about 15 GiB. If it runs out of memory at load, raise `LLM_NCMOE` or lower `LLM_CTX`.
2. **Speed:** read the timing lines in `~/llm-server.log` (`prompt eval time` and `eval time`) and record them in section 11.
3. **Hardware links** (read once while the model is busy; idle cards downshift):
   ```bash
   nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv   # want gen 3, width 16
   sudo dmidecode -t memory | grep -E "Size|Speed|Locator"                             # want 2 DIMMs populated
   grep MHz /proc/cpuinfo                                                              # cores should reach ~4 GHz under load
   ```
4. **Cache reuse (the important one):** hold a long conversation in OpenCode, send one more short message, then run:
   ```bash
   grep -n "forcing full prompt re-processing" ~/llm-server.log
   ```
   You want **no matches**, and the second request should process only the few new tokens ("prompt eval" with a small token count). If it reprocesses everything, see troubleshooting.

---

## 5. Tuning guide

All tunables are environment variables on `llm-up.sh` (full list in the header of that file), for example `LLM_NCMOE=16 ./scripts/llm-up.sh`.

### 5.1 `LLM_NCMOE`: the biggest lever

The GPU sits mostly idle when too many experts stay on the CPU; that is expected, not a bug. Move layers to the GPU until VRAM is nearly full:

1. Start at **18**.
2. Watch VRAM: `watch -n1 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`
3. Step down by 2 (16, 14, 12 ...). Each step moves roughly 0.37 GB of experts onto the GPU (an estimate for this IQ4_XS file).
4. Stop at about **14.8-15 GiB used**, leaving about 1 GB spare. If it fails to load, go back up one step.

### 5.2 Context size trade-off

The KV cache is preallocated, so a big context permanently costs VRAM: about 2.7 GiB at 262144 with q8/q8 (estimate). If your sessions rarely pass about 100K tokens, use the faster profile:

| Profile | Server | `opencode.json` `limit` / `reserved` | Compaction point (approx.) |
|---|---|---|---|
| **Default (256K)** | `LLM_CTX=262144` | context 262144, input 229376, output 32768, reserved 49152 | about 180K tokens |
| **Faster (128K)** | `LLM_CTX=131072 LLM_NCMOE=15` | context 131072, input 114688, output 32768, reserved 24576 | about 90K tokens |

The compaction point is my reading of the settings (`input` minus `reserved`). The 128K profile frees about 1.3 GiB, which lets you lower `LLM_NCMOE` by roughly 3. Never set OpenCode's limits above the server's `LLM_CTX`.

Choose the compaction point from your own measured prefill speed: tokens divided by prefill tok/s is how long a full re-read takes. For example, 180K tokens at 350 tok/s is about 8.5 minutes.

### 5.3 Threads

`llm-up.sh` auto-detects physical cores for `-t` and logical CPUs for `-tb`. On the i3-10100 that is `-t 4 -tb 8`. Try `LLM_THREADS=3` too: leaving one core free for the thread that feeds the GPU sometimes helps on a 4-core CPU. Compare `eval time` tok/s on the same prompt.

### 5.4 Batch size

With experts on the CPU, prefill is sensitive to `-ub`. Only after the offload split is tuned and VRAM remains, test `LLM_UBATCH=1024`. It costs VRAM.

### 5.5 Prompt-cache checkpoints (only if step 9.4 fails)

Qwen3.6 is a hybrid model and llama-server has had widely reported full-prompt re-processing with it. Flag names differ between builds, and the script detects which ones exist:

```bash
LLM_CHECKPOINTS=32 LLM_CHECKPOINT_STEP=2048 ./scripts/llm-up.sh 2>&1 | tee -a ~/llm-server.log
```
Each checkpoint uses some system RAM (roughly 60 MiB each for this model, an estimate), so leave RAM headroom. Also update llama.cpp (`./scripts/02-build-llama-cpp.sh`).

### 5.6 Optional experiments (A/B test each, one at a time)

| Setting | What it does | Trade-off |
|---|---|---|
| `LLM_PRESERVE_THINKING=1` | Qwen recommends it for agent use; may improve cache reuse across turns. Must be set on the server, because OpenCode drops client-side `chat_template_kwargs`. | Keeps reasoning in history, so it uses more context. |
| `LLM_REASONING_BUDGET=4096` | Caps thinking tokens when replies are slow because of very long reasoning. | Lower quality on hard problems. Keep `output` in `opencode.json` above the budget plus your largest tool call. |

Not recommended: the `-MTP` GGUF variants (net loss on 16 GB).

---

## 6. Daily workflow

1. Terminal A: `./scripts/llm-up.sh 2>&1 | tee -a ~/llm-server.log`
2. VS Code terminal in the project: `opencode`
3. Work normally. The agent appends milestones to `.agent/SCRATCH.md` as it goes.
4. When the context gets heavy, or you are done for the day: run `/handoff`, then `/new`. The new session loads `HANDOFF.md` automatically and reads the tail of `SCRATCH.md` itself, so there is no re-teaching. Run `/handoff` right before `/new`, because rewriting `HANDOFF.md` invalidates the cache anyway.
5. If you do nothing, auto-compaction triggers near the compaction point in section 5.2. You can also run `/compact` yourself before then.
6. For long, noisy investigations, ask the agent to keep command output filtered (`tail`, `grep`). Pruning is off, so everything printed stays in context.

---

## 7. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `no kernel image is available for execution` | llama.cpp was built without sm_70. Rebuild with `scripts/02-build-llama-cpp.sh`. |
| CUDA out of memory at load | Raise `LLM_NCMOE`, lower `LLM_CTX`, or lower `LLM_UBATCH`. |
| `-fa on` errors, or quantized KV fails | Use `LLM_KV_V=f16`, then fall back to `LLM_CTX=131072`. |
| Low GPU utilisation | Expected while many experts run on the CPU. Lower `LLM_NCMOE` (section 5.1). |
| Slow generation on every turn | Check `-t` equals physical cores (section 5.3); check RAM is dual-channel and the GPU link is x16 (step 9.3). |
| Every turn re-reads the whole history | Section 5.5; keep `prune: false`; do not edit `HANDOFF.md` mid-session. |
| OpenCode request dies after about 5 minutes | `headerTimeout` / `chunkTimeout` must be `false` in the `v100` provider options. |
| Tool calls show up as plain text, or tools never run | Confirm `--jinja` is in the server command, re-download a fresh GGUF, and update llama.cpp. Local models in OpenCode have known tool-call issues. |
| "port 8080 already in use" | Another `llama-server` is running. Stop it, or set `LLM_PORT` (and update `baseURL` in `opencode.json`). |
| `hf: command not found` | Open a new shell, or `export PATH="$HOME/.local/bin:$PATH"`. |
| Config looks ignored after an upgrade | Check `opencode --version`. This bundle targets OpenCode 1. |
| Compaction summary looks generic | The plugin hook is experimental; check that `plugins/better-compaction.ts` is installed and re-test with `/compact`. Rely on `SCRATCH.md` and `/handoff` regardless. |

---

## 8. Caveats and what is not verified

- These files were assembled from documentation, issue reports and benchmarks. The scripts were syntax-checked, and `llm-up.sh`, `04-install-opencode-config.sh` and `05-init-project.sh` were dry-run against a stub server and temporary directories. They were **not run on your machine** by the author of this guide, so the build, download and GPU steps are untested here. Treat flag values as a starting point.
- `--n-cpu-moe 18` and the VRAM and RAM figures in this guide are estimates from file sizes and the model's published architecture.
- Cache reuse on this hybrid model is a known weak spot in llama.cpp; verify it (step 9.4).
- The compaction plugin uses an experimental hook, and one user reported it not taking effect on a later version.
- OpenCode 2 compatibility is unchecked.
- The `small_model` line uses an OpenCode Zen free model, and **Zen's free model IDs rotate** (sources disagree on the current one). Check the live list, which needs no key:
  ```bash
  curl -s https://opencode.ai/zen/v1/models | python3 -m json.tool | grep '"id"' | grep -i free
  ```
  Then set `"small_model": "opencode/<id>"` in `~/.config/opencode/opencode.json`. If you delete the line instead, OpenCode falls back to the main model for titles, which uses your single local slot (likely disturbing the cache).
- Free third-party endpoints may use your prompts for training. Everything else here stays on your machine.

---

## 9. Maintenance

- **Update llama.cpp:** `./scripts/02-build-llama-cpp.sh` (pulls and rebuilds). Re-run step 9 afterwards.
- **Update the model:** re-run `./scripts/03-download-model.sh`; it fetches the latest files for the quant.
- **Re-install config:** `./scripts/04-install-opencode-config.sh` is safe to repeat; differing files are backed up first.
- **Record changes** to the hardware or versions in section 1 so the setup stays reproducible.

---

## 10. Sources consulted

- Qwen3.6-35B-A3B model card and GGUF sizes: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- OpenCode docs: https://opencode.ai/docs/ , https://opencode.ai/docs/config/ , https://opencode.ai/docs/rules/
- OpenCode compaction prompt and prune issues: https://github.com/anomalyco/opencode/issues/16512 , https://github.com/anomalyco/opencode/issues/59321
- llama.cpp hybrid-model cache re-processing: https://github.com/ggml-org/llama.cpp/issues/22746 , https://github.com/ggml-org/llama.cpp/issues/21831 , https://github.com/ggml-org/llama.cpp/discussions/29885
- MoE offload tuning: https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0 , https://docs.tokios.com/guides/llama-cpp-optimization
- Tesla V100 + Qwen3.6 + llama.cpp: https://miyagadget.page/en/blog/2026/07/15/tesla-v100-32gb-qwen36-benchmark-en/ , https://github.com/fishlikeX/sm70-attn
- 16 GB GPU benchmarks: https://www.glukhov.org/llm-performance/benchmarks/best-llm-on-16gb-vram-gpu/ , https://www.glukhov.org/llm-performance/benchmarks/comparing-qwen-3-6-mtp-vs-standard/
- NVIDIA llama.cpp playbook: https://build.nvidia.com/playbooks/llama-cpp/instructions.md

---

## 11. Your measurements (fill in)

| Item | Value |
|---|---|
| Installed RAM / channels / speed | |
| NVIDIA driver / CUDA version | |
| OpenCode version | |
| llama.cpp commit | |
| `LLM_NCMOE` used | |
| VRAM used at load | |
| Prefill speed (tok/s, long prompt) | |
| Decode speed (tok/s) | |
| Cache reuse OK (step 9.4)? | |

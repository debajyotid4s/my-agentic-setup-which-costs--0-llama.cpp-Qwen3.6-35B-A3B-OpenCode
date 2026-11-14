# syntax=docker/dockerfile:1
#
# llama.cpp server image (OpenCode itself runs on the host and talks to it over
# http://127.0.0.1:8080/v1). Pinned to a llama.cpp release that supports the
# flags the launchers use, including SM70 Flash-Attention for Ornith-1.5.
#
# Build (pick your GPU arch):
#   docker build --build-arg CMAKE_CUDA_ARCHITECTURES=70 -t local-llm-opencode:latest .
#
# Run (requires NVIDIA Container Toolkit). Qwen3.6-35B-A3B is the default CMD:
#   docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
#     -v ~/models/qwen3.6:/models/qwen3.6:ro \
#     local-llm-opencode:latest
#
# Ornith-1.5-35B-A3B (IQ4_XS, 256K, same flags as scripts/ornith-1.5-35b-a3b/llm-up.sh):
#   docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
#     -v ~/models/ornith1.5:/models/ornith1.5:ro \
#     local-llm-opencode:latest \
#     -m /models/ornith1.5/Ornith-1.5-35B-A3B-IQ4_XS.gguf \
#     --alias ornith-1.5-35b-a3b-iq4-xs -ngl auto --n-cpu-moe 18 --fit off \
#     -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --jinja \
#     --reasoning off --chat-template-kwargs '{"enable_thinking":false}' \
#     --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
#     --host 127.0.0.1 --port 8080
#
# The model is mounted from the host so the image stays small.
# Download it first: scripts/03-download-model.sh
#
# Mixed K/V cache types (e.g. q8_0/q4_0) require -DGGML_CUDA_FA_ALL_QUANTS=ON
# in the build stage; the default carries only a few predefined quant pairs
# (q4_0/q4_0, q8_0/q8_0, f16, bf16).

ARG BASE=nvidia/cuda:12.9.0-devel-ubuntu24.04
ARG RUNTIME=nvidia/cuda:12.9.0-runtime-ubuntu24.04
# Current llama.cpp release tag. Keep this at or above commit 77e2db8 for
# Ornith-1.5 (SM70 FA for Dk=256) and 70de1e28 for --reasoning-budget.
ARG LLAMA_REF=b12375
ARG CMAKE_CUDA_ARCHITECTURES=70

# ── Build stage ──────────────────────────────────────────────────────────────
FROM ${BASE} AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl wget pkg-config \
      python3 python3-pip pciutils \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp for the specified CUDA architecture, pinned to LLAMA_REF
RUN git clone --depth 1 --single-branch --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /llama.cpp
RUN cd /llama.cpp && \
    cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
      -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/llama-server

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM ${RUNTIME}

LABEL org.opencontainers.image.description="Local LLM server with llama.cpp + OpenCode"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/llama-server /usr/local/bin/llama-server

# Model directories (mounted from host at runtime)
# Qwen3.6-35B-A3B: ~/models/qwen3.6/
# Ornith-1.5-35B-A3B: ~/models/ornith1.5/
RUN mkdir -p /models/qwen3.6 /models/ornith1.5

# Prompt cache directory (mounted from host for warm-start persistence)
RUN mkdir -p /llama-cache
VOLUME ["/llama-cache"]

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llama-server"]

# Default: Qwen3.6-35B-A3B (UD-IQ4_XS) at 256K, matched to
# scripts/qwen3.6-35b-a3b/llm-up.sh. Append flags after the image name to
# override anything; see the header for the Ornith-1.5 command.
# Warm-start flags: --timeout 0 disables idle KV cache eviction.
# For prompt-cache persistence, mount /llama-cache and add:
#   --prompt-cache /llama-cache/qwen3.6-35b-a3b-kvcache.bin --prompt-cache-all
CMD ["-m", "/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "--alias", "qwen3.6-35b-a3b", "-ngl", "99", "--n-cpu-moe", "18", "-fit", "off", "-c", "262144", "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512", "--jinja", "--chat-template-kwargs", "{\"preserve_thinking\": true}", "--timeout", "0", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0", "--presence-penalty", "0.5", "--reasoning-budget", "4096", "--reasoning-budget-message", "Reasoning budget reached. Stop analysing; run one concrete command or test for your leading hypothesis, then continue.", "--host", "127.0.0.1", "--port", "8080"]

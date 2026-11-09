# syntax=docker/dockerfile:1
#
# Build:
#   docker build -t local-llm-opencode:latest .
#
# Run (requires NVIDIA Container Toolkit):
#   docker run --rm -it --gpus all -p 127.0.0.1:8080:8080 \
#     -v ~/models/qwen3.6:/models/qwen3.6:ro \
#     local-llm-opencode:latest
#
# For your GPU arch, rebuild with the correct arch:
#   docker build --build-arg CMAKE_CUDA_ARCHITECTURES=86 -t local-llm-opencode:latest .
#
# Override server flags by appending to docker run:
#   docker run --rm -it --gpus all -p 127.0.0.1:8080:8080 \
#     -v ~/models/qwen3.6:/models/qwen3.6:ro \
#     local-llm-opencode:latest \
#     --n-cpu-moe 16 -c 131072
#
# The model is mounted from the host so the image stays small.
# Download it first: scripts/03-download-model.sh

ARG BASE=nvidia/cuda:12.9.0-devel-ubuntu24.04
ARG RUNTIME=nvidia/cuda:12.9.0-runtime-ubuntu24.04
ARG LLAMA_REF=v9061
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

# Model lives at /models/qwen3.6 (mounted from host at runtime)
RUN mkdir -p /models/qwen3.6

# Prompt cache directory (mounted from host for warm-start persistence)
RUN mkdir -p /llama-cache
VOLUME ["/llama-cache"]

EXPOSE 127.0.0.1:8080

ENTRYPOINT ["/usr/local/bin/llama-server"]
# Warm-start flags: --timeout 0 disables idle KV cache eviction.
# For prompt-cache persistence, mount /llama-cache and add:
#   --prompt-cache /llama-cache/qwen3.6-35b-a3b-kvcache.bin --prompt-cache-all
# (or run via llm-up.sh with LLM_PROMPT_CACHE=1)
CMD ["-m", "/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "--alias", "qwen3.6-35b-a3b", "-ngl", "99", "--n-cpu-moe", "18", "-fit", "off", "-c", "262144", "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512", "--jinja", "--chat-template-kwargs", "{\"preserve_thinking\": true}", "--timeout", "0", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0", "--presence-penalty", "0.0", "--host", "127.0.0.1", "--port", "8080"]

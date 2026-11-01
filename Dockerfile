# syntax=docker/dockerfile:1
#
# Build:
#   docker build -t qwen-v100-opencode:latest .
#
# Run (requires NVIDIA Container Toolkit):
#   docker run --rm -it --gpus all -p 8080:8080 \
#     -v ~/models/qwen3.6:/models/qwen3.6:ro \
#     qwen-v100-opencode:latest
#
# Override flags by appending to docker run:
#   docker run --rm -it --gpus all -p 8080:8080 \
#     -v ~/models/qwen3.6:/models/qwen3.6:ro \
#     qwen-v100-opencode:latest \
#     --n-cpu-moe 16 -c 131072
#
# The model is mounted from the host so the image stays small.
# Download it first: scripts/03-download-model.sh

ARG BASE=nvidia/cuda:12.9.0-devel-ubuntu24.04
ARG RUNTIME=nvidia/cuda:12.9.0-runtime-ubuntu24.04

# ── Build stage ──────────────────────────────────────────────────────────────
FROM ${BASE} AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl wget pkg-config \
      python3 python3-pip pciutils \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp for Volta (sm_70)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp
RUN cd /llama.cpp && \
    cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/llama-server

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM ${RUNTIME}

LABEL org.opencontainers.image.description="Qwen3.6-35B-A3B + OpenCode on Tesla V100 (Volta / sm_70)"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/llama-server /usr/local/bin/llama-server

# Model lives at /models/qwen3.6 (mounted from host at runtime)
RUN mkdir -p /models/qwen3.6

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llama-server"]
CMD ["-m", "/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "--alias", "qwen3.6-35b-a3b", "-ngl", "99", "--n-cpu-moe", "18", "-fit", "off", "-c", "262144", "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512", "--jinja", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0", "--presence-penalty", "0.0", "--host", "0.0.0.0", "--port", "8080"]

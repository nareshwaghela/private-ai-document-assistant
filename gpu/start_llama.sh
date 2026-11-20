#!/usr/bin/env bash
set -euo pipefail
# Run on Linux with NVIDIA GPU, installed NVIDIA Container Toolkit and Docker.
# Keeps the llama.cpp API reachable ONLY on the local machine's 127.0.0.1.
docker run --rm --gpus all \
  -p 127.0.0.1:8080:8080 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
  --alias local --n-gpu-layers all \
  --ctx-size 4096 --parallel 1 \
  --batch-size 512 --ubatch-size 128 \
  --flash-attn off --cache-type-k f16 --cache-type-v f16 \
  --fit off --no-context-shift \
  --host 0.0.0.0 --port 8080

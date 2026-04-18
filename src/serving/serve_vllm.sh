#!/usr/bin/env bash
# =============================================================================
# Football-LLM: Launch vLLM OpenAI-compatible API server
#
# Serves Llama 3.1 8B Instruct + QLoRA adapter with 4-bit quantization.
# Optimized for T4 GPU (16 GB VRAM).
#
# Usage:
#   chmod +x src/serving/serve_vllm.sh
#   ./src/serving/serve_vllm.sh
#
# The server exposes an OpenAI-compatible API at http://localhost:8000
# =============================================================================

set -euo pipefail

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID="zanwenfu/football-llm-qlora"
PORT=8000
MAX_MODEL_LEN=768
GPU_MEM_UTIL=0.9

echo "=============================================="
echo "  Football-LLM — vLLM Serving"
echo "=============================================="
echo "  Base model:  ${MODEL_ID}"
echo "  LoRA adapter: ${ADAPTER_ID}"
echo "  Port:         ${PORT}"
echo "  Max seq len:  ${MAX_MODEL_LEN}"
echo "  GPU mem util: ${GPU_MEM_UTIL}"
echo "  Quantization: 4-bit (bitsandbytes)"
echo "=============================================="

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --enable-lora \
    --lora-modules football-llm="${ADAPTER_ID}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype float16 \
    --max-lora-rank 16 \
    --trust-remote-code

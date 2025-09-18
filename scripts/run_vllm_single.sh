#!/bin/bash

echo "🚀 Starting vLLM Server (Single Model with Multiple Names)"
echo "========================================================="

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null

# Create logs directory
mkdir -p logs

# vLLM doesn't support multiple --model flags in a single process
# We need to either:
# 1. Run multiple vLLM instances (one per model) on different ports
# 2. Use a model routing proxy
# 3. Run one model and route requests

echo ""
echo "⚠️  IMPORTANT: vLLM limitation"
echo "--------------------------------"
echo "vLLM can only serve ONE model per process."
echo ""
echo "Options:"
echo "1. Run 3 separate vLLM servers (requires ~24GB VRAM total)"
echo "2. Run 1 model and use it for all requests"
echo "3. Use model switching with restart (slow)"
echo ""

# Option 1: Run single model for all requests
echo "📦 Starting single model server (Qwen2.5-3B)..."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --port 8000 \
    --host 0.0.0.0 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen2p5_3b \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --trust-remote-code \
    2>&1 | tee logs/vllm_single.log
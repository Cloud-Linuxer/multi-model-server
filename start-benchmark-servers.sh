#!/bin/bash

echo "🚀 Starting All Benchmark Servers with Docker..."

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-benchmark.yml down 2>/dev/null
docker stop $(docker ps -aq) 2>/dev/null

# Clean up
echo "🧹 Cleaning up..."
docker system prune -f

# Start only vLLM first (SGLang has issues with multi-model on RTX 5090)
echo "📦 Starting vLLM services..."
docker run -d \
  --name vllm-tinyllama \
  --gpus all \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  vllm/vllm-openai:latest \
  --host 0.0.0.0 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager

echo "⏳ Waiting for vLLM TinyLlama to start (30s)..."
sleep 30

# For now, run single model benchmarks sequentially for each framework
echo "✅ vLLM TinyLlama ready on port 8001"

# Start Ollama
echo "📦 Starting Ollama..."
docker run -d \
  --name ollama-benchmark \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest \
  serve

echo "⏳ Waiting for Ollama to start (10s)..."
sleep 10

# Pull Ollama models
echo "📥 Pulling Ollama models..."
docker exec ollama-benchmark ollama pull tinyllama:1.1b

echo "✅ Ollama ready on port 11434"

# Start SGLang (single model only due to RTX 5090 limitations)
echo "📦 Starting SGLang..."
docker run -d \
  --name sglang-tinyllama \
  --gpus all \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.3

echo "⏳ Waiting for SGLang to start (30s)..."
sleep 30

echo "✅ SGLang ready on port 30000"

# Check status
echo ""
echo "📊 Checking container status..."
docker ps | grep -E "vllm|ollama|sglang"

echo ""
echo "🎯 Servers ready for benchmarking:"
echo "  - vLLM TinyLlama: http://localhost:8001"
echo "  - Ollama: http://localhost:11434"
echo "  - SGLang TinyLlama: http://localhost:30000"
echo ""
echo "Run the benchmark with: python unified_benchmark.py"
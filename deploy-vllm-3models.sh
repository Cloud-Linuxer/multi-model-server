#!/bin/bash

echo "🚀 vLLM 3개 모델 배포 시작"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop vllm-tinyllama vllm-qwen vllm-yi 2>/dev/null
docker rm vllm-tinyllama vllm-qwen vllm-yi 2>/dev/null

# GPU 상태 확인
echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# 1. TinyLlama 배포
echo ""
echo "1️⃣ TinyLlama 1.1B 배포..."
docker run -d \
  --name vllm-tinyllama \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 16g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 1024 \
  --enforce-eager

echo "⏳ TinyLlama 초기화 중 (30초)..."
sleep 30

# 2. Qwen 2.5-3B 배포
echo ""
echo "2️⃣ Qwen 2.5-3B 배포..."
docker run -d \
  --name vllm-qwen \
  --runtime nvidia \
  --gpus all \
  -p 40002:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 24g \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 1536 \
  --enforce-eager

echo "⏳ Qwen 초기화 중 (40초)..."
sleep 40

# 3. Yi-6B 배포
echo ""
echo "3️⃣ Yi-6B 배포..."
docker run -d \
  --name vllm-yi \
  --runtime nvidia \
  --gpus all \
  -p 40003:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 32g \
  vllm/vllm-openai:latest \
  --model 01-ai/Yi-6B-Chat \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 2048 \
  --enforce-eager

echo "⏳ Yi 초기화 중 (50초)..."
sleep 50

# 최종 상태 확인
echo ""
echo "✅ vLLM 배포 완료!"
echo ""
echo "📊 최종 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
echo "🐳 실행 중인 컨테이너:"
docker ps | grep vllm

echo ""
echo "📡 API 엔드포인트:"
echo "  - TinyLlama: http://localhost:40001/v1/completions"
echo "  - Qwen 2.5-3B: http://localhost:40002/v1/completions"
echo "  - Yi-6B: http://localhost:40003/v1/completions"

echo ""
echo "💡 메모리 할당 전략 (vLLM):"
echo "  - TinyLlama: 15% utilization"
echo "  - Qwen: 25% utilization"
echo "  - Yi: 45% utilization"
echo "  - 총 사용: 85% of available GPU memory"
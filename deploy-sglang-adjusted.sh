#!/bin/bash

echo "🚀 SGLang 3개 모델 조정된 메모리 배포"
echo "전략: 각 모델 크기보다 충분히 큰 메모리 할당"
echo ""

# 기존 컨테이너 정리
docker stop sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null
docker rm sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null

echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 1. TinyLlama (2.15GB 모델 → 10% = 3.2GB)
echo ""
echo "1️⃣ TinyLlama 배포 (mem: 0.10)..."
docker run -d \
  --name sglang-tinyllama \
  --runtime nvidia \
  --gpus all \
  -p 30001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --shm-size 8g \
  sglang:blackwell-final-v2 \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.10 \
  --max-total-tokens 1024 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

sleep 40
echo "TinyLlama 상태:"
docker ps | grep sglang-tinyllama || echo "❌ TinyLlama 실패"
nvidia-smi --query-gpu=memory.used --format=csv

# 2. Qwen (5.88GB 모델 → 최소 20% = 6.5GB)
echo ""
echo "2️⃣ Qwen 배포 (mem: 0.20)..."
docker run -d \
  --name sglang-qwen \
  --runtime nvidia \
  --gpus all \
  -p 30002:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --shm-size 16g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.20 \
  --max-total-tokens 1024 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

sleep 50
echo "Qwen 상태:"
docker ps | grep sglang-qwen || echo "❌ Qwen 실패"
nvidia-smi --query-gpu=memory.used --format=csv

# 3. Yi (12GB 모델 → 최소 40% = 13GB)
echo ""
echo "3️⃣ Yi 배포 (mem: 0.40)..."
docker run -d \
  --name sglang-yi \
  --runtime nvidia \
  --gpus all \
  -p 30003:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --shm-size 24g \
  sglang:blackwell-final-v2 \
  --model-path 01-ai/Yi-6B-Chat \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.40 \
  --max-total-tokens 1024 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

sleep 60
echo "Yi 상태:"
docker ps | grep sglang-yi || echo "❌ Yi 실패"

echo ""
echo "📊 최종 상태:"
docker ps | grep sglang
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "📝 메모리 할당:"
echo "  TinyLlama: 10% (3.2GB)"
echo "  Qwen: 20% (6.5GB)"
echo "  Yi: 40% (13GB)"
echo "  총합: 70% (22.7GB/32GB)"
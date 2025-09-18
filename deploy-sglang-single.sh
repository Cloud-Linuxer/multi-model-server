#!/bin/bash

echo "🚀 SGLang 단일 모델 배포 (멀티모델 불가능)"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop sglang-tinyllama 2>/dev/null
docker rm sglang-tinyllama 2>/dev/null

# GPU 상태 확인
echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# TinyLlama만 배포 (SGLang은 한 번에 하나만 가능)
echo ""
echo "1️⃣ TinyLlama 1.1B 배포 (SGLang은 단일 모델만 지원)..."
docker run -d \
  --name sglang-tinyllama \
  --runtime nvidia \
  --gpus all \
  -p 30001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --shm-size 16g \
  sglang:blackwell-final-v2 \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 2048 \
  --max-running-requests 256 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --log-level info

echo "⏳ SGLang 초기화 중 (40초)..."
sleep 40

# 상태 확인
echo ""
echo "✅ SGLang 배포 완료!"
echo ""
echo "📊 최종 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
echo "🐳 실행 중인 컨테이너:"
docker ps | grep sglang

echo ""
echo "📡 API 엔드포인트:"
echo "  - SGLang TinyLlama: http://localhost:30001/generate"

echo ""
echo "⚠️  SGLang 제한사항 (RTX 5090):"
echo "  - 단일 모델만 실행 가능 (멀티모델 불가)"
echo "  - RadixAttention 비활성화됨 (RTX 5090 미지원)"
echo "  - 여러 최적화 기능 비활성화됨"
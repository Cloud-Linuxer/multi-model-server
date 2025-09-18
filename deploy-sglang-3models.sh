#!/bin/bash

# SGLang 3개 모델 동시 배포 스크립트
# RTX 5090 최적화 설정

echo "🚀 SGLang 3개 모델 배포 시작..."

# 기존 컨테이너 정리
echo "📦 기존 컨테이너 정리 중..."
docker stop sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null
docker rm sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null

echo ""
echo "1️⃣ TinyLlama 1.1B 시작..."
docker run -d \
  --name sglang-tinyllama \
  --runtime nvidia \
  --gpus all \
  -p 30001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_VISIBLE_DEVICES=0 \
  --shm-size 8g \
  sglang:blackwell-final-v2 \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.12 \
  --max-total-tokens 2048 \
  --max-running-requests 48 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --dtype float16 \
  --log-level info

echo "⏳ TinyLlama 초기화 대기 (15초)..."
sleep 15

echo ""
echo "2️⃣ Qwen 2.5-3B 시작..."
docker run -d \
  --name sglang-qwen \
  --runtime nvidia \
  --gpus all \
  -p 30002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_VISIBLE_DEVICES=0 \
  --shm-size 16g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.20 \
  --max-total-tokens 3072 \
  --max-running-requests 32 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --dtype float16 \
  --log-level info

echo "⏳ Qwen 초기화 대기 (20초)..."
sleep 20

echo ""
echo "3️⃣ Yi-6B 시작..."
docker run -d \
  --name sglang-yi \
  --runtime nvidia \
  --gpus all \
  -p 30003:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_VISIBLE_DEVICES=0 \
  --shm-size 24g \
  sglang:blackwell-final-v2 \
  --model-path 01-ai/Yi-6B-Chat \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.40 \
  --max-total-tokens 4096 \
  --max-running-requests 24 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --dtype float16 \
  --log-level info

echo "⏳ Yi 초기화 대기 (30초)..."
sleep 30

echo ""
echo "✅ 컨테이너 상태 확인..."
docker ps | grep sglang

echo ""
echo "📊 메모리 사용량 확인..."
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo ""
echo "🔍 서버 상태 확인..."
echo "TinyLlama (30001):"
curl -s http://localhost:30001/health || echo "❌ 아직 준비 중..."

echo "Qwen (30002):"
curl -s http://localhost:30002/health || echo "❌ 아직 준비 중..."

echo "Yi (30003):"
curl -s http://localhost:30003/health || echo "❌ 아직 준비 중..."

echo ""
echo "🎯 배포 완료!"
echo "엔드포인트:"
echo "  - TinyLlama: http://localhost:30001"
echo "  - Qwen 2.5-3B: http://localhost:30002"
echo "  - Yi-6B: http://localhost:30003"
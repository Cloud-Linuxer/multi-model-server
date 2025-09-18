#!/bin/bash

echo "🧪 SGLang 멀티컨테이너 적절한 메모리 설정 테스트"
echo ""

# 1. TinyLlama - 적절한 메모리
echo "1️⃣ TinyLlama 시작 (mem-fraction: 0.10)..."
docker run -d \
  --name sglang-tiny \
  --runtime nvidia \
  --gpus all \
  -p 30001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 4g \
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

echo "대기 40초..."
sleep 40

echo "메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""
echo "컨테이너 상태:"
docker ps | grep sglang-tiny

# 2. 두 번째 TinyLlama
echo ""
echo "2️⃣ 두 번째 TinyLlama 시작 (mem-fraction: 0.10)..."
docker run -d \
  --name sglang-tiny2 \
  --runtime nvidia \
  --gpus all \
  -p 30002:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 4g \
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

echo "대기 40초..."
sleep 40

echo "메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""
echo "컨테이너 상태:"
docker ps | grep sglang

echo ""
echo "✅ 테스트 완료!"
echo ""
echo "API 테스트:"
curl -X POST http://localhost:30001/generate -H "Content-Type: application/json" -d '{"text": "Hello", "max_new_tokens": 10}' 2>/dev/null | jq -r '.text' 2>/dev/null || echo "API 1 실패"
curl -X POST http://localhost:30002/generate -H "Content-Type: application/json" -d '{"text": "World", "max_new_tokens": 10}' 2>/dev/null | jq -r '.text' 2>/dev/null || echo "API 2 실패"

#!/bin/bash

echo "🧪 SGLang 극소 메모리 설정 테스트"
echo "목표: 3개 모델을 매우 작은 메모리로 실행"
echo ""

# 1. TinyLlama - 극소 메모리
echo "1️⃣ TinyLlama 시작 (mem-fraction: 0.05)..."
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
  --mem-fraction-static 0.05 \
  --max-total-tokens 512 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

echo "대기 30초..."
sleep 30

echo "메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
docker logs sglang-tiny --tail 5

# 2. 두 번째 TinyLlama
echo ""
echo "2️⃣ 두 번째 TinyLlama 시작 (mem-fraction: 0.05)..."
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
  --mem-fraction-static 0.05 \
  --max-total-tokens 512 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

echo "대기 30초..."
sleep 30

echo "메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "컨테이너 상태:"
docker ps | grep sglang

echo ""
echo "두 번째 컨테이너 로그:"
docker logs sglang-tiny2 --tail 10

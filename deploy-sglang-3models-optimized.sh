#!/bin/bash

echo "🚀 SGLang 3개 모델 최적화 배포"
echo "메모리 할당 전략: 모델 크기 기반 적절한 할당"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null
docker rm sglang-tinyllama sglang-qwen sglang-yi 2>/dev/null

# GPU 메모리 상태 확인
echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# 1. TinyLlama (1.1B) - 가장 작은 메모리
echo ""
echo "1️⃣ TinyLlama 1.1B 배포 중..."
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
  --max-total-tokens 2048 \
  --max-running-requests 32 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --log-level info

echo "⏳ TinyLlama 초기화 중 (30초)..."
sleep 30

# 메모리 확인
echo "📊 TinyLlama 실행 후:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 2. Qwen 2.5-3B
echo ""
echo "2️⃣ Qwen 2.5-3B 배포 중..."
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
  --mem-fraction-static 0.15 \
  --max-total-tokens 2048 \
  --max-running-requests 24 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --log-level info

echo "⏳ Qwen 초기화 중 (40초)..."
sleep 40

# 메모리 확인
echo "📊 Qwen 실행 후:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 3. Yi-6B
echo ""
echo "3️⃣ Yi-6B 배포 중..."
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
  --mem-fraction-static 0.30 \
  --max-total-tokens 2048 \
  --max-running-requests 16 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --log-level info

echo "⏳ Yi 초기화 중 (50초)..."
sleep 50

# 최종 상태 확인
echo ""
echo "✅ 배포 완료!"
echo ""
echo "📊 최종 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
echo "🐳 실행 중인 컨테이너:"
docker ps | grep sglang

echo ""
echo "🔍 헬스 체크:"
for port in 30001 30002 30003; do
  echo -n "포트 $port: "
  curl -s http://localhost:$port/health 2>/dev/null && echo " ✅" || echo " ⏳ 준비 중..."
done

echo ""
echo "📡 API 엔드포인트:"
echo "  - TinyLlama: http://localhost:30001/generate"
echo "  - Qwen 2.5-3B: http://localhost:30002/generate"
echo "  - Yi-6B: http://localhost:30003/generate"

echo ""
echo "💡 메모리 할당 전략:"
echo "  - TinyLlama: 10% (3.2GB) - 모델 2.15GB + KV캐시"
echo "  - Qwen: 15% (4.9GB) - 모델 3GB + KV캐시"
echo "  - Yi: 30% (9.8GB) - 모델 6GB + KV캐시"
echo "  - 총 사용: 55% (약 18GB/32GB)"
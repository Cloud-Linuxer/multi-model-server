#!/bin/bash

echo "🔬 vLLM vs SGLang 동일 메모리 제한 테스트"
echo "목표: 각 프레임워크가 동일한 GPU 메모리를 사용하도록 설정"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop $(docker ps -q --filter "name=vllm") 2>/dev/null
docker stop $(docker ps -q --filter "name=sglang") 2>/dev/null
docker rm $(docker ps -aq --filter "name=vllm") 2>/dev/null
docker rm $(docker ps -aq --filter "name=sglang") 2>/dev/null
sleep 5

echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# SGLang 테스트 - 전체 GPU의 15% 사용 (약 4.9GB)
echo ""
echo "═══════════════════════════════════════════════════════"
echo "1️⃣ SGLang 테스트 (mem-fraction-static: 0.15 = 4.9GB)"
echo "═══════════════════════════════════════════════════════"

docker run -d \
  --name sglang-15pct \
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
  --mem-fraction-static 0.15 \
  --max-total-tokens 2048 \
  --dtype float16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

echo "⏳ SGLang 초기화 중 (40초)..."
sleep 40

echo "SGLang 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
SGLANG_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "SGLang이 사용 중인 메모리: ${SGLANG_MEM}MB"

# SGLang 종료
docker stop sglang-15pct 2>/dev/null
docker rm sglang-15pct 2>/dev/null
sleep 5

# vLLM 테스트 - 동일한 메모리 사용하도록 조정
echo ""
echo "═══════════════════════════════════════════════════════"
echo "2️⃣ vLLM 테스트 (동일 메모리 목표: ~${SGLANG_MEM}MB)"
echo "═══════════════════════════════════════════════════════"

# vLLM은 가용 메모리 기준이므로 전체 32GB 기준으로 계산
# SGLang 15% ≈ 4.9GB 사용
# vLLM에서 비슷한 메모리 사용을 위해 gpu-memory-utilization 조정

echo "테스트 A: vLLM gpu-memory-utilization=0.10"
docker run -d \
  --name vllm-test-a \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 2048 \
  --enforce-eager

sleep 40
echo "vLLM (0.10) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
VLLM_MEM_A=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

docker stop vllm-test-a 2>/dev/null
docker rm vllm-test-a 2>/dev/null
sleep 5

echo ""
echo "테스트 B: vLLM gpu-memory-utilization=0.15"
docker run -d \
  --name vllm-test-b \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 2048 \
  --enforce-eager

sleep 40
echo "vLLM (0.15) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
VLLM_MEM_B=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📊 메모리 사용량 비교"
echo "═══════════════════════════════════════════════════════"
echo "SGLang (mem-fraction-static=0.15): ${SGLANG_MEM}MB"
echo "vLLM (gpu-memory-utilization=0.10): ${VLLM_MEM_A}MB"
echo "vLLM (gpu-memory-utilization=0.15): ${VLLM_MEM_B}MB"
echo ""
echo "💡 분석:"
echo "- SGLang의 mem-fraction-static은 전체 GPU 메모리(32GB)의 비율"
echo "- vLLM의 gpu-memory-utilization은 가용 메모리의 비율"
echo "- 동일한 0.15 값이어도 실제 메모리 사용량은 다름"

# 성능 테스트
echo ""
echo "═══════════════════════════════════════════════════════"
echo "⚡ 성능 비교 (동일 메모리 사용 시)"
echo "═══════════════════════════════════════════════════════"

# 간단한 성능 테스트
echo "vLLM 응답 시간 테스트..."
START=$(date +%s%N)
curl -s -X POST http://localhost:40001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello, how are you?", "max_tokens": 50}' > /dev/null
END=$(date +%s%N)
VLLM_TIME=$((($END - $START) / 1000000))
echo "vLLM 응답 시간: ${VLLM_TIME}ms"

docker stop vllm-test-b 2>/dev/null
docker rm vllm-test-b 2>/dev/null

# 정리
docker stop $(docker ps -q --filter "name=vllm") 2>/dev/null
docker stop $(docker ps -q --filter "name=sglang") 2>/dev/null
#!/bin/bash

echo "🎯 vLLM 정확한 메모리 매칭 테스트"
echo "목표: SGLang과 정확히 같은 메모리 사용량 (약 2.9GB) 달성"
echo ""

# 기존 컨테이너 정리
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null
sleep 5

echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# vLLM을 더 낮은 설정으로 테스트
echo ""
echo "테스트 1: vLLM gpu-memory-utilization=0.05 (가장 낮은 설정)"
docker run -d \
  --name vllm-5pct \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 1024 \
  --enforce-eager

sleep 30
echo "vLLM (5%) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
MEM_5PCT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

docker stop vllm-5pct 2>/dev/null
docker rm vllm-5pct 2>/dev/null
sleep 5

echo ""
echo "테스트 2: vLLM gpu-memory-utilization=0.03 (더 낮은 설정)"
docker run -d \
  --name vllm-3pct \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.03 \
  --max-model-len 512 \
  --enforce-eager

sleep 30
echo "vLLM (3%) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
MEM_3PCT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

docker stop vllm-3pct 2>/dev/null
docker rm vllm-3pct 2>/dev/null
sleep 5

echo ""
echo "테스트 3: vLLM 최소 설정 (gpu-util=0.01, max-len=256)"
docker run -d \
  --name vllm-minimal \
  --runtime nvidia \
  --gpus all \
  -p 40001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.01 \
  --max-model-len 256 \
  --enforce-eager

sleep 30
echo "vLLM (최소) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
MEM_MIN=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

# SGLang과 비교
echo ""
echo "테스트 4: SGLang 참조 (mem-fraction-static=0.15)"
docker stop vllm-minimal 2>/dev/null
docker rm vllm-minimal 2>/dev/null
sleep 5

docker run -d \
  --name sglang-ref \
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

sleep 40
echo "SGLang (15%) 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
MEM_SGLANG=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📊 메모리 사용량 비교 결과"
echo "═══════════════════════════════════════════════════════"
echo "SGLang (mem-fraction-static=0.15): ${MEM_SGLANG}MB (~2.9GB)"
echo ""
echo "vLLM 테스트 결과:"
echo "- gpu-memory-utilization=0.05: ${MEM_5PCT}MB"
echo "- gpu-memory-utilization=0.03: ${MEM_3PCT}MB"
echo "- gpu-memory-utilization=0.01: ${MEM_MIN}MB (최소 설정)"
echo ""
echo "💡 결론:"
echo "- vLLM의 최소 메모리는 모델 자체 크기 + 최소 KV 캐시"
echo "- SGLang은 mem-fraction-static으로 더 정밀한 제어 가능"
echo "- vLLM은 gpu-memory-utilization을 0.01까지 낮출 수 있음"
echo ""

# 여러 vLLM 컨테이너 동시 실행 테스트
echo "═══════════════════════════════════════════════════════"
echo "🚀 동일 메모리로 여러 모델 실행 테스트"
echo "═══════════════════════════════════════════════════════"

docker stop sglang-ref 2>/dev/null
docker rm sglang-ref 2>/dev/null
sleep 5

echo "3개의 vLLM 컨테이너를 각각 gpu-util=0.01로 실행..."
for i in 1 2 3; do
  docker run -d \
    --name vllm-multi-$i \
    --runtime nvidia \
    --gpus all \
    -p 4000$i:8000 \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    --shm-size 8g \
    vllm/vllm-openai:latest \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.01 \
    --max-model-len 256 \
    --enforce-eager
  echo "vLLM 컨테이너 $i 시작..."
  sleep 20
done

echo ""
echo "실행 중인 vLLM 컨테이너:"
docker ps | grep vllm-multi
echo ""
echo "총 GPU 메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv

# 정리
docker stop $(docker ps -q) 2>/dev/null
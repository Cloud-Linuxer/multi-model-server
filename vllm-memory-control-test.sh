#!/bin/bash

echo "🔬 vLLM 메모리 제어 옵션 테스트"
echo ""
echo "vLLM의 메모리 제어 파라미터들:"
echo "1. --gpu-memory-utilization: KV 캐시용 가용 메모리 비율 (0.1~0.9)"
echo "2. --max-model-len: 최대 시퀀스 길이 (메모리 사용량 직접 영향)"
echo "3. --enforce-eager: Eager 모드 (메모리 사용량 감소)"
echo "4. --kv-cache-dtype: KV 캐시 데이터 타입 (fp8 사용 시 메모리 50% 절약)"
echo "5. --quantization: 모델 양자화 (AWQ, GPTQ 등)"
echo ""

# 기존 컨테이너 정리
docker stop vllm-test1 vllm-test2 vllm-test3 2>/dev/null
docker rm vllm-test1 vllm-test2 vllm-test3 2>/dev/null

echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 테스트 1: 매우 낮은 메모리 사용 (10%)
echo ""
echo "테스트 1: 최소 메모리 설정 (gpu-util=0.05, max-len=512)"
docker run -d \
  --name vllm-test1 \
  --runtime nvidia \
  --gpus all \
  -p 41001:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 512 \
  --enforce-eager \
  --dtype float16

sleep 30
echo "메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
docker ps | grep vllm-test1 || echo "실패"

# 테스트 2: KV 캐시 타입 변경으로 메모리 절약
echo ""
echo "테스트 2: FP8 KV 캐시 사용 (메모리 50% 절약)"
docker run -d \
  --name vllm-test2 \
  --runtime nvidia \
  --gpus all \
  -p 41002:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.1 \
  --max-model-len 1024 \
  --kv-cache-dtype fp8 \
  --enforce-eager

sleep 30
echo "메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
docker ps | grep vllm-test2 || echo "실패"

# 테스트 3: 환경변수로 추가 제어
echo ""
echo "테스트 3: 환경변수 CUDA_VISIBLE_DEVICES로 메모리 제한 시뮬레이션"
docker run -d \
  --name vllm-test3 \
  --runtime nvidia \
  --gpus all \
  -p 41003:8000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  --shm-size 8g \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.08 \
  --max-model-len 768 \
  --enforce-eager

sleep 30
echo "메모리 사용량:"
nvidia-smi --query-gpu=memory.used --format=csv
docker ps | grep vllm-test3 || echo "실패"

echo ""
echo "📊 최종 상태:"
docker ps | grep vllm-test
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "💡 vLLM 메모리 제어 전략:"
echo "1. gpu-memory-utilization을 0.05까지 낮출 수 있음 (SGLang의 mem-fraction-static과 유사)"
echo "2. max-model-len을 줄여 KV 캐시 크기 제한"
echo "3. kv-cache-dtype을 fp8로 변경하여 메모리 50% 절약"
echo "4. enforce-eager 모드로 추가 메모리 절약"
echo "5. PYTORCH_CUDA_ALLOC_CONF로 메모리 할당 세분화"
echo ""
echo "⚠️ 차이점:"
echo "- vLLM: 가용 메모리의 비율로 계산 (다른 프로세스 고려)"
echo "- SGLang: 전체 GPU 메모리의 비율로 계산 (독점적)"
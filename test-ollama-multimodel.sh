#!/bin/bash

echo "🔬 Ollama 멀티모델 서빙 테스트"
echo "================================"
echo ""

# GPU 상태 확인
echo "📊 초기 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

# TinyLlama 테스트
echo "1️⃣ TinyLlama 모델 호출:"
START=$(date +%s%N)
curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "tinyllama:1.1b",
    "prompt": "Sing a beautiful song about spring with blooming flowers.",
    "stream": false,
    "options": {"num_predict": 100, "temperature": 0.7}
  }' | jq -r '.response' | head -5
END=$(date +%s%N)
ELAPSED=$((($END - $START) / 1000000))
echo "⏱️ TinyLlama 응답 시간: ${ELAPSED}ms"
echo ""

# GPU 상태 확인
echo "📊 TinyLlama 실행 후 GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

# Qwen 테스트
echo "2️⃣ Qwen 2.5-3B 모델 호출:"
START=$(date +%s%N)
curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "qwen2.5:3b",
    "prompt": "唱一首关于春天百花盛开之美的歌曲。",
    "stream": false,
    "options": {"num_predict": 100, "temperature": 0.7}
  }' | jq -r '.response' | head -5
END=$(date +%s%N)
ELAPSED=$((($END - $START) / 1000000))
echo "⏱️ Qwen 응답 시간: ${ELAPSED}ms"
echo ""

# GPU 상태 확인
echo "📊 Qwen 실행 후 GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

# Yi 테스트
echo "3️⃣ Yi-6B 모델 호출:"
START=$(date +%s%N)
curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "yi:6b",
    "prompt": "봄에 피어나는 꽃들의 아름다움을 노래해주세요.",
    "stream": false,
    "options": {"num_predict": 100, "temperature": 0.7}
  }' | jq -r '.response' | head -5
END=$(date +%s%N)
ELAPSED=$((($END - $START) / 1000000))
echo "⏱️ Yi 응답 시간: ${ELAPSED}ms"
echo ""

# GPU 상태 확인
echo "📊 Yi 실행 후 GPU 메모리:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

# 다시 TinyLlama 호출 (모델 전환 테스트)
echo "4️⃣ TinyLlama 재호출 (모델 전환 테스트):"
START=$(date +%s%N)
curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "tinyllama:1.1b",
    "prompt": "What is the capital of France?",
    "stream": false,
    "options": {"num_predict": 20}
  }' | jq -r '.response'
END=$(date +%s%N)
ELAPSED=$((($END - $START) / 1000000))
echo "⏱️ TinyLlama 재호출 응답 시간: ${ELAPSED}ms"
echo ""

# 최종 GPU 상태
echo "📊 최종 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

# 현재 로드된 모델 확인
echo "📋 현재 메모리에 있는 모델:"
docker exec ollama ollama list
echo ""

echo "✅ Ollama 멀티모델 서빙 특징:"
echo "  • 단일 API 엔드포인트 (11434 포트)"
echo "  • 동적 모델 로딩/언로딩"
echo "  • 자동 메모리 관리"
echo "  • 모델 간 자유로운 전환"
echo "  • GGUF 포맷으로 메모리 효율적 운영"
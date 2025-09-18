#!/bin/bash

echo "🚀 Ollama 간단한 멀티모델 테스트"
echo ""

# Ollama 서버 시작 (백그라운드)
echo "📦 Ollama 서버 시작..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"
sleep 5

# 서버 상태 확인
echo "🔍 서버 상태 확인..."
curl -s http://localhost:11434/api/tags || echo "서버 시작 중..."
sleep 5

echo ""
echo "📊 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# TinyLlama 테스트 (이미 있을 수 있음)
echo ""
echo "1️⃣ TinyLlama 테스트..."
curl -s -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "tinyllama:1.1b",
    "prompt": "Hello, how are you?",
    "stream": false,
    "options": {
      "temperature": 0.7,
      "num_predict": 50
    }
  }' | jq -r '.response' 2>/dev/null || (echo "TinyLlama 다운로드 중..." && ollama pull tinyllama:1.1b)

# 모델 리스트 확인
echo ""
echo "📋 사용 가능한 모델:"
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || ollama list

echo ""
echo "💡 Ollama 특징:"
echo "  - 동적 모델 로딩/언로딩"
echo "  - 자동 메모리 관리 (사용하지 않는 모델 자동 언로드)"
echo "  - 단일 API 엔드포인트 (11434 포트)"
echo "  - GGUF 포맷 사용 (CPU/GPU 혼합 실행 가능)"

# 벤치마크 준비
echo ""
echo "📊 간단한 성능 테스트 (TinyLlama)..."

# 10번 테스트
total_time=0
success_count=0

for i in {1..10}; do
    start_time=$(date +%s%N)

    response=$(curl -s -X POST http://localhost:11434/api/generate \
      -d '{
        "model": "tinyllama:1.1b",
        "prompt": "Sing a song about spring",
        "stream": false,
        "options": {
          "num_predict": 100
        }
      }' 2>/dev/null)

    end_time=$(date +%s%N)

    if [ ! -z "$response" ]; then
        elapsed=$((($end_time - $start_time) / 1000000))  # ms
        echo "  Test $i: ${elapsed}ms"
        total_time=$((total_time + elapsed))
        success_count=$((success_count + 1))
    else
        echo "  Test $i: Failed"
    fi

    sleep 0.1
done

if [ $success_count -gt 0 ]; then
    avg_time=$((total_time / success_count))
    echo ""
    echo "✅ 결과:"
    echo "  성공률: $success_count/10"
    echo "  평균 응답 시간: ${avg_time}ms"
fi

echo ""
echo "📡 API 정보:"
echo "  엔드포인트: http://localhost:11434/api/generate"
echo "  모델 리스트: http://localhost:11434/api/tags"

# 프로세스 정리
echo ""
echo "🧹 Ollama 서버 종료..."
kill $OLLAMA_PID 2>/dev/null
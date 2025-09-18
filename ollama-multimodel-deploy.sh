#!/bin/bash

echo "🚀 Ollama 멀티모델 서빙 테스트"
echo "목표: TinyLlama, Qwen, Yi 모델을 Ollama로 동시 서빙"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# Ollama 서비스 시작
echo "📦 Ollama 서비스 시작..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"
sleep 5

echo "📊 초기 GPU 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# 모델 다운로드 및 로드
echo ""
echo "1️⃣ TinyLlama 1.1B 다운로드 및 로드..."
ollama pull tinyllama:1.1b
ollama run tinyllama:1.1b "test" &
sleep 10

echo "📊 TinyLlama 로드 후:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "2️⃣ Qwen 2.5:3B 다운로드 및 로드..."
ollama pull qwen2.5:3b
ollama run qwen2.5:3b "test" &
sleep 10

echo "📊 Qwen 로드 후:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo ""
echo "3️⃣ Yi 1.5:6B 다운로드 및 로드..."
ollama pull yi:6b
ollama run yi:6b "test" &
sleep 10

echo "📊 Yi 로드 후:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 모델 상태 확인
echo ""
echo "✅ 로드된 모델 확인:"
ollama list

echo ""
echo "📊 최종 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# API 테스트
echo ""
echo "🔍 API 엔드포인트 테스트:"

echo "Testing TinyLlama..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "tinyllama:1.1b",
  "prompt": "Hello",
  "stream": false
}' | jq -r '.response' 2>/dev/null || echo "실패"

echo ""
echo "Testing Qwen..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Hello",
  "stream": false
}' | jq -r '.response' 2>/dev/null || echo "실패"

echo ""
echo "Testing Yi..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "yi:6b",
  "prompt": "Hello",
  "stream": false
}' | jq -r '.response' 2>/dev/null || echo "실패"

echo ""
echo "💡 Ollama 멀티모델 특징:"
echo "  - 동적 모델 로딩/언로딩"
echo "  - 메모리 자동 관리"
echo "  - 단일 API 엔드포인트 (포트 11434)"
echo "  - 모델 간 자동 전환"

echo ""
echo "📡 API 엔드포인트:"
echo "  - 모든 모델: http://localhost:11434/api/generate"
echo "  - model 파라미터로 모델 선택"
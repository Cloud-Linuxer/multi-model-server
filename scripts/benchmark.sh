#!/bin/bash

echo "🚀 vLLM 성능 벤치마크"
echo "====================="
echo ""

# Qwen 벤치마크
echo "📊 Qwen 2.5-3B 응답 시간 (5회 평균):"
total=0
for i in {1..5}; do
    start=$(date +%s%3N)
    curl -s -X POST http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "qwen25_3b", "prompt": "Hello", "max_tokens": 10}' > /dev/null
    end=$(date +%s%3N)
    elapsed=$((end - start))
    echo "  요청 $i: ${elapsed}ms"
    total=$((total + elapsed))
done
avg=$((total / 5))
echo "  평균: ${avg}ms"
echo ""

# TinyLlama 벤치마크
echo "📊 TinyLlama 1.1B 응답 시간 (5회 평균):"
total=0
for i in {1..5}; do
    start=$(date +%s%3N)
    curl -s -X POST http://localhost:8002/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "tinyllama_1b", "prompt": "Hello", "max_tokens": 10}' > /dev/null
    end=$(date +%s%3N)
    elapsed=$((end - start))
    echo "  요청 $i: ${elapsed}ms"
    total=$((total + elapsed))
done
avg=$((total / 5))
echo "  평균: ${avg}ms"
echo ""

# 동시 요청 테스트
echo "🔥 동시 요청 테스트 (각 모델 3개씩):"
(
    curl -s -X POST http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{"model": "qwen25_3b", "prompt": "AI is", "max_tokens": 10}' > /tmp/qwen1.txt &
    curl -s -X POST http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{"model": "qwen25_3b", "prompt": "ML is", "max_tokens": 10}' > /tmp/qwen2.txt &
    curl -s -X POST http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{"model": "qwen25_3b", "prompt": "NLP is", "max_tokens": 10}' > /tmp/qwen3.txt &
    curl -s -X POST http://localhost:8002/v1/completions -H "Content-Type: application/json" -d '{"model": "tinyllama_1b", "prompt": "AI is", "max_tokens": 10}' > /tmp/tiny1.txt &
    curl -s -X POST http://localhost:8002/v1/completions -H "Content-Type: application/json" -d '{"model": "tinyllama_1b", "prompt": "ML is", "max_tokens": 10}' > /tmp/tiny2.txt &
    curl -s -X POST http://localhost:8002/v1/completions -H "Content-Type: application/json" -d '{"model": "tinyllama_1b", "prompt": "NLP is", "max_tokens": 10}' > /tmp/tiny3.txt &
    wait
)
echo "✅ 6개 동시 요청 완료"
echo ""

# GPU 상태
echo "💾 현재 GPU 상태:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
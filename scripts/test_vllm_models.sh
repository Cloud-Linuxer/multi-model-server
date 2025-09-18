#!/bin/bash

echo "🚀 vLLM 멀티모델 서버 테스트"
echo "=============================="
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 헬스체크
echo "1️⃣ 헬스 체크:"
echo "--------------"
for port in 8001 8002 8003; do
    case $port in
        8001) name="Qwen 2.5-3B" ;;
        8002) name="TinyLlama 1.1B" ;;
        8003) name="Phi-2 2.7B" ;;
    esac

    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $name (Port $port) - 정상 작동"
    else
        echo -e "${YELLOW}⏳${NC} $name (Port $port) - 아직 준비 안됨"
    fi
done
echo ""

# 모델 테스트
echo "2️⃣ 모델 추론 테스트:"
echo "-------------------"

# Qwen 테스트
echo -e "${YELLOW}Testing Qwen 2.5-3B...${NC}"
response=$(curl -s -X POST http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25_3b",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.7
    }' 2>/dev/null)

if echo "$response" | grep -q "choices"; then
    echo -e "${GREEN}✅ Qwen 응답:${NC}"
    echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['text'][:100])" 2>/dev/null || echo "응답 파싱 실패"
else
    echo -e "${RED}❌ Qwen 응답 없음${NC}"
fi
echo ""

# TinyLlama 테스트
echo -e "${YELLOW}Testing TinyLlama 1.1B...${NC}"
response=$(curl -s -X POST http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tinyllama_1b",
        "prompt": "What is artificial intelligence?",
        "max_tokens": 10,
        "temperature": 0.7
    }' 2>/dev/null)

if echo "$response" | grep -q "choices"; then
    echo -e "${GREEN}✅ TinyLlama 응답:${NC}"
    echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['text'][:100])" 2>/dev/null || echo "응답 파싱 실패"
else
    echo -e "${RED}❌ TinyLlama 응답 없음${NC}"
fi
echo ""

# Phi 테스트
echo -e "${YELLOW}Testing Phi-2 2.7B...${NC}"
response=$(curl -s -X POST http://localhost:8003/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi2_2b",
        "prompt": "Write a Python function to",
        "max_tokens": 10,
        "temperature": 0.7
    }' 2>/dev/null)

if echo "$response" | grep -q "choices"; then
    echo -e "${GREEN}✅ Phi-2 응답:${NC}"
    echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['text'][:100])" 2>/dev/null || echo "응답 파싱 실패"
else
    echo -e "${RED}❌ Phi-2 응답 없음${NC}"
fi
echo ""

# 게이트웨이 테스트
echo "3️⃣ 통합 게이트웨이 테스트 (Port 8000):"
echo "--------------------------------------"
curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null || echo "게이트웨이 응답 없음"
echo ""

# GPU 상태
echo "4️⃣ GPU 리소스 사용:"
echo "------------------"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

echo "💡 팁: 모델 로딩에 2-3분 소요될 수 있습니다."
echo "     docker logs vllm-qwen 으로 진행상황 확인 가능"
#!/bin/bash

echo "📊 vLLM Docker 상태 확인"
echo "========================"
echo ""

# 컨테이너 상태
echo "🐳 컨테이너 상태:"
docker compose -f docker-compose-vllm.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""

# 각 모델 헬스 체크
echo "🏥 모델 헬스 체크:"
for port in 8001 8002 8003; do
    case $port in
        8001) name="Qwen 2.5-3B" ;;
        8002) name="Llama 3.2-3B" ;;
        8003) name="Gemma 2-2B" ;;
    esac

    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $name (Port $port) - 정상 작동"
    else
        echo "⏳ $name (Port $port) - 로딩 중..."
    fi
done
echo ""

# 통합 게이트웨이 체크
echo "🌐 통합 게이트웨이 (Port 8000):"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Nginx Gateway - 정상 작동"
else
    echo "❌ Nginx Gateway - 응답 없음"
fi
echo ""

# 모델 목록
echo "📋 사용 가능한 모델:"
curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "아직 준비 중..."
echo ""

# 리소스 사용량
echo "💾 GPU 메모리 사용량:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU 정보 없음"
echo ""

echo "💡 팁: 모델 로딩에 1-2분 정도 소요됩니다."
echo "     로그 확인: docker logs vllm-qwen"
#!/bin/bash

# vLLM 멀티모델 서버 관리 스크립트
# 3개 모델을 백그라운드로 실행하고 관리

set -e

# 설정
LOG_DIR="./logs"
PID_DIR="./pids"
mkdir -p $LOG_DIR $PID_DIR

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 모델 시작
start_model() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local PORT=$3
    local GPU_MEM=${4:-0.30}

    echo -e "${YELLOW}Starting $MODEL_NAME on port $PORT...${NC}"

    # 기존 프로세스 종료
    if [ -f "$PID_DIR/$MODEL_NAME.pid" ]; then
        old_pid=$(cat "$PID_DIR/$MODEL_NAME.pid")
        if kill -0 $old_pid 2>/dev/null; then
            kill $old_pid 2>/dev/null || true
            sleep 2
        fi
    fi

    # vLLM 서버 시작 (백그라운드)
    nohup python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $PORT \
        --model $MODEL_PATH \
        --served-model-name $MODEL_NAME \
        --max-model-len 2048 \
        --gpu-memory-utilization $GPU_MEM \
        --dtype auto \
        --trust-remote-code \
        > "$LOG_DIR/${MODEL_NAME}.log" 2>&1 &

    # PID 저장
    echo $! > "$PID_DIR/$MODEL_NAME.pid"
    echo -e "${GREEN}✓ $MODEL_NAME started (PID: $!)${NC}"
}

# 함수: 모든 모델 시작
start_all() {
    echo -e "${GREEN}=== Starting vLLM Multi-Model Servers ===${NC}"

    # Qwen 2.5-3B
    start_model "qwen25_3b" "Qwen/Qwen2.5-3B-Instruct" 8001 0.33
    sleep 5

    # Llama 3.2-3B
    start_model "llama32_3b" "meta-llama/Llama-3.2-3B-Instruct" 8002 0.33
    sleep 5

    # Gemma 2-2B
    start_model "gemma2_2b" "google/gemma-2-2b-it" 8003 0.30

    echo -e "${GREEN}All models started!${NC}"
    echo ""
    status
}

# 함수: 상태 확인
status() {
    echo -e "${YELLOW}=== Model Server Status ===${NC}"

    for pidfile in $PID_DIR/*.pid; do
        if [ -f "$pidfile" ]; then
            model=$(basename "$pidfile" .pid)
            pid=$(cat "$pidfile")

            if kill -0 $pid 2>/dev/null; then
                port=$(grep -oP 'port \K\d+' "$LOG_DIR/${model}.log" | head -1)
                echo -e "${GREEN}✓${NC} $model (PID: $pid, Port: $port) - Running"

                # 헬스체크
                if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
                    echo -e "  └─ Health: ${GREEN}OK${NC}"
                else
                    echo -e "  └─ Health: ${YELLOW}Starting...${NC}"
                fi
            else
                echo -e "${RED}✗${NC} $model - Not running"
            fi
        fi
    done
}

# 함수: 모든 모델 중지
stop_all() {
    echo -e "${YELLOW}=== Stopping All Model Servers ===${NC}"

    for pidfile in $PID_DIR/*.pid; do
        if [ -f "$pidfile" ]; then
            model=$(basename "$pidfile" .pid)
            pid=$(cat "$pidfile")

            if kill -0 $pid 2>/dev/null; then
                kill $pid
                echo -e "${GREEN}✓${NC} Stopped $model (PID: $pid)"
            fi

            rm "$pidfile"
        fi
    done

    # vLLM 프로세스 정리
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    echo -e "${GREEN}All models stopped!${NC}"
}

# 함수: 로그 보기
logs() {
    local MODEL=${1:-all}

    if [ "$MODEL" == "all" ]; then
        tail -f $LOG_DIR/*.log
    else
        tail -f "$LOG_DIR/${MODEL}.log"
    fi
}

# 함수: 테스트
test_models() {
    echo -e "${YELLOW}=== Testing Model Endpoints ===${NC}"

    # Qwen 테스트
    echo -e "\n${YELLOW}Testing Qwen 2.5-3B...${NC}"
    curl -s -X POST http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "qwen25_3b", "prompt": "Hello, how are", "max_tokens": 10}' | \
        python -m json.tool | head -20

    # Llama 테스트
    echo -e "\n${YELLOW}Testing Llama 3.2-3B...${NC}"
    curl -s -X POST http://localhost:8002/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "llama32_3b", "prompt": "The weather today", "max_tokens": 10}' | \
        python -m json.tool | head -20

    # Gemma 테스트
    echo -e "\n${YELLOW}Testing Gemma 2-2B...${NC}"
    curl -s -X POST http://localhost:8003/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gemma2_2b", "prompt": "AI technology", "max_tokens": 10}' | \
        python -m json.tool | head -20
}

# 메인 명령어 처리
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        status
        ;;
    logs)
        logs $2
        ;;
    test)
        test_models
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all model servers"
        echo "  stop    - Stop all model servers"
        echo "  restart - Restart all model servers"
        echo "  status  - Check server status"
        echo "  logs    - View logs (all or specific model)"
        echo "  test    - Test all model endpoints"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs qwen25_3b"
        echo "  $0 status"
        exit 1
        ;;
esac
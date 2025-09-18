#!/bin/bash

# tmux를 사용한 vLLM 멀티모델 관리
# 각 모델이 별도 tmux 창에서 실행됨

SESSION="vllm-models"

# tmux 세션 시작
start_tmux() {
    # 기존 세션 종료
    tmux kill-session -t $SESSION 2>/dev/null

    # 새 세션 생성 (Qwen)
    tmux new-session -d -s $SESSION -n "qwen" \
        "python -m vllm.entrypoints.openai.api_server \
            --host 0.0.0.0 --port 8001 \
            --model Qwen/Qwen2.5-3B-Instruct \
            --served-model-name qwen25_3b \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.33 \
            --dtype auto"

    # Llama 창 추가
    tmux new-window -t $SESSION -n "llama" \
        "python -m vllm.entrypoints.openai.api_server \
            --host 0.0.0.0 --port 8002 \
            --model meta-llama/Llama-3.2-3B-Instruct \
            --served-model-name llama32_3b \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.33 \
            --dtype auto"

    # Gemma 창 추가
    tmux new-window -t $SESSION -n "gemma" \
        "python -m vllm.entrypoints.openai.api_server \
            --host 0.0.0.0 --port 8003 \
            --model google/gemma-2-2b-it \
            --served-model-name gemma2_2b \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.30 \
            --dtype auto"

    echo "✅ tmux 세션 '$SESSION' 시작됨"
    echo ""
    echo "사용법:"
    echo "  모든 창 보기:     tmux attach -t $SESSION"
    echo "  특정 모델 보기:   tmux attach -t $SESSION:qwen"
    echo "  창 전환:          Ctrl+B, 숫자 (0=qwen, 1=llama, 2=gemma)"
    echo "  세션 나가기:      Ctrl+B, D"
    echo "  세션 종료:        tmux kill-session -t $SESSION"
}

case "$1" in
    start)
        start_tmux
        ;;
    attach)
        tmux attach -t $SESSION
        ;;
    stop)
        tmux kill-session -t $SESSION
        echo "✅ tmux 세션 종료됨"
        ;;
    list)
        tmux list-windows -t $SESSION
        ;;
    *)
        echo "Usage: $0 {start|attach|stop|list}"
        ;;
esac
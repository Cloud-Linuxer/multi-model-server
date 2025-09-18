#!/bin/bash

echo "🛑 Stopping vLLM servers..."

# Read PIDs from files
if [ -d "logs" ]; then
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat $pidfile)
            model=$(basename $pidfile .pid)
            if kill -0 $pid 2>/dev/null; then
                kill $pid
                echo "✅ Stopped $model (PID: $pid)"
            fi
            rm $pidfile
        fi
    done
fi

# Kill any remaining vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null

echo "✅ All vLLM servers stopped"
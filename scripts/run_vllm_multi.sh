#!/bin/bash

echo "🚀 Starting vLLM Multi-Model Server"
echo "===================================="

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null

# Model configurations
declare -A MODELS
MODELS["qwen"]="Qwen/Qwen2.5-3B-Instruct"
MODELS["llama"]="meta-llama/Llama-3.2-3B-Instruct"
MODELS["gemma"]="google/gemma-2-2b-it"

# Port assignments
declare -A PORTS
PORTS["qwen"]=8001
PORTS["llama"]=8002
PORTS["gemma"]=8003

# GPU memory allocation (adjust per model)
declare -A GPU_MEM
GPU_MEM["qwen"]=0.33
GPU_MEM["llama"]=0.33
GPU_MEM["gemma"]=0.30

# Function to start a model server
start_model() {
    local name=$1
    local model=${MODELS[$name]}
    local port=${PORTS[$name]}
    local gpu_mem=${GPU_MEM[$name]}

    echo "🔄 Starting $name on port $port..."

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --port $port \
        --model $model \
        --served-model-name "${name}_model" \
        --max-model-len 2048 \
        --gpu-memory-utilization $gpu_mem \
        --dtype auto \
        --trust-remote-code \
        > logs/${name}_vllm.log 2>&1 &

    local pid=$!
    echo "   PID: $pid"
    echo $pid > logs/${name}.pid
}

# Create logs directory
mkdir -p logs

# Start all models
echo ""
echo "📦 Launching model servers..."
echo "-----------------------------"

for model_name in "${!MODELS[@]}"; do
    start_model $model_name
    sleep 5  # Give each model time to initialize
done

echo ""
echo "⏳ Waiting for models to initialize (30s)..."
sleep 30

# Check status
echo ""
echo "🏥 Checking server health..."
echo "-----------------------------"

for model_name in "${!MODELS[@]}"; do
    port=${PORTS[$model_name]}
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $model_name on port $port - HEALTHY"
    else
        echo "❌ $model_name on port $port - FAILED"
        echo "   Check logs/\${model_name}_vllm.log for details"
    fi
done

echo ""
echo "📍 Service Endpoints:"
echo "--------------------"
echo "  Qwen 2.5-3B:  http://localhost:8001"
echo "  Llama 3.2-3B: http://localhost:8002"
echo "  Gemma 2-2B:   http://localhost:8003"
echo ""
echo "📝 Logs available in: logs/"
echo ""
echo "To stop all servers: ./scripts/stop_vllm.sh"
echo ""
echo "✨ vLLM Multi-Model Server is running!"
#!/bin/bash

echo "🚀 Starting Multi-Model Server..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed (v1 or v2)
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Set Docker Compose command based on version
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: NVIDIA Docker runtime not available. GPU acceleration disabled."
    echo "To enable GPU, install nvidia-docker2 package."
fi

# Create necessary directories
mkdir -p ~/.cache/huggingface

# Build and start services
echo "📦 Building Docker images..."
$DOCKER_COMPOSE build --parallel

echo "🔄 Starting services..."
$DOCKER_COMPOSE up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
for service in qwen-model llama-model gemma-model nginx api-gateway; do
    if $DOCKER_COMPOSE ps | grep $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
    fi
done

# Display endpoints
echo ""
echo "📍 Service Endpoints:"
echo "   - Main API Gateway: http://localhost:8000"
echo "   - FastAPI Gateway: http://localhost:8080"
echo "   - Qwen Model: http://localhost:8001"
echo "   - Llama Model: http://localhost:8002"
echo "   - Gemma Model: http://localhost:8003"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "🎯 Quick Test:"
echo "   curl http://localhost:8000/health"
echo ""
echo "✨ Multi-Model Server is ready!"
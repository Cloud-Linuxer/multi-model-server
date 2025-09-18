#!/bin/bash

echo "🚀 Starting Multi-Model Server (CPU Mode)..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Set Docker Compose command based on version
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Create necessary directories
mkdir -p ~/.cache/huggingface

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
$DOCKER_COMPOSE -f docker-compose-cpu.yml down 2>/dev/null

# Build and start services
echo "📦 Building Docker images..."
$DOCKER_COMPOSE -f docker-compose-cpu.yml build

echo "🔄 Starting services..."
$DOCKER_COMPOSE -f docker-compose-cpu.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo "🏥 Checking service health..."
for service in api-gateway mock-model nginx redis prometheus grafana; do
    if $DOCKER_COMPOSE -f docker-compose-cpu.yml ps | grep $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
    fi
done

# Display endpoints
echo ""
echo "📍 Service Endpoints:"
echo "   - Main API Gateway (Nginx): http://localhost:8000"
echo "   - FastAPI Gateway: http://localhost:8080"
echo "   - Redis Cache: localhost:6379"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "🎯 Quick Test:"
echo "   curl http://localhost:8080/"
echo "   curl http://localhost:8080/health"
echo ""
echo "⚠️  Note: This is running in CPU mode with mock models for testing"
echo "✨ Multi-Model Server is ready!"
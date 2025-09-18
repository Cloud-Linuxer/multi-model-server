#!/bin/bash

echo "🛑 Stopping Multi-Model Server..."

# Set Docker Compose command based on version
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Stop all services
$DOCKER_COMPOSE down

# Optional: Remove volumes (uncomment if needed)
# $DOCKER_COMPOSE down -v

echo "✅ All services stopped successfully"
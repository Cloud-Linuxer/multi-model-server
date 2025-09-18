#!/bin/bash

echo "🔍 Testing Multi-Model Server Endpoints"
echo "========================================"
echo ""

echo "1️⃣ API Gateway Root:"
curl -s http://localhost:8080/ | python3 -m json.tool | head -10
echo ""

echo "2️⃣ Available Models:"
curl -s http://localhost:8080/v1/models | python3 -m json.tool
echo ""

echo "3️⃣ Test Completion (Qwen):"
curl -s -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2p5_3b", "prompt": "Hello world", "max_tokens": 10}' | python3 -m json.tool
echo ""

echo "4️⃣ Test Chat Completion (Llama):"
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama32_3b", "messages": [{"role": "user", "content": "Hi there"}]}' | python3 -m json.tool
echo ""

echo "5️⃣ Health Check:"
curl -s http://localhost:8080/health | python3 -m json.tool
echo ""

echo "✅ All tests completed!"
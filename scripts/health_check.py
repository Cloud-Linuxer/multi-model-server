#!/usr/bin/env python3

import requests
import json
import sys
from typing import Dict, Any
from datetime import datetime

def check_service_health(url: str, service_name: str) -> Dict[str, Any]:
    """Check health of a single service"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            return {
                "service": service_name,
                "status": "✅ Healthy",
                "response_time": response.elapsed.total_seconds(),
                "details": response.json() if response.content else None
            }
        else:
            return {
                "service": service_name,
                "status": f"⚠️  Unhealthy (Status: {response.status_code})",
                "response_time": response.elapsed.total_seconds()
            }
    except requests.exceptions.RequestException as e:
        return {
            "service": service_name,
            "status": f"❌ Failed ({str(e)})",
            "response_time": None
        }

def test_model_inference(url: str, model_name: str) -> Dict[str, Any]:
    """Test model inference"""
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
            "temperature": 0.7
        }
        response = requests.post(f"{url}/v1/completions", json=payload, timeout=30)
        if response.status_code == 200:
            return {
                "model": model_name,
                "status": "✅ Inference successful",
                "response_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "model": model_name,
                "status": f"❌ Inference failed (Status: {response.status_code})",
                "error": response.text
            }
    except requests.exceptions.RequestException as e:
        return {
            "model": model_name,
            "status": f"❌ Request failed ({str(e)})"
        }

def main():
    print("=" * 60)
    print("🏥 Multi-Model Server Health Check")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Service endpoints
    services = {
        "API Gateway (Nginx)": "http://localhost:8000",
        "FastAPI Gateway": "http://localhost:8080",
        "Qwen Model": "http://localhost:8001",
        "Llama Model": "http://localhost:8002",
        "Gemma Model": "http://localhost:8003",
        "Prometheus": "http://localhost:9090",
        "Grafana": "http://localhost:3000"
    }

    # Check service health
    print("\n📊 Service Health Status:")
    print("-" * 40)
    all_healthy = True

    for service_name, url in services.items():
        result = check_service_health(url, service_name)
        print(f"{result['service']}: {result['status']}")
        if result['response_time']:
            print(f"   Response time: {result['response_time']:.3f}s")
        if "❌" in result['status'] or "⚠️" in result['status']:
            all_healthy = False

    # Test model inference
    print("\n🤖 Model Inference Test:")
    print("-" * 40)

    models = ["qwen2p5_3b", "llama32_3b", "gemma2_2b"]
    gateway_url = "http://localhost:8000"

    for model in models:
        result = test_model_inference(gateway_url, model)
        print(f"{result['model']}: {result['status']}")
        if 'response_time' in result and result['response_time']:
            print(f"   Response time: {result['response_time']:.3f}s")
        if 'error' in result:
            print(f"   Error: {result['error'][:100]}")

    # List available models
    try:
        response = requests.get(f"{gateway_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print("\n📋 Available Models:")
            print("-" * 40)
            for model in models_data.get('data', []):
                print(f"   - {model['id']}")
    except:
        pass

    # Final status
    print("\n" + "=" * 60)
    if all_healthy:
        print("✅ All services are healthy!")
        sys.exit(0)
    else:
        print("⚠️  Some services need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()
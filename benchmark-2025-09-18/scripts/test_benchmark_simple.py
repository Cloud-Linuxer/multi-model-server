#!/usr/bin/env python3
"""
Simple test to verify LLM serving endpoints
"""

import requests
import time
import json

def test_vllm_tinyllama():
    """Test vLLM TinyLlama endpoint"""
    url = "http://localhost:8001/v1/completions"
    payload = {
        "model": "tinyllama",
        "prompt": "Write a poem about spring",
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ vLLM TinyLlama working!")
            print(f"Response: {data['choices'][0]['text'][:100]}...")
            return True
        else:
            print(f"❌ vLLM TinyLlama failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ vLLM TinyLlama error: {str(e)}")
        return False

def test_ollama_tinyllama():
    """Test Ollama TinyLlama endpoint"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "tinyllama:1.1b",
        "prompt": "Write a poem about spring",
        "stream": False,
        "options": {
            "num_predict": 50
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Ollama TinyLlama working!")
            print(f"Response: {data['response'][:100]}...")
            return True
        else:
            print(f"❌ Ollama TinyLlama failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama TinyLlama error: {str(e)}")
        return False

def test_ollama_qwen():
    """Test Ollama Qwen endpoint"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b",
        "prompt": "Write a poem about spring",
        "stream": False,
        "options": {
            "num_predict": 50
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Ollama Qwen working!")
            print(f"Response: {data['response'][:100]}...")
            return True
        else:
            print(f"❌ Ollama Qwen failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama Qwen error: {str(e)}")
        return False

def main():
    print("🚀 Testing LLM Serving Endpoints")
    print("="*50)

    # Test each endpoint
    print("\n1. Testing vLLM TinyLlama...")
    vllm_ok = test_vllm_tinyllama()

    print("\n2. Testing Ollama TinyLlama...")
    ollama_tiny_ok = test_ollama_tinyllama()

    print("\n3. Testing Ollama Qwen...")
    ollama_qwen_ok = test_ollama_qwen()

    # Summary
    print("\n" + "="*50)
    print("Summary:")
    if vllm_ok:
        print("  ✅ vLLM TinyLlama: READY")
    else:
        print("  ❌ vLLM TinyLlama: NOT READY")

    if ollama_tiny_ok:
        print("  ✅ Ollama TinyLlama: READY")
    else:
        print("  ❌ Ollama TinyLlama: NOT READY")

    if ollama_qwen_ok:
        print("  ✅ Ollama Qwen: READY")
    else:
        print("  ❌ Ollama Qwen: NOT READY")

    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()
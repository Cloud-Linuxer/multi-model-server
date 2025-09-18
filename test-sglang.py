#!/usr/bin/env python3
"""
SGLang 테스트 스크립트
SGLang의 고급 기능들을 테스트합니다.
"""

import json
import time
import requests
from typing import List, Dict

# SGLang 엔드포인트
ENDPOINTS = {
    "tinyllama": "http://localhost:30001",
    "qwen": "http://localhost:30002",
    "yi": "http://localhost:30003"
}

def test_basic_completion(endpoint: str, model_name: str):
    """기본 텍스트 완성 테스트"""
    print(f"\n🧪 {model_name} 기본 완성 테스트")

    payload = {
        "text": "The future of AI is",
        "max_new_tokens": 50,
        "temperature": 0.7
    }

    start = time.time()
    response = requests.post(f"{endpoint}/generate", json=payload)
    latency = (time.time() - start) * 1000

    if response.status_code == 200:
        result = response.json()
        print(f"✅ 응답: {result.get('text', '')[:100]}...")
        print(f"⏱️ 지연시간: {latency:.2f}ms")
    else:
        print(f"❌ 오류: {response.status_code}")

def test_structured_generation(endpoint: str, model_name: str):
    """구조화된 생성 테스트 (SGLang 특화 기능)"""
    print(f"\n🧪 {model_name} 구조화된 생성 테스트")

    # JSON 스키마 강제
    payload = {
        "text": "Generate a person profile:",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "regex": r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}'
    }

    start = time.time()
    response = requests.post(f"{endpoint}/generate", json=payload)
    latency = (time.time() - start) * 1000

    if response.status_code == 200:
        result = response.json()
        print(f"✅ 구조화된 응답: {result.get('text', '')}")
        print(f"⏱️ 지연시간: {latency:.2f}ms")
    else:
        print(f"❌ 오류: {response.status_code}")

def test_batch_generation(endpoint: str, model_name: str):
    """배치 생성 테스트"""
    print(f"\n🧪 {model_name} 배치 생성 테스트")

    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "Define artificial intelligence"
    ]

    payload = {
        "text": prompts,
        "max_new_tokens": 30,
        "temperature": 0.5,
        "parallel_sample_num": len(prompts)
    }

    start = time.time()
    response = requests.post(f"{endpoint}/generate", json=payload)
    latency = (time.time() - start) * 1000

    if response.status_code == 200:
        results = response.json()
        print(f"✅ 배치 크기: {len(prompts)}")
        print(f"⏱️ 총 지연시간: {latency:.2f}ms")
        print(f"⏱️ 평균 지연시간: {latency/len(prompts):.2f}ms/요청")
    else:
        print(f"❌ 오류: {response.status_code}")

def test_radix_attention_cache(endpoint: str, model_name: str):
    """RadixAttention 캐시 효과 테스트"""
    print(f"\n🧪 {model_name} RadixAttention 캐시 테스트")

    # 같은 프리픽스로 여러 요청
    prefix = "You are a helpful AI assistant. Please answer the following question: "
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?"
    ]

    latencies = []
    for i, question in enumerate(questions):
        payload = {
            "text": prefix + question,
            "max_new_tokens": 20,
            "temperature": 0.1
        }

        start = time.time()
        response = requests.post(f"{endpoint}/generate", json=payload)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if response.status_code == 200:
            print(f"  요청 {i+1}: {latency:.2f}ms")

    if latencies:
        print(f"✅ 첫 요청: {latencies[0]:.2f}ms")
        print(f"✅ 캐시된 요청 평균: {sum(latencies[1:])/len(latencies[1:]):.2f}ms")
        speedup = latencies[0] / (sum(latencies[1:])/len(latencies[1:]))
        print(f"🚀 속도 향상: {speedup:.2f}x")

def compare_with_vllm():
    """vLLM과 성능 비교"""
    print("\n" + "="*50)
    print("📊 SGLang vs vLLM 성능 비교")
    print("="*50)

    # SGLang 특징
    print("\n✨ SGLang 장점:")
    print("  - RadixAttention: 프리픽스 캐싱으로 최대 10x 속도 향상")
    print("  - 구조화된 생성: JSON/코드 생성 보장")
    print("  - 더 빠른 첫 토큰 시간 (TTFT)")
    print("  - 효율적인 배치 처리")

    print("\n📉 SGLang 단점:")
    print("  - 더 적은 모델 지원")
    print("  - 덜 성숙한 생태계")
    print("  - 일부 기능 제한적")

    print("\n🎯 사용 추천 시나리오:")
    print("  SGLang: API 서빙, 구조화된 출력, 반복적 프롬프트")
    print("  vLLM: 다양한 모델, 안정성 우선, 프로덕션")

def main():
    """메인 테스트 실행"""
    print("🚀 SGLang 멀티모델 테스트 시작")

    for model_name, endpoint in ENDPOINTS.items():
        print(f"\n{'='*50}")
        print(f"📦 {model_name.upper()} 모델 테스트")
        print('='*50)

        try:
            # 헬스 체크
            response = requests.get(f"{endpoint}/health", timeout=2)
            if response.status_code != 200:
                print(f"⏳ {model_name} 아직 준비 안됨")
                continue

            # 테스트 실행
            test_basic_completion(endpoint, model_name)
            test_structured_generation(endpoint, model_name)
            test_batch_generation(endpoint, model_name)
            test_radix_attention_cache(endpoint, model_name)

        except requests.exceptions.RequestException as e:
            print(f"❌ {model_name} 연결 실패: {e}")

    # 비교 분석
    compare_with_vllm()

if __name__ == "__main__":
    main()
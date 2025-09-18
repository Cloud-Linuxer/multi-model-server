#!/usr/bin/env python3
"""
SGLang vs vLLM 성능 비교 테스트
RTX 5090에서 실행
"""

import time
import json
import requests
from typing import Dict, List
import statistics

# 서버 설정
SERVERS = {
    "vLLM": {
        "TinyLlama": "http://localhost:8001/v1/completions",
        "Qwen-3B": "http://localhost:8002/v1/completions",
        "Yi-6B": "http://localhost:8003/v1/completions"
    },
    "SGLang": {
        "TinyLlama": "http://localhost:30001/generate",
        # SGLang은 단일 모델만 실행 가능 (메모리 제한)
    }
}

# 테스트 프롬프트
PROMPTS = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms",
    "Write a Python function to sort a list",
    "What are the benefits of renewable energy?",
    "Describe the water cycle",
]

def test_vllm(url: str, prompt: str, max_tokens: int = 100) -> Dict:
    """vLLM API 테스트"""
    start = time.time()

    payload = {
        "model": "model",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        end = time.time()
        result = response.json()

        return {
            "success": True,
            "latency": (end - start) * 1000,  # ms
            "tokens": len(result["choices"][0]["text"].split()),
            "text": result["choices"][0]["text"][:100] + "..."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": 0,
            "tokens": 0
        }

def test_sglang(url: str, prompt: str, max_tokens: int = 100) -> Dict:
    """SGLang API 테스트"""
    start = time.time()

    payload = {
        "text": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        end = time.time()
        result = response.json()

        return {
            "success": True,
            "latency": (end - start) * 1000,  # ms
            "e2e_latency": result.get("meta_info", {}).get("e2e_latency", 0) * 1000,
            "tokens": result.get("meta_info", {}).get("completion_tokens", 0),
            "text": result.get("text", "")[:100] + "..."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": 0,
            "tokens": 0
        }

def run_benchmark():
    """벤치마크 실행"""
    print("=" * 80)
    print("🚀 SGLang vs vLLM 성능 비교 테스트")
    print("=" * 80)

    results = {}

    # vLLM 테스트
    print("\n📊 vLLM 테스트 중...")
    for model_name, url in SERVERS["vLLM"].items():
        print(f"\n  🔹 {model_name} 테스트...")
        model_results = []

        for prompt in PROMPTS:
            result = test_vllm(url, prompt)
            model_results.append(result)
            if result["success"]:
                print(f"    ✅ Latency: {result['latency']:.2f}ms, Tokens: {result['tokens']}")
            else:
                print(f"    ❌ Error: {result['error']}")

        # 통계 계산
        successful_results = [r for r in model_results if r["success"]]
        if successful_results:
            avg_latency = statistics.mean([r["latency"] for r in successful_results])
            avg_tokens = statistics.mean([r["tokens"] for r in successful_results])

            results[f"vLLM_{model_name}"] = {
                "avg_latency_ms": avg_latency,
                "avg_tokens": avg_tokens,
                "success_rate": len(successful_results) / len(model_results) * 100,
                "total_tests": len(model_results)
            }

            print(f"    📈 평균: {avg_latency:.2f}ms, {avg_tokens:.1f} tokens")

    # SGLang 테스트 (단일 모델만)
    print("\n📊 SGLang 테스트 중...")
    for model_name, url in SERVERS["SGLang"].items():
        print(f"\n  🔹 {model_name} 테스트...")
        model_results = []

        for prompt in PROMPTS:
            result = test_sglang(url, prompt)
            model_results.append(result)
            if result["success"]:
                print(f"    ✅ Latency: {result['latency']:.2f}ms, E2E: {result['e2e_latency']:.2f}ms, Tokens: {result['tokens']}")
            else:
                print(f"    ❌ Error: {result['error']}")

        # 통계 계산
        successful_results = [r for r in model_results if r["success"]]
        if successful_results:
            avg_latency = statistics.mean([r["latency"] for r in successful_results])
            avg_e2e = statistics.mean([r["e2e_latency"] for r in successful_results if r.get("e2e_latency")])
            avg_tokens = statistics.mean([r["tokens"] for r in successful_results])

            results[f"SGLang_{model_name}"] = {
                "avg_latency_ms": avg_latency,
                "avg_e2e_latency_ms": avg_e2e,
                "avg_tokens": avg_tokens,
                "success_rate": len(successful_results) / len(model_results) * 100,
                "total_tests": len(model_results)
            }

            print(f"    📈 평균: {avg_latency:.2f}ms, E2E: {avg_e2e:.2f}ms, {avg_tokens:.1f} tokens")

    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 최종 결과 요약")
    print("=" * 80)

    print("\n📋 성능 비교표:")
    print(f"{'Framework':<20} {'Model':<15} {'Avg Latency (ms)':<20} {'Success Rate':<15}")
    print("-" * 70)

    for key, value in results.items():
        framework, model = key.split("_", 1)
        print(f"{framework:<20} {model:<15} {value['avg_latency_ms']:<20.2f} {value['success_rate']:<15.1f}%")

    # JSON 파일로 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 결과 저장: {filename}")

    # 비교 분석
    print("\n🔍 분석:")

    # TinyLlama 비교 (둘 다 실행 가능한 경우)
    if "vLLM_TinyLlama" in results and "SGLang_TinyLlama" in results:
        vllm_latency = results["vLLM_TinyLlama"]["avg_latency_ms"]
        sglang_latency = results["SGLang_TinyLlama"]["avg_latency_ms"]

        if sglang_latency < vllm_latency:
            improvement = ((vllm_latency - sglang_latency) / vllm_latency) * 100
            print(f"  ✅ SGLang이 vLLM보다 {improvement:.1f}% 빠름 (TinyLlama)")
        else:
            degradation = ((sglang_latency - vllm_latency) / vllm_latency) * 100
            print(f"  ⚠️ SGLang이 vLLM보다 {degradation:.1f}% 느림 (TinyLlama)")

    print("\n📝 주요 발견사항:")
    print("  - vLLM: 3개 모델 동시 실행 가능")
    print("  - SGLang: 메모리 제한으로 단일 모델만 실행 (RTX 5090)")
    print("  - SGLang: RadixAttention 비활성화로 인한 성능 제한")

if __name__ == "__main__":
    # 서버 준비 시간
    print("⏳ 서버 준비 중 (10초 대기)...")
    time.sleep(10)

    # 벤치마크 실행
    run_benchmark()
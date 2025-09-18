#!/usr/bin/env python3
"""
SGLang vs vLLM 단일 모델 성능 비교
TinyLlama 모델만 사용
"""

import time
import json
import requests
import statistics
import subprocess

# 테스트 프롬프트들
PROMPTS = [
    "What is machine learning?",
    "Explain the solar system",
    "How does a computer work?",
    "What causes rain?",
    "Why is the sky blue?",
]

def test_sglang():
    """SGLang TinyLlama 테스트"""
    print("\n" + "="*60)
    print("🚀 SGLang TinyLlama 테스트")
    print("="*60)

    # SGLang 컨테이너 시작
    print("📦 SGLang 컨테이너 시작 중...")
    subprocess.run("docker rm -f sglang-test 2>/dev/null", shell=True)

    subprocess.run([
        "docker", "run", "-d",
        "--name", "sglang-test",
        "--runtime", "nvidia",
        "--gpus", "all",
        "-p", "30001:8000",
        "-v", "~/.cache/huggingface:/root/.cache/huggingface",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--shm-size", "8g",
        "sglang:blackwell-final-v2",
        "--model-path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--mem-fraction-static", "0.85",
        "--max-total-tokens", "2048",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--disable-custom-all-reduce",
        "--disable-flashinfer",
        "--disable-radix-cache",
        "--attention-backend", "torch_native",
        "--dtype", "float16"
    ])

    print("⏳ 서버 초기화 대기 (60초)...")
    time.sleep(60)

    # 테스트 실행
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n테스트 {i}/{len(PROMPTS)}: {prompt[:30]}...")

        payload = {
            "text": prompt,
            "max_new_tokens": 100,
            "temperature": 0.7
        }

        try:
            start = time.time()
            response = requests.post("http://localhost:30001/generate", json=payload, timeout=30)
            end = time.time()

            if response.status_code == 200:
                data = response.json()
                latency = (end - start) * 1000
                e2e_latency = data.get("meta_info", {}).get("e2e_latency", 0) * 1000
                tokens = data.get("meta_info", {}).get("completion_tokens", 0)

                results.append({
                    "latency_ms": latency,
                    "e2e_latency_ms": e2e_latency,
                    "tokens": tokens
                })

                print(f"  ✅ Latency: {latency:.2f}ms, E2E: {e2e_latency:.2f}ms, Tokens: {tokens}")
            else:
                print(f"  ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")

    # 통계
    if results:
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_e2e = statistics.mean([r["e2e_latency_ms"] for r in results])
        avg_tokens = statistics.mean([r["tokens"] for r in results])

        print(f"\n📊 SGLang 평균 성능:")
        print(f"  - Latency: {avg_latency:.2f}ms")
        print(f"  - E2E Latency: {avg_e2e:.2f}ms")
        print(f"  - Tokens: {avg_tokens:.1f}")
    else:
        avg_latency = avg_e2e = avg_tokens = 0

    # 컨테이너 정리
    subprocess.run("docker stop sglang-test", shell=True)

    return avg_latency, avg_e2e, avg_tokens

def test_vllm():
    """vLLM TinyLlama 테스트"""
    print("\n" + "="*60)
    print("🚀 vLLM TinyLlama 테스트")
    print("="*60)

    # vLLM 컨테이너 시작
    print("📦 vLLM 컨테이너 시작 중...")
    subprocess.run("docker rm -f vllm-test 2>/dev/null", shell=True)

    subprocess.run([
        "docker", "run", "-d",
        "--name", "vllm-test",
        "--runtime", "nvidia",
        "--gpus", "all",
        "-p", "8001:8000",
        "-v", "~/.cache/huggingface:/root/.cache/huggingface",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "vllm/vllm-openai:latest",
        "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
        "--enforce-eager",
        "--dtype", "float16"
    ])

    print("⏳ 서버 초기화 대기 (60초)...")
    time.sleep(60)

    # 테스트 실행
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n테스트 {i}/{len(PROMPTS)}: {prompt[:30]}...")

        payload = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }

        try:
            start = time.time()
            response = requests.post("http://localhost:8001/v1/completions", json=payload, timeout=30)
            end = time.time()

            if response.status_code == 200:
                data = response.json()
                latency = (end - start) * 1000
                text = data["choices"][0]["text"]
                tokens = len(text.split())

                results.append({
                    "latency_ms": latency,
                    "tokens": tokens
                })

                print(f"  ✅ Latency: {latency:.2f}ms, Tokens: {tokens}")
            else:
                print(f"  ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")

    # 통계
    if results:
        avg_latency = statistics.mean([r["latency_ms"] for r in results])
        avg_tokens = statistics.mean([r["tokens"] for r in results])

        print(f"\n📊 vLLM 평균 성능:")
        print(f"  - Latency: {avg_latency:.2f}ms")
        print(f"  - Tokens: {avg_tokens:.1f}")
    else:
        avg_latency = avg_tokens = 0

    # 컨테이너 정리
    subprocess.run("docker stop vllm-test", shell=True)

    return avg_latency, avg_tokens

def main():
    """메인 함수"""
    print("🔬 SGLang vs vLLM 성능 비교 테스트")
    print("모델: TinyLlama 1.1B")
    print("GPU: RTX 5090")

    # SGLang 테스트
    sglang_latency, sglang_e2e, sglang_tokens = test_sglang()

    print("\n⏳ 10초 대기...")
    time.sleep(10)

    # vLLM 테스트
    vllm_latency, vllm_tokens = test_vllm()

    # 최종 비교
    print("\n" + "="*60)
    print("📊 최종 비교 결과")
    print("="*60)

    print(f"\n{'Framework':<15} {'Avg Latency':<15} {'Avg Tokens':<15}")
    print("-" * 45)
    print(f"{'SGLang':<15} {f'{sglang_latency:.2f}ms':<15} {f'{sglang_tokens:.1f}':<15}")
    print(f"{'vLLM':<15} {f'{vllm_latency:.2f}ms':<15} {f'{vllm_tokens:.1f}':<15}")

    if sglang_latency > 0 and vllm_latency > 0:
        if sglang_latency < vllm_latency:
            improvement = ((vllm_latency - sglang_latency) / vllm_latency) * 100
            print(f"\n✅ SGLang이 vLLM보다 {improvement:.1f}% 빠름")
        else:
            degradation = ((sglang_latency - vllm_latency) / vllm_latency) * 100
            print(f"\n⚠️ SGLang이 vLLM보다 {degradation:.1f}% 느림")

    print("\n📝 주요 발견사항:")
    print("  - SGLang: RadixAttention 비활성화 상태")
    print("  - vLLM: PagedAttention 사용")
    print("  - 둘 다 RTX 5090에서 제한적 지원")

    # 결과 저장
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "TinyLlama-1.1B",
        "gpu": "RTX 5090",
        "sglang": {
            "avg_latency_ms": sglang_latency,
            "avg_e2e_latency_ms": sglang_e2e,
            "avg_tokens": sglang_tokens
        },
        "vllm": {
            "avg_latency_ms": vllm_latency,
            "avg_tokens": vllm_tokens
        }
    }

    filename = f"benchmark_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 결과 저장: {filename}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
vLLM only benchmark with same conditions as SGLang
"""

import time
import json
import requests
import csv
from datetime import datetime
import pandas as pd
import numpy as np

# 다국어 프롬프트 - SGLang과 동일
PROMPTS = {
    "english": "Sing a beautiful song about spring with blooming flowers.",
    "chinese": "唱一首关于春天百花盛开之美的歌曲。",
    "korean": "봄에 피어나는 꽃들의 아름다움을 노래해주세요."
}

NUM_ITERATIONS = 100  # SGLang과 동일한 테스트 횟수

def test_vllm(prompt, language, iteration):
    """vLLM 테스트"""
    url = "http://localhost:40001/v1/completions"
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms

        result = response.json()
        tokens = result["usage"]["completion_tokens"]

        # TTFT 근사치 (전체 시간의 일부로 추정)
        ttft_estimate = total_time * 0.1  # 첫 토큰까지 약 10%

        return {
            "framework": "vLLM",
            "language": language,
            "iteration": iteration,
            "success": True,
            "latency_ms": total_time,
            "ttft_ms": ttft_estimate,
            "tokens": tokens,
            "tokens_per_second": tokens / (total_time / 1000) if total_time > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "framework": "vLLM",
            "language": language,
            "iteration": iteration,
            "success": False,
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }

def main():
    print("=" * 70)
    print("🚀 vLLM Multilingual Benchmark")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  GPU: NVIDIA RTX 5090 (32GB)")
    print(f"📝 Languages: English, Chinese, Korean")
    print(f"🔢 Iterations per language: {NUM_ITERATIONS}")
    print(f"📊 Total tests: {NUM_ITERATIONS * 3}")
    print("=" * 70)

    # GPU 메모리 확인
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        used, free = map(int, result.stdout.strip().split(", "))
        print(f"\n📊 GPU Memory: {used}MB used, {free}MB free")

    results = []

    print("\n📝 Testing vLLM...")

    for language, prompt in PROMPTS.items():
        print(f"\n  Testing {language}: ", end="", flush=True)
        success_count = 0

        for i in range(NUM_ITERATIONS):
            result = test_vllm(prompt, language, i)
            results.append(result)

            if result["success"]:
                success_count += 1

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

            time.sleep(0.05)  # 요청 간 간격

        print(f" [{success_count}/{NUM_ITERATIONS} success]")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"/home/multi-model-server/vllm_benchmark_{timestamp}.csv"

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Results saved to {csv_file}")

    # 분석
    print("\n" + "=" * 70)
    print("📊 vLLM PERFORMANCE ANALYSIS")
    print("=" * 70)

    success_df = df[df['success'] == True]

    print(f"\n📈 Overall Statistics:")
    print(f"  Total tests: {len(df)}")
    print(f"  Successful: {len(success_df)}")
    print(f"  Success rate: {len(success_df)/len(df)*100:.1f}%")

    if len(success_df) > 0:
        print(f"\n⚡ Latency Analysis:")
        print(f"  Mean: {success_df['latency_ms'].mean():.2f}ms")
        print(f"  Median (P50): {success_df['latency_ms'].median():.2f}ms")
        print(f"  P75: {success_df['latency_ms'].quantile(0.75):.2f}ms")
        print(f"  P90: {success_df['latency_ms'].quantile(0.90):.2f}ms")
        print(f"  P95: {success_df['latency_ms'].quantile(0.95):.2f}ms")
        print(f"  P99: {success_df['latency_ms'].quantile(0.99):.2f}ms")
        print(f"  Min: {success_df['latency_ms'].min():.2f}ms")
        print(f"  Max: {success_df['latency_ms'].max():.2f}ms")
        print(f"  Std Dev: {success_df['latency_ms'].std():.2f}ms")

        print(f"\n🚀 Throughput Analysis:")
        print(f"  Mean: {success_df['tokens_per_second'].mean():.2f} tok/s")
        print(f"  Max: {success_df['tokens_per_second'].max():.2f} tok/s")
        print(f"  Min: {success_df['tokens_per_second'].min():.2f} tok/s")

        print(f"\n🌍 Language-Specific Performance:")
        for lang in ["english", "chinese", "korean"]:
            lang_df = success_df[success_df['language'] == lang]
            if len(lang_df) > 0:
                print(f"\n  {lang.capitalize()}:")
                print(f"    Success rate: {len(lang_df)}/{NUM_ITERATIONS} ({len(lang_df)/NUM_ITERATIONS*100:.1f}%)")
                print(f"    Avg latency: {lang_df['latency_ms'].mean():.2f}ms")
                print(f"    P50 latency: {lang_df['latency_ms'].median():.2f}ms")
                print(f"    P95 latency: {lang_df['latency_ms'].quantile(0.95):.2f}ms")
                print(f"    Avg throughput: {lang_df['tokens_per_second'].mean():.2f} tok/s")
                print(f"    Avg tokens: {lang_df['tokens'].mean():.1f}")

    # SGLang 비교 (이전 결과 참조)
    print("\n" + "=" * 70)
    print("📊 COMPARISON WITH SGLANG")
    print("=" * 70)

    print("\n📝 Previous SGLang Results (Reference):")
    print("  Success rate: 100% (300/300)")
    print("  Avg latency: 393.94ms")
    print("  P50 latency: 526.71ms")
    print("  P95 latency: 538.42ms")
    print("  Avg throughput: 69.38 tok/s")

    if len(success_df) > 0:
        print("\n🏆 vLLM Current Results:")
        print(f"  Success rate: {len(success_df)/len(df)*100:.1f}%")
        print(f"  Avg latency: {success_df['latency_ms'].mean():.2f}ms")
        print(f"  P50 latency: {success_df['latency_ms'].median():.2f}ms")
        print(f"  P95 latency: {success_df['latency_ms'].quantile(0.95):.2f}ms")
        print(f"  Avg throughput: {success_df['tokens_per_second'].mean():.2f} tok/s")

        # 비교 분석
        vllm_lat = success_df['latency_ms'].mean()
        sglang_lat = 393.94
        vllm_thr = success_df['tokens_per_second'].mean()
        sglang_thr = 69.38

        print("\n⚖️  Performance Comparison:")
        if vllm_lat < sglang_lat:
            improvement = ((sglang_lat - vllm_lat) / sglang_lat) * 100
            print(f"  Latency: vLLM is {improvement:.1f}% faster")
        else:
            improvement = ((vllm_lat - sglang_lat) / sglang_lat) * 100
            print(f"  Latency: vLLM is {improvement:.1f}% slower")

        if vllm_thr > sglang_thr:
            improvement = ((vllm_thr - sglang_thr) / sglang_thr) * 100
            print(f"  Throughput: vLLM is {improvement:.1f}% higher")
        else:
            improvement = ((sglang_thr - vllm_thr) / sglang_thr) * 100
            print(f"  Throughput: vLLM is {improvement:.1f}% lower")

    print("\n" + "=" * 70)
    print("✅ Benchmark completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
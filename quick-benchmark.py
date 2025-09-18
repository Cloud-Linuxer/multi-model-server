#!/usr/bin/env python3
"""
Quick multilingual benchmark for SGLang vs vLLM
"""

import time
import json
import requests
import csv
from datetime import datetime
import subprocess
import pandas as pd

# 다국어 프롬프트
PROMPTS = {
    "english": "Sing a beautiful song about spring with blooming flowers.",
    "chinese": "唱一首关于春天百花盛开之美的歌曲。",
    "korean": "봄에 피어나는 꽃들의 아름다움을 노래해주세요."
}

NUM_ITERATIONS = 100  # 각 언어당 100번 = 총 300번 테스트 per framework

def test_sglang(prompt, language, iteration):
    """SGLang 테스트"""
    url = "http://localhost:30001/generate"
    payload = {
        "text": prompt,
        "max_new_tokens": 100,
        "temperature": 0.7
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        total_time = (time.perf_counter() - start_time) * 1000

        result = response.json()
        text = result.get("text", "")
        tokens = len(text.split())

        return {
            "framework": "SGLang",
            "language": language,
            "iteration": iteration,
            "success": True,
            "latency_ms": total_time,
            "tokens": tokens,
            "tokens_per_second": tokens / (total_time / 1000) if total_time > 0 else 0
        }
    except Exception as e:
        return {
            "framework": "SGLang",
            "language": language,
            "iteration": iteration,
            "success": False,
            "error": str(e)
        }

def test_vllm(prompt, language, iteration):
    """vLLM 테스트"""
    url = "http://localhost:40001/v1/completions"
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        total_time = (time.perf_counter() - start_time) * 1000

        result = response.json()
        tokens = result["usage"]["completion_tokens"]

        return {
            "framework": "vLLM",
            "language": language,
            "iteration": iteration,
            "success": True,
            "latency_ms": total_time,
            "tokens": tokens,
            "tokens_per_second": tokens / (total_time / 1000) if total_time > 0 else 0
        }
    except Exception as e:
        return {
            "framework": "vLLM",
            "language": language,
            "iteration": iteration,
            "success": False,
            "error": str(e)[:100]
        }

def main():
    print("🚀 Quick Multilingual Benchmark: SGLang vs vLLM")
    print(f"Languages: English, Chinese, Korean")
    print(f"Iterations per language: {NUM_ITERATIONS}")
    print(f"Total tests per framework: {NUM_ITERATIONS * 3}")
    print("="*60)

    results = []

    # SGLang 테스트
    print("\n📝 Testing SGLang...")
    # SGLang이 이미 실행 중이어야 함

    for language, prompt in PROMPTS.items():
        print(f"  {language}: ", end="", flush=True)
        for i in range(NUM_ITERATIONS):
            result = test_sglang(prompt, language, i)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
            time.sleep(0.05)
        print(f" ✓")

    # vLLM 테스트를 위해 SGLang 종료하고 vLLM 시작
    print("\n🔄 Switching to vLLM...")
    subprocess.run(["docker", "stop", "sglang-tinyllama"], capture_output=True)
    time.sleep(5)

    # vLLM 시작 (이미 실행 중이어야 함)
    print("⏳ Waiting for vLLM to be ready...")
    time.sleep(45)

    print("\n📝 Testing vLLM...")

    for language, prompt in PROMPTS.items():
        print(f"  {language}: ", end="", flush=True)
        for i in range(NUM_ITERATIONS):
            result = test_vllm(prompt, language, i)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
            time.sleep(0.05)
        print(f" ✓")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"/home/multi-model-server/quick_benchmark_{timestamp}.csv"

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Results saved to {csv_file}")

    # 분석
    print("\n" + "="*60)
    print("📊 RESULTS ANALYSIS")
    print("="*60)

    for framework in ["SGLang", "vLLM"]:
        fdf = df[df['framework'] == framework]
        success_df = fdf[fdf['success'] == True]

        print(f"\n{framework}:")
        print(f"  Success rate: {len(success_df)}/{len(fdf)} ({len(success_df)/len(fdf)*100:.1f}%)")

        if len(success_df) > 0:
            print(f"  Avg latency: {success_df['latency_ms'].mean():.2f}ms")
            print(f"  P50 latency: {success_df['latency_ms'].median():.2f}ms")
            print(f"  P95 latency: {success_df['latency_ms'].quantile(0.95):.2f}ms")
            print(f"  Avg throughput: {success_df['tokens_per_second'].mean():.2f} tok/s")

            print(f"\n  By language:")
            for lang in ["english", "chinese", "korean"]:
                lang_df = success_df[success_df['language'] == lang]
                if len(lang_df) > 0:
                    print(f"    {lang}: {lang_df['latency_ms'].mean():.2f}ms, {lang_df['tokens_per_second'].mean():.2f} tok/s")

    # 최종 비교
    sg_success = df[(df['framework'] == 'SGLang') & (df['success'] == True)]
    vl_success = df[(df['framework'] == 'vLLM') & (df['success'] == True)]

    if len(sg_success) > 0 and len(vl_success) > 0:
        print("\n" + "="*60)
        print("🏆 WINNER ANALYSIS")
        print("="*60)

        sg_lat = sg_success['latency_ms'].mean()
        vl_lat = vl_success['latency_ms'].mean()
        sg_thr = sg_success['tokens_per_second'].mean()
        vl_thr = vl_success['tokens_per_second'].mean()

        if sg_lat < vl_lat:
            print(f"⚡ Latency: SGLang wins ({sg_lat:.2f}ms vs {vl_lat:.2f}ms)")
        else:
            print(f"⚡ Latency: vLLM wins ({vl_lat:.2f}ms vs {sg_lat:.2f}ms)")

        if sg_thr > vl_thr:
            print(f"🚀 Throughput: SGLang wins ({sg_thr:.2f} vs {vl_thr:.2f} tok/s)")
        else:
            print(f"🚀 Throughput: vLLM wins ({vl_thr:.2f} vs {sg_thr:.2f} tok/s)")

if __name__ == "__main__":
    main()
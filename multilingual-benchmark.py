#!/usr/bin/env python3
"""
Multilingual Performance Benchmark for SGLang vs vLLM
Tests with Chinese, English, and Korean prompts about seasons
"""

import time
import json
import requests
import csv
import statistics
import numpy as np
from datetime import datetime
import subprocess
import sys
from typing import List, Dict, Any
import pandas as pd

# 다국어 프롬프트 - 계절을 노래하라
PROMPTS = {
    "english": [
        "Sing a song about the beauty of spring with blooming flowers.",
        "Create a melody about the warm summer days at the beach.",
        "Compose lyrics about autumn leaves falling gently.",
        "Write a song about winter snowflakes dancing in the air.",
        "Sing about the four seasons changing throughout the year."
    ],
    "chinese": [
        "唱一首关于春天百花盛开之美的歌曲。",
        "创作一首关于夏日海滩温暖时光的旋律。",
        "谱写关于秋叶轻轻飘落的歌词。",
        "写一首关于冬天雪花在空中飞舞的歌。",
        "歌唱四季更替的美妙变化。"
    ],
    "korean": [
        "봄에 피어나는 꽃들의 아름다움을 노래해주세요.",
        "해변에서 보내는 따뜻한 여름날의 멜로디를 만들어주세요.",
        "가을 낙엽이 부드럽게 떨어지는 것에 대한 가사를 써주세요.",
        "겨울 눈송이가 공중에서 춤추는 것에 대한 노래를 써주세요.",
        "일년 내내 변화하는 사계절을 노래해주세요."
    ]
}

# 테스트 설정
NUM_ITERATIONS = 10  # 각 프롬프트당 테스트 횟수 (총 10 * 15 = 150 tests per framework) - 빠른 테스트용
OUTPUT_TOKENS = 100   # 생성할 토큰 수

class BenchmarkRunner:
    def __init__(self):
        self.results = []

    def test_sglang(self, prompt: str, language: str, prompt_idx: int, iteration: int) -> Dict[str, Any]:
        """SGLang API 테스트"""
        url = "http://localhost:30001/generate"
        payload = {
            "text": prompt,
            "max_new_tokens": OUTPUT_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9
        }

        # Time to First Token (TTFT) 측정
        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            first_byte_time = time.perf_counter()

            result = response.json()
            end_time = time.perf_counter()

            generated_text = result.get("text", "")
            tokens = len(generated_text.split())

            ttft = (first_byte_time - start_time) * 1000  # ms
            total_latency = (end_time - start_time) * 1000  # ms
            generation_latency = total_latency - ttft

            return {
                "framework": "SGLang",
                "language": language,
                "prompt_idx": prompt_idx,
                "iteration": iteration,
                "success": True,
                "ttft_ms": ttft,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency,
                "output_tokens": tokens,
                "tokens_per_second": tokens / (total_latency / 1000) if total_latency > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "framework": "SGLang",
                "language": language,
                "prompt_idx": prompt_idx,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_vllm(self, prompt: str, language: str, prompt_idx: int, iteration: int) -> Dict[str, Any]:
        """vLLM API 테스트"""
        url = "http://localhost:40001/v1/completions"
        payload = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt,
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            first_byte_time = time.perf_counter()

            result = response.json()
            end_time = time.perf_counter()

            generated_text = result["choices"][0]["text"]
            tokens = result["usage"]["completion_tokens"]

            ttft = (first_byte_time - start_time) * 1000  # ms
            total_latency = (end_time - start_time) * 1000  # ms
            generation_latency = total_latency - ttft

            return {
                "framework": "vLLM",
                "language": language,
                "prompt_idx": prompt_idx,
                "iteration": iteration,
                "success": True,
                "ttft_ms": ttft,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency,
                "output_tokens": tokens,
                "tokens_per_second": tokens / (total_latency / 1000) if total_latency > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "framework": "vLLM",
                "language": language,
                "prompt_idx": prompt_idx,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def run_benchmark(self, framework: str) -> List[Dict]:
        """벤치마크 실행"""
        print(f"\n{'='*80}")
        print(f"🚀 {framework} Multilingual Benchmark Starting")
        print(f"Languages: Chinese, English, Korean")
        print(f"Iterations per prompt: {NUM_ITERATIONS}")
        print(f"{'='*80}\n")

        results = []
        total_tests = len(PROMPTS) * 5 * NUM_ITERATIONS  # 3 languages * 5 prompts * iterations
        current_test = 0

        for language, prompt_list in PROMPTS.items():
            print(f"\n📝 Testing {language.upper()} prompts...")

            for prompt_idx, prompt in enumerate(prompt_list):
                print(f"  Prompt {prompt_idx + 1}: ", end="")
                successful_tests = 0

                for iteration in range(NUM_ITERATIONS):
                    current_test += 1

                    if framework == "SGLang":
                        result = self.test_sglang(prompt, language, prompt_idx, iteration)
                    else:
                        result = self.test_vllm(prompt, language, prompt_idx, iteration)

                    results.append(result)

                    if result["success"]:
                        successful_tests += 1

                    # Progress indicator
                    if (iteration + 1) % 20 == 0:
                        print(f".", end="", flush=True)

                    # Small delay between requests
                    time.sleep(0.05)

                print(f" [{successful_tests}/{NUM_ITERATIONS} success]")

        print(f"\n✅ {framework} benchmark completed: {len(results)} tests")
        return results

    def save_to_csv(self, results: List[Dict], filename: str):
        """결과를 CSV 파일로 저장"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
        return df

    def analyze_results(self, sglang_df: pd.DataFrame, vllm_df: pd.DataFrame) -> Dict:
        """결과 분석"""
        analysis = {}

        for framework, df in [("SGLang", sglang_df), ("vLLM", vllm_df)]:
            # 성공한 테스트만 분석
            success_df = df[df['success'] == True]

            analysis[framework] = {
                "total_tests": len(df),
                "successful_tests": len(success_df),
                "success_rate": len(success_df) / len(df) * 100,

                "overall": {
                    "avg_ttft_ms": success_df['ttft_ms'].mean(),
                    "p50_ttft_ms": success_df['ttft_ms'].median(),
                    "p95_ttft_ms": success_df['ttft_ms'].quantile(0.95),
                    "p99_ttft_ms": success_df['ttft_ms'].quantile(0.99),

                    "avg_total_latency_ms": success_df['total_latency_ms'].mean(),
                    "p50_total_latency_ms": success_df['total_latency_ms'].median(),
                    "p95_total_latency_ms": success_df['total_latency_ms'].quantile(0.95),
                    "p99_total_latency_ms": success_df['total_latency_ms'].quantile(0.99),

                    "avg_tokens_per_second": success_df['tokens_per_second'].mean(),
                    "max_tokens_per_second": success_df['tokens_per_second'].max(),
                    "min_tokens_per_second": success_df['tokens_per_second'].min(),
                },

                "by_language": {}
            }

            # 언어별 분석
            for language in ['english', 'chinese', 'korean']:
                lang_df = success_df[success_df['language'] == language]
                if len(lang_df) > 0:
                    analysis[framework]["by_language"][language] = {
                        "count": len(lang_df),
                        "avg_ttft_ms": lang_df['ttft_ms'].mean(),
                        "avg_latency_ms": lang_df['total_latency_ms'].mean(),
                        "avg_tokens_per_second": lang_df['tokens_per_second'].mean(),
                    }

        return analysis

def main():
    print("\n" + "="*80)
    print("🔬 Multilingual Performance Benchmark: SGLang vs vLLM")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🖥️  GPU: NVIDIA RTX 5090 (32GB)")
    print("📝 Languages: Chinese, English, Korean")
    print(f"🔢 Tests per prompt: {NUM_ITERATIONS}")
    print(f"📊 Total tests per framework: {NUM_ITERATIONS * 15}")
    print("="*80)

    runner = BenchmarkRunner()

    # 기존 컨테이너 정리하고 새로 시작
    print("\n🔄 Setting up test environment...")
    subprocess.run(["docker", "stop", "$(docker ps -q)"], shell=True, capture_output=True)
    time.sleep(5)

    # SGLang 시작
    print("Starting SGLang container...")
    subprocess.run(["/home/multi-model-server/deploy-sglang-single.sh"], capture_output=True)
    print("⏳ Waiting for SGLang initialization (45s)...")
    time.sleep(45)

    # SGLang 벤치마크
    print("\n" + "="*80)
    print("PHASE 1: SGLang Benchmark")
    print("="*80)
    sglang_results = runner.run_benchmark("SGLang")

    # SGLang 종료, vLLM 시작
    print("\n🔄 Switching to vLLM...")
    subprocess.run(["docker", "stop", "sglang-tinyllama"], capture_output=True)
    time.sleep(5)

    subprocess.run(["bash", "-c", """
        docker run -d \
          --name vllm-benchmark \
          --runtime nvidia \
          --gpus all \
          -p 40001:8000 \
          -v /root/.cache/huggingface:/root/.cache/huggingface \
          --shm-size 8g \
          vllm/vllm-openai:latest \
          --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
          --host 0.0.0.0 \
          --port 8000 \
          --gpu-memory-utilization 0.05 \
          --max-model-len 2048 \
          --enforce-eager
    """], capture_output=True)

    print("⏳ Waiting for vLLM initialization (45s)...")
    time.sleep(45)

    # vLLM 벤치마크
    print("\n" + "="*80)
    print("PHASE 2: vLLM Benchmark")
    print("="*80)
    vllm_results = runner.run_benchmark("vLLM")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sglang_csv = f"/home/multi-model-server/benchmark_sglang_{timestamp}.csv"
    vllm_csv = f"/home/multi-model-server/benchmark_vllm_{timestamp}.csv"

    print("\n" + "="*80)
    print("📊 Saving Results")
    print("="*80)

    sglang_df = runner.save_to_csv(sglang_results, sglang_csv)
    vllm_df = runner.save_to_csv(vllm_results, vllm_csv)

    # 분석
    print("\n" + "="*80)
    print("📈 Analyzing Results")
    print("="*80)

    analysis = runner.analyze_results(sglang_df, vllm_df)

    # 분석 결과 출력
    for framework in ["SGLang", "vLLM"]:
        print(f"\n🏷️  {framework} Results:")
        print(f"  ✅ Success Rate: {analysis[framework]['success_rate']:.1f}%")
        print(f"  ⚡ Avg TTFT: {analysis[framework]['overall']['avg_ttft_ms']:.2f}ms")
        print(f"  ⚡ P50 TTFT: {analysis[framework]['overall']['p50_ttft_ms']:.2f}ms")
        print(f"  ⚡ P95 TTFT: {analysis[framework]['overall']['p95_ttft_ms']:.2f}ms")
        print(f"  📊 Avg Throughput: {analysis[framework]['overall']['avg_tokens_per_second']:.1f} tokens/s")

        print(f"\n  By Language:")
        for lang in ['english', 'chinese', 'korean']:
            if lang in analysis[framework]['by_language']:
                stats = analysis[framework]['by_language'][lang]
                print(f"    {lang.capitalize()}:")
                print(f"      - Avg TTFT: {stats['avg_ttft_ms']:.2f}ms")
                print(f"      - Avg Latency: {stats['avg_latency_ms']:.2f}ms")
                print(f"      - Throughput: {stats['avg_tokens_per_second']:.1f} tok/s")

    # 분석 결과 JSON 저장
    analysis_file = f"/home/multi-model-server/benchmark_analysis_{timestamp}.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n💾 Analysis saved to {analysis_file}")

    # 정리
    subprocess.run(["docker", "stop", "vllm-benchmark"], capture_output=True)

    print("\n✅ Benchmark completed successfully!")

    return analysis

if __name__ == "__main__":
    main()
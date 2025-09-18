#!/usr/bin/env python3
"""
SGLang vs vLLM 벤치마크 스크립트
RTX 5090 환경에서 성능 비교
"""

import time
import json
import requests
import statistics
from typing import List, Dict, Any
from datetime import datetime
import subprocess
import sys

# 테스트 프롬프트 설정
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "What are the benefits of exercise?",
    "How does photosynthesis work?",
    "Describe the process of machine learning.",
    "What is the difference between CPU and GPU?",
    "Explain the concept of blockchain.",
    "How do neural networks work?",
    "What causes climate change?"
]

# 다양한 길이의 출력 테스트
OUTPUT_LENGTHS = [32, 64, 128, 256]

def test_sglang(prompt: str, max_tokens: int) -> Dict[str, Any]:
    """SGLang API 테스트"""
    url = "http://localhost:30001/generate"
    payload = {
        "text": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        latency = time.time() - start_time

        result = response.json()
        generated_text = result.get("text", "")
        tokens = len(generated_text.split())  # 대략적인 토큰 수

        return {
            "success": True,
            "latency": latency,
            "tokens": tokens,
            "tokens_per_second": tokens / latency if latency > 0 else 0,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "tokens": 0,
            "tokens_per_second": 0,
            "error": str(e)
        }

def test_vllm(prompt: str, max_tokens: int, port: int = 40001) -> Dict[str, Any]:
    """vLLM API 테스트"""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 모델 이름 명시
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        latency = time.time() - start_time

        result = response.json()
        generated_text = result["choices"][0]["text"]
        tokens = result["usage"]["completion_tokens"]

        return {
            "success": True,
            "latency": latency,
            "tokens": tokens,
            "tokens_per_second": tokens / latency if latency > 0 else 0,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "tokens": 0,
            "tokens_per_second": 0,
            "error": str(e)
        }

def run_benchmark(framework: str, prompts: List[str], output_lengths: List[int]) -> Dict[str, Any]:
    """벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"🚀 {framework} 벤치마크 시작")
    print(f"{'='*60}")

    results = []
    total_tests = len(prompts) * len(output_lengths)
    current_test = 0

    for length in output_lengths:
        print(f"\n📊 Output Length: {length} tokens")
        for i, prompt in enumerate(prompts):
            current_test += 1
            print(f"  [{current_test}/{total_tests}] Testing prompt {i+1}...", end="")

            if framework == "SGLang":
                result = test_sglang(prompt, length)
            else:  # vLLM
                result = test_vllm(prompt, length)

            result["output_length"] = length
            result["prompt_index"] = i
            results.append(result)

            if result["success"]:
                print(f" ✅ {result['latency']:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
            else:
                print(f" ❌ Failed: {result['error']}")

            # 요청 간 간격
            time.sleep(0.5)

    # 통계 계산
    successful_results = [r for r in results if r["success"]]

    if successful_results:
        latencies = [r["latency"] for r in successful_results]
        tps_values = [r["tokens_per_second"] for r in successful_results]

        stats = {
            "framework": framework,
            "total_tests": total_tests,
            "successful_tests": len(successful_results),
            "failed_tests": total_tests - len(successful_results),
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p50_latency": statistics.median(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
            "avg_tokens_per_second": statistics.mean(tps_values),
            "max_tokens_per_second": max(tps_values),
            "raw_results": results
        }
    else:
        stats = {
            "framework": framework,
            "total_tests": total_tests,
            "successful_tests": 0,
            "failed_tests": total_tests,
            "error": "All tests failed",
            "raw_results": results
        }

    return stats

def get_gpu_memory():
    """GPU 메모리 사용량 확인"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, free = map(int, result.stdout.strip().split(", "))
            return {"used_mb": used, "free_mb": free}
    except:
        pass
    return {"used_mb": 0, "free_mb": 0}

def main():
    print("\n" + "="*80)
    print("🔬 SGLang vs vLLM Performance Benchmark")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🖥️  GPU: NVIDIA RTX 5090 (32GB)")
    print("📝 Model: TinyLlama 1.1B")
    print("="*80)

    # GPU 메모리 상태
    gpu_mem = get_gpu_memory()
    print(f"\n📊 GPU Memory: {gpu_mem['used_mb']}MB used, {gpu_mem['free_mb']}MB free")

    # 컨테이너 상태 확인
    print("\n🐳 Container Status:")
    subprocess.run(["docker", "ps", "--filter", "name=sglang", "--filter", "name=vllm"])

    results = {}

    # SGLang 테스트
    print("\n▶️  Starting SGLang benchmark...")
    time.sleep(2)
    results["sglang"] = run_benchmark("SGLang", TEST_PROMPTS, OUTPUT_LENGTHS)

    # SGLang 종료하고 vLLM 시작
    print("\n🔄 Switching from SGLang to vLLM...")
    subprocess.run(["docker", "stop", "sglang-tinyllama"], capture_output=True)
    time.sleep(5)
    subprocess.run(["/home/multi-model-server/deploy-vllm-3models.sh"], capture_output=True)
    print("⏳ Waiting for vLLM to initialize (60s)...")
    time.sleep(60)

    # vLLM 테스트
    print("\n▶️  Starting vLLM benchmark...")
    time.sleep(2)
    results["vllm"] = run_benchmark("vLLM", TEST_PROMPTS, OUTPUT_LENGTHS)

    # 결과 비교
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS COMPARISON")
    print("="*80)

    for framework in ["sglang", "vllm"]:
        if framework in results:
            r = results[framework]
            print(f"\n🏷️  {r['framework']}:")
            print(f"  ✅ Success Rate: {r['successful_tests']}/{r['total_tests']} ({r['successful_tests']/r['total_tests']*100:.1f}%)")
            if r['successful_tests'] > 0:
                print(f"  ⚡ Avg Latency: {r['avg_latency']:.3f}s")
                print(f"  ⚡ P50 Latency: {r['p50_latency']:.3f}s")
                print(f"  ⚡ P95 Latency: {r['p95_latency']:.3f}s")
                print(f"  📈 Avg Throughput: {r['avg_tokens_per_second']:.1f} tokens/s")
                print(f"  📈 Max Throughput: {r['max_tokens_per_second']:.1f} tokens/s")

    # 우승자 결정
    if results.get("sglang", {}).get("successful_tests", 0) > 0 and results.get("vllm", {}).get("successful_tests", 0) > 0:
        sglang_tps = results["sglang"]["avg_tokens_per_second"]
        vllm_tps = results["vllm"]["avg_tokens_per_second"]

        print("\n" + "="*80)
        print("🏆 WINNER ANALYSIS")
        print("="*80)

        if sglang_tps > vllm_tps:
            improvement = ((sglang_tps - vllm_tps) / vllm_tps) * 100
            print(f"  🥇 SGLang is {improvement:.1f}% faster than vLLM")
        else:
            improvement = ((vllm_tps - sglang_tps) / sglang_tps) * 100
            print(f"  🥇 vLLM is {improvement:.1f}% faster than SGLang")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/multi-model-server/benchmark_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {filename}")

    # 멀티모델 서빙 능력 비교
    print("\n" + "="*80)
    print("🎯 MULTI-MODEL SERVING CAPABILITY")
    print("="*80)
    print("  vLLM: ✅ Can run 3 models simultaneously (TinyLlama, Qwen, Yi)")
    print("  SGLang: ❌ Can only run 1 model at a time on RTX 5090")
    print("\n📝 Note: SGLang's limitation is due to memory management on RTX 5090")
    print("  Each SGLang container sees the full GPU and cannot coordinate memory")

if __name__ == "__main__":
    main()
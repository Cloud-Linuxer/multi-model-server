#!/usr/bin/env python3
"""
SGLang Performance Test Script
Tests actual throughput and latency with multiple languages
"""
import requests
import time
import json
import statistics
from datetime import datetime

# Test prompts in different languages
prompts = {
    "english": "Write a story about a robot discovering emotions for the first time",
    "chinese": "写一个关于机器人第一次发现情感的故事",
    "korean": "로봇이 처음으로 감정을 발견하는 이야기를 써주세요"
}

def test_sglang(prompt, language, iteration, port=30000):
    """Test SGLang with given prompt"""
    try:
        url = f"http://localhost:{port}/generate"

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 100
            }
        }

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            latency_ms = (end_time - start_time) * 1000

            # Get generated text and count tokens (approximate)
            generated_text = result.get("text", "")
            # Simple token counting (word-based approximation)
            tokens = len(generated_text.split())

            throughput = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

            return {
                "success": True,
                "language": language,
                "iteration": iteration,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens,
                "throughput_tok_s": round(throughput, 2),
                "response_preview": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            }
        else:
            return {
                "success": False,
                "language": language,
                "iteration": iteration,
                "error": f"HTTP {response.status_code}: {response.text[:100]}"
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "language": language,
            "iteration": iteration,
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "success": False,
            "language": language,
            "iteration": iteration,
            "error": str(e)
        }

def run_benchmark(iterations=10):
    """Run complete benchmark"""
    results = []

    print(f"\n{'='*60}")
    print(f"SGLang Benchmark Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Check if SGLang is accessible
    try:
        response = requests.get("http://localhost:30000/get_model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ SGLang server is running")
            print(f"📦 Model: {model_info.get('model_path', 'Unknown')}")
            print(f"{'='*60}\n")
        else:
            print("⚠️ SGLang server responded but may not be ready")
    except:
        print("❌ SGLang server is not accessible on port 30000")
        print("Please start SGLang first with:")
        print("python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 30000")
        return None

    # Run tests for each language
    for language, prompt in prompts.items():
        print(f"\n🌍 Testing {language.upper()}...")

        language_latencies = []
        language_throughputs = []

        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}...", end="")
            result = test_sglang(prompt, language, i)
            results.append(result)

            if result["success"]:
                language_latencies.append(result["latency_ms"])
                language_throughputs.append(result["throughput_tok_s"])
                print(f" ✓ {result['latency_ms']}ms, {result['throughput_tok_s']} tok/s")
            else:
                print(f" ✗ {result['error']}")

        # Print language statistics
        if language_latencies:
            print(f"\n  📊 {language.upper()} Statistics:")
            print(f"     Avg Latency: {statistics.mean(language_latencies):.2f}ms")
            print(f"     Avg Throughput: {statistics.mean(language_throughputs):.2f} tok/s")
            print(f"     Min Latency: {min(language_latencies):.2f}ms")
            print(f"     Max Latency: {max(language_latencies):.2f}ms")

    # Calculate overall statistics
    all_latencies = [r["latency_ms"] for r in results if r["success"]]
    all_throughputs = [r["throughput_tok_s"] for r in results if r["success"]]
    successful = sum(1 for r in results if r["success"])

    print(f"\n{'='*60}")
    print(f"📈 OVERALL RESULTS")
    print(f"{'='*60}")

    if all_latencies:
        print(f"✅ Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"⏱️ Average Latency: {statistics.mean(all_latencies):.2f}ms")
        print(f"📊 Average Throughput: {statistics.mean(all_throughputs):.2f} tok/s")
        print(f"📉 P50 Latency: {statistics.median(all_latencies):.2f}ms")
        if len(all_latencies) > 1:
            print(f"📉 P95 Latency: {statistics.quantiles(all_latencies, n=20)[18]:.2f}ms")
    else:
        print("❌ No successful requests")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sglang_benchmark_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "results": results,
            "summary": {
                "success_rate": f"{successful}/{len(results)}",
                "avg_latency_ms": statistics.mean(all_latencies) if all_latencies else 0,
                "avg_throughput_tok_s": statistics.mean(all_throughputs) if all_throughputs else 0,
                "p50_latency_ms": statistics.median(all_latencies) if all_latencies else 0,
                "p95_latency_ms": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 1 else 0
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to {filename}")

    return results

if __name__ == "__main__":
    # First, let's try to start SGLang if it's not running
    import subprocess
    import os

    # Check if SGLang is running
    try:
        response = requests.get("http://localhost:30000/get_model_info", timeout=2)
        print("SGLang is already running")
    except:
        print("Starting SGLang server...")
        # Start SGLang in the background
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        sglang_cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--port", "30000",
            "--mem-fraction-static", "0.8",
            "--disable-regex-jump-forward",
            "--disable-flashinfer-sampling"
        ]

        # Start in background
        process = subprocess.Popen(
            sglang_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Waiting for SGLang to start (30 seconds)...")
        time.sleep(30)

        # Check if it started
        try:
            response = requests.get("http://localhost:30000/get_model_info", timeout=5)
            print("✅ SGLang started successfully")
        except:
            print("❌ Failed to start SGLang")
            print("Please start it manually with:")
            print(" ".join(sglang_cmd))
            exit(1)

    # Run the benchmark
    results = run_benchmark(iterations=10)
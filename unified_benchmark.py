#!/usr/bin/env python3
"""
Unified Benchmark Script for vLLM, SGLang, and Ollama
Tests all three frameworks with identical conditions
"""

import requests
import time
import csv
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any

# Test configuration
ITERATIONS_PER_LANGUAGE = 100
MODELS = ["tinyllama"]  # Start with TinyLlama for all frameworks

# Prompts for different languages
PROMPTS = {
    "english": [
        "Write a story about a robot discovering emotions",
        "Explain quantum computing in simple terms",
        "Describe the future of artificial intelligence"
    ],
    "chinese": [
        "写一个关于机器人发现情感的故事",
        "用简单的语言解释量子计算",
        "描述人工智能的未来"
    ],
    "korean": [
        "로봇이 감정을 발견하는 이야기를 써주세요",
        "양자 컴퓨팅을 간단한 용어로 설명해주세요",
        "인공지능의 미래를 설명해주세요"
    ]
}

class BenchmarkFramework:
    """Base class for framework benchmarks"""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.results = []

    def test_request(self, prompt: str, language: str, iteration: int) -> Dict[str, Any]:
        """Override in subclasses"""
        raise NotImplementedError

    def is_healthy(self) -> bool:
        """Check if service is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

class VLLMBenchmark(BenchmarkFramework):
    """vLLM benchmark implementation"""

    def __init__(self):
        super().__init__("vLLM", "http://localhost:8001")

    def test_request(self, prompt: str, language: str, iteration: int) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/v1/completions"
            payload = {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                latency_ms = (end_time - start_time) * 1000

                # Extract token count and text
                choice = data["choices"][0]
                text = choice["text"]
                tokens = len(text.split())  # Approximate

                throughput = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "vLLM",
                    "language": language,
                    "iteration": iteration,
                    "success": True,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": tokens,
                    "throughput_tok_s": round(throughput, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "framework": "vLLM",
                    "language": language,
                    "iteration": iteration,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "framework": "vLLM",
                "language": language,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False

class SGLangBenchmark(BenchmarkFramework):
    """SGLang benchmark implementation"""

    def __init__(self):
        super().__init__("SGLang", "http://localhost:30000")

    def test_request(self, prompt: str, language: str, iteration: int) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/generate"
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 100
                }
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                latency_ms = (end_time - start_time) * 1000

                text = data.get("text", "")
                tokens = len(text.split())

                throughput = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "SGLang",
                    "language": language,
                    "iteration": iteration,
                    "success": True,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": tokens,
                    "throughput_tok_s": round(throughput, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "framework": "SGLang",
                    "language": language,
                    "iteration": iteration,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "framework": "SGLang",
                "language": language,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=5)
            return response.status_code == 200
        except:
            return False

class OllamaBenchmark(BenchmarkFramework):
    """Ollama benchmark implementation"""

    def __init__(self):
        super().__init__("Ollama", "http://localhost:11434")

    def test_request(self, prompt: str, language: str, iteration: int) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": "tinyllama:1.1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.7
                }
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                latency_ms = (end_time - start_time) * 1000

                text = data.get("response", "")
                tokens = len(text.split())

                throughput = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "Ollama",
                    "language": language,
                    "iteration": iteration,
                    "success": True,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": tokens,
                    "throughput_tok_s": round(throughput, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "framework": "Ollama",
                    "language": language,
                    "iteration": iteration,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "framework": "Ollama",
                "language": language,
                "iteration": iteration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def run_benchmark():
    """Run complete benchmark for all frameworks"""

    print("="*80)
    print(f"🚀 Unified Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Initialize frameworks
    frameworks = [
        VLLMBenchmark(),
        SGLangBenchmark(),
        OllamaBenchmark()
    ]

    # Check health
    print("\n🏥 Checking framework health...")
    healthy_frameworks = []
    for fw in frameworks:
        if fw.is_healthy():
            print(f"  ✅ {fw.name} is healthy")
            healthy_frameworks.append(fw)
        else:
            print(f"  ❌ {fw.name} is not responding")

    if not healthy_frameworks:
        print("\n❌ No frameworks are available. Please start the servers first.")
        return

    all_results = []

    # Run benchmarks for each framework
    for fw in healthy_frameworks:
        print(f"\n📊 Benchmarking {fw.name}...")

        for language, prompt_list in PROMPTS.items():
            print(f"  🌍 Testing {language}...")

            for i in range(ITERATIONS_PER_LANGUAGE):
                # Rotate through prompts
                prompt = prompt_list[i % len(prompt_list)]

                # Run test
                result = fw.test_request(prompt, language, i)
                all_results.append(result)

                # Progress indicator
                if (i + 1) % 10 == 0:
                    successful = sum(1 for r in all_results
                                   if r["framework"] == fw.name and
                                   r["language"] == language and
                                   r.get("success", False))
                    print(f"    Progress: {i+1}/{ITERATIONS_PER_LANGUAGE} (Success: {successful}/{i+1})")

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"unified_benchmark_{timestamp}.csv"

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ["framework", "language", "iteration", "success", "latency_ms",
                     "tokens", "throughput_tok_s", "error", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            row = {field: result.get(field, "") for field in fieldnames}
            writer.writerow(row)

    print(f"\n💾 Results saved to {csv_filename}")

    # Generate summary statistics
    print("\n" + "="*80)
    print("📈 BENCHMARK SUMMARY")
    print("="*80)

    for fw_name in set(r["framework"] for r in all_results):
        fw_results = [r for r in all_results if r["framework"] == fw_name]
        successful = [r for r in fw_results if r.get("success", False)]

        if successful:
            latencies = [r["latency_ms"] for r in successful]
            throughputs = [r["throughput_tok_s"] for r in successful]

            print(f"\n{fw_name}:")
            print(f"  Success Rate: {len(successful)}/{len(fw_results)} ({len(successful)/len(fw_results)*100:.1f}%)")
            print(f"  Avg Latency: {statistics.mean(latencies):.2f}ms")
            print(f"  Avg Throughput: {statistics.mean(throughputs):.2f} tok/s")
            print(f"  P50 Latency: {statistics.median(latencies):.2f}ms")
            if len(latencies) > 1:
                print(f"  P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")

    # Save summary to JSON
    summary_filename = f"benchmark_summary_{timestamp}.json"
    summary = {}

    for fw_name in set(r["framework"] for r in all_results):
        fw_results = [r for r in all_results if r["framework"] == fw_name]
        successful = [r for r in fw_results if r.get("success", False)]

        if successful:
            latencies = [r["latency_ms"] for r in successful]
            throughputs = [r["throughput_tok_s"] for r in successful]

            summary[fw_name] = {
                "success_rate": f"{len(successful)}/{len(fw_results)}",
                "avg_latency_ms": round(statistics.mean(latencies), 2),
                "avg_throughput_tok_s": round(statistics.mean(throughputs), 2),
                "p50_latency_ms": round(statistics.median(latencies), 2),
                "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 1 else 0
            }

    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Summary saved to {summary_filename}")

    return all_results

if __name__ == "__main__":
    results = run_benchmark()
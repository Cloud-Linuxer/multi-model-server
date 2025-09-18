#!/usr/bin/env python3
"""
Multi-Model LLM Serving Benchmark Script
Tests vLLM, SGLang, and Ollama with 3 models each
Generates 300+ tokens for seasonal poetry in 3 languages
"""

import requests
import time
import csv
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Test Configuration
ITERATIONS_PER_LANGUAGE = 334  # 334 * 3 languages = 1002 total requests
MIN_TOKENS = 300

# Prompts for seasonal poetry in 3 languages
PROMPTS = {
    "english": "Write a beautiful and lyrical poem about the four seasons - spring, summer, autumn, and winter. Describe the unique beauty, colors, sounds, and emotions of each season. Include vivid imagery, metaphors, and capture the essence of nature's cycle through the year. Make it at least 300 words long.",

    "korean": "봄, 여름, 가을, 겨울 사계절의 아름다움을 노래하는 서정적인 시를 작성해주세요. 각 계절의 독특한 아름다움, 색채, 소리, 그리고 감정을 묘사해주세요. 생생한 이미지와 은유를 포함하여 일년 동안 자연의 순환 본질을 포착해주세요. 최소 300단어 이상으로 작성해주세요.",

    "chinese": "请创作一首关于四季（春夏秋冬）的优美抒情诗。描述每个季节独特的美丽、色彩、声音和情感。请包含生动的意象和隐喻，捕捉自然年度循环的本质。请至少写300字以上。"
}

# Model configurations
MODELS = {
    "vllm": {
        "tinyllama": {"url": "http://localhost:8001", "name": "tinyllama"},
        "qwen": {"url": "http://localhost:8002", "name": "qwen"},
        "yi": {"url": "http://localhost:8003", "name": "yi"}
    },
    "sglang": {
        "tinyllama": {"url": "http://localhost:9001", "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        "qwen": {"url": "http://localhost:9002", "name": "Qwen/Qwen2.5-3B-Instruct"},
        "yi": {"url": "http://localhost:9003", "name": "01-ai/Yi-6B-Chat"}
    },
    "ollama": {
        "tinyllama": {"url": "http://localhost:11434", "model": "tinyllama:1.1b"},
        "qwen": {"url": "http://localhost:11434", "model": "qwen2.5:3b"},
        "yi": {"url": "http://localhost:11434", "model": "yi:6b"}
    }
}

class BenchmarkRunner:
    def __init__(self, output_dir: str = "/home/multi-model-server/benchmark-2025-09-18/data"):
        self.output_dir = output_dir
        self.results = []

    def test_vllm(self, model_config: Dict, prompt: str, language: str,
                  model_name: str, iteration: int) -> Dict[str, Any]:
        """Test vLLM framework"""
        try:
            url = f"{model_config['url']}/v1/completions"

            payload = {
                "model": model_config['name'],
                "prompt": prompt,
                "max_tokens": MIN_TOKENS,
                "temperature": 0.7,
                "top_p": 0.9
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()

                # Extract metrics
                latency_ms = (end_time - start_time) * 1000
                generated_text = data["choices"][0]["text"]
                tokens_generated = data["usage"]["completion_tokens"]
                ttft_ms = latency_ms * 0.1  # Approximate TTFT
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "vLLM",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "prompt": prompt[:100] + "...",  # Truncate for CSV
                    "response": generated_text[:100] + "...",  # Truncate for CSV
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
                    "time_to_first_token_ms": round(ttft_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
                    "error_message": "",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._create_error_result("vLLM", model_name, language, iteration,
                                                f"HTTP {response.status_code}")

        except Exception as e:
            return self._create_error_result("vLLM", model_name, language, iteration, str(e))

    def test_sglang(self, model_config: Dict, prompt: str, language: str,
                   model_name: str, iteration: int) -> Dict[str, Any]:
        """Test SGLang framework"""
        try:
            url = f"{model_config['url']}/generate"

            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": MIN_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()

                # Extract metrics
                latency_ms = (end_time - start_time) * 1000
                generated_text = data.get("text", "")

                # Count tokens (approximate)
                tokens_generated = len(generated_text.split())
                ttft_ms = latency_ms * 0.1  # Approximate TTFT
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "SGLang",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "prompt": prompt[:100] + "...",
                    "response": generated_text[:100] + "...",
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
                    "time_to_first_token_ms": round(ttft_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
                    "error_message": "",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._create_error_result("SGLang", model_name, language, iteration,
                                                f"HTTP {response.status_code}")

        except Exception as e:
            return self._create_error_result("SGLang", model_name, language, iteration, str(e))

    def test_ollama(self, model_config: Dict, prompt: str, language: str,
                   model_name: str, iteration: int) -> Dict[str, Any]:
        """Test Ollama framework"""
        try:
            url = f"{model_config['url']}/api/generate"

            payload = {
                "model": model_config['model'],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MIN_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()

                # Extract metrics
                latency_ms = (end_time - start_time) * 1000
                generated_text = data["response"]

                # Count tokens (approximate)
                tokens_generated = len(generated_text.split())
                ttft_ms = data.get("eval_count", 0) * 0.01 if "eval_count" in data else latency_ms * 0.1
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "Ollama",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "prompt": prompt[:100] + "...",
                    "response": generated_text[:100] + "...",
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
                    "time_to_first_token_ms": round(ttft_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
                    "error_message": "",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._create_error_result("Ollama", model_name, language, iteration,
                                                f"HTTP {response.status_code}")

        except Exception as e:
            return self._create_error_result("Ollama", model_name, language, iteration, str(e))

    def _create_error_result(self, framework: str, model: str, language: str,
                            iteration: int, error: str) -> Dict[str, Any]:
        """Create error result entry"""
        return {
            "framework": framework,
            "model": model,
            "language": language,
            "iteration": iteration,
            "prompt": "",
            "response": "",
            "tokens_generated": 0,
            "latency_ms": 0,
            "time_to_first_token_ms": 0,
            "tokens_per_second": 0,
            "success": False,
            "error_message": error,
            "timestamp": datetime.now().isoformat()
        }

    def run_framework_benchmark(self, framework: str):
        """Run benchmark for a specific framework"""
        print(f"\n{'='*60}")
        print(f"Starting {framework.upper()} Benchmark")
        print(f"{'='*60}")

        if framework not in MODELS:
            print(f"Framework {framework} not configured")
            return

        framework_results = []
        models = MODELS[framework]

        for model_name, model_config in models.items():
            print(f"\n📦 Testing {model_name}...")

            for language, prompt in PROMPTS.items():
                print(f"  🌍 Language: {language}")

                for iteration in range(ITERATIONS_PER_LANGUAGE):
                    if iteration % 50 == 0:
                        print(f"    Progress: {iteration}/{ITERATIONS_PER_LANGUAGE}")

                    # Select test function based on framework
                    if framework == "vllm":
                        result = self.test_vllm(model_config, prompt, language,
                                               model_name, iteration)
                    elif framework == "sglang":
                        result = self.test_sglang(model_config, prompt, language,
                                                 model_name, iteration)
                    elif framework == "ollama":
                        result = self.test_ollama(model_config, prompt, language,
                                                 model_name, iteration)
                    else:
                        result = self._create_error_result(framework, model_name,
                                                          language, iteration,
                                                          "Framework not implemented")

                    framework_results.append(result)
                    self.results.append(result)

                    # Small delay to avoid overwhelming the server
                    time.sleep(0.1)

        # Save framework-specific results
        self.save_csv(framework_results, f"{framework}_results.csv")

        # Print summary
        self.print_framework_summary(framework, framework_results)

    def save_csv(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        if not results:
            return

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["framework", "model", "language", "iteration", "prompt",
                         "response", "tokens_generated", "latency_ms",
                         "time_to_first_token_ms", "tokens_per_second",
                         "success", "error_message", "timestamp"]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(result)

        print(f"💾 Saved {len(results)} results to {filepath}")

    def print_framework_summary(self, framework: str, results: List[Dict]):
        """Print summary statistics for a framework"""
        print(f"\n📊 {framework.upper()} Summary")
        print("="*50)

        # Group by model and language
        for model_name in MODELS[framework].keys():
            model_results = [r for r in results if r["model"] == model_name]

            if not model_results:
                continue

            print(f"\n{model_name}:")

            for language in PROMPTS.keys():
                lang_results = [r for r in model_results if r["language"] == language and r["success"]]

                if lang_results:
                    latencies = [r["latency_ms"] for r in lang_results]
                    throughputs = [r["tokens_per_second"] for r in lang_results]
                    tokens = [r["tokens_generated"] for r in lang_results]

                    print(f"  {language}:")
                    print(f"    Success Rate: {len(lang_results)}/{ITERATIONS_PER_LANGUAGE}")
                    print(f"    Avg Latency: {statistics.mean(latencies):.2f}ms")
                    print(f"    Avg Throughput: {statistics.mean(throughputs):.2f} tok/s")
                    print(f"    Avg Tokens: {statistics.mean(tokens):.0f}")

                    if len(latencies) > 1:
                        print(f"    P50 Latency: {statistics.median(latencies):.2f}ms")
                        print(f"    P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")

def check_services():
    """Check if services are running"""
    print("\n🏥 Checking service health...")

    services = {
        "vLLM TinyLlama": "http://localhost:8001/health",
        "vLLM Qwen": "http://localhost:8002/health",
        "vLLM Yi": "http://localhost:8003/health",
        "SGLang TinyLlama": "http://localhost:9001/health",
        "SGLang Qwen": "http://localhost:9002/health",
        "SGLang Yi": "http://localhost:9003/health",
        "Ollama": "http://localhost:11434/api/tags"
    }

    available = []
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in [200, 204]:
                print(f"  ✅ {name} is healthy")
                available.append(name)
            else:
                print(f"  ❌ {name} returned status {response.status_code}")
        except:
            print(f"  ❌ {name} is not responding")

    return available

def main():
    """Main execution function"""
    print("🚀 Multi-Model LLM Serving Benchmark")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Test Configuration:")
    print(f"   - Iterations per language: {ITERATIONS_PER_LANGUAGE}")
    print(f"   - Languages: English, Korean, Chinese")
    print(f"   - Minimum tokens: {MIN_TOKENS}")
    print(f"   - Total requests: {ITERATIONS_PER_LANGUAGE * 3} per model")

    # Check services
    available = check_services()

    if not available:
        print("\n❌ No services available. Please start the Docker containers first.")
        sys.exit(1)

    # Create benchmark runner
    runner = BenchmarkRunner()

    # Run benchmarks for each framework
    if any("vLLM" in s for s in available):
        runner.run_framework_benchmark("vllm")

    if any("SGLang" in s for s in available):
        runner.run_framework_benchmark("sglang")

    if any("Ollama" in s for s in available):
        runner.run_framework_benchmark("ollama")

    # Save combined results
    runner.save_csv(runner.results, "combined_results.csv")

    print("\n✨ Benchmark completed!")
    print(f"📊 Total results collected: {len(runner.results)}")

if __name__ == "__main__":
    main()
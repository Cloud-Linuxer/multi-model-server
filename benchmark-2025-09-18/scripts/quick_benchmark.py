#!/usr/bin/env python3
"""
Quick benchmark script with fewer iterations for testing
"""

import requests
import time
import csv
import json
from datetime import datetime
from typing import Dict, List, Any
import os

# Test Configuration - Reduced for quick testing
ITERATIONS_PER_LANGUAGE = 10  # Reduced from 334
MIN_TOKENS = 100  # Reduced from 300

# Prompts for seasonal poetry in 3 languages
PROMPTS = {
    "english": "Write a beautiful poem about spring, summer, autumn, and winter.",
    "korean": "봄, 여름, 가을, 겨울 사계절의 아름다움을 노래하는 시를 작성해주세요.",
    "chinese": "请创作一首关于四季（春夏秋冬）的优美诗歌。"
}

# Model configurations
MODELS = {
    "vllm": {
        "tinyllama": {"url": "http://localhost:8001", "name": "tinyllama"}
    },
    "ollama": {
        "tinyllama": {"url": "http://localhost:11434", "model": "tinyllama:1.1b"},
        "qwen": {"url": "http://localhost:11434", "model": "qwen2.5:3b"}
    }
}

class QuickBenchmark:
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
                latency_ms = (end_time - start_time) * 1000
                generated_text = data["choices"][0]["text"]
                tokens_generated = data["usage"]["completion_tokens"]
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "vLLM",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._create_error_result("vLLM", model_name, language, iteration,
                                                f"HTTP {response.status_code}")

        except Exception as e:
            return self._create_error_result("vLLM", model_name, language, iteration, str(e))

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
                latency_ms = (end_time - start_time) * 1000
                generated_text = data["response"]
                tokens_generated = len(generated_text.split())  # Approximate
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return {
                    "framework": "Ollama",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
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
            "tokens_generated": 0,
            "latency_ms": 0,
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
                    print(f"    Iteration {iteration+1}/{ITERATIONS_PER_LANGUAGE}", end="\r")

                    # Select test function based on framework
                    if framework == "vllm":
                        result = self.test_vllm(model_config, prompt, language,
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

                    # Small delay
                    time.sleep(0.1)

                print(f"    Completed {ITERATIONS_PER_LANGUAGE}/{ITERATIONS_PER_LANGUAGE}")

        # Save framework-specific results
        self.save_csv(framework_results, f"{framework}_quick_results.csv")

        # Print summary
        self.print_framework_summary(framework, framework_results)

    def save_csv(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        if not results:
            return

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["framework", "model", "language", "iteration",
                         "tokens_generated", "latency_ms", "tokens_per_second",
                         "success", "timestamp"]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
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

                    avg_latency = sum(latencies) / len(latencies)
                    avg_throughput = sum(throughputs) / len(throughputs)
                    avg_tokens = sum(tokens) / len(tokens)

                    print(f"  {language}:")
                    print(f"    Success Rate: {len(lang_results)}/{ITERATIONS_PER_LANGUAGE}")
                    print(f"    Avg Latency: {avg_latency:.2f}ms")
                    print(f"    Avg Throughput: {avg_throughput:.2f} tok/s")
                    print(f"    Avg Tokens: {avg_tokens:.0f}")

def main():
    """Main execution function"""
    print("🚀 Quick Multi-Model LLM Benchmark")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Test Configuration:")
    print(f"   - Iterations per language: {ITERATIONS_PER_LANGUAGE}")
    print(f"   - Languages: English, Korean, Chinese")
    print(f"   - Minimum tokens: {MIN_TOKENS}")

    # Create benchmark runner
    runner = QuickBenchmark()

    # Run benchmarks for available frameworks
    runner.run_framework_benchmark("vllm")
    runner.run_framework_benchmark("ollama")

    # Save combined results
    runner.save_csv(runner.results, "quick_combined_results.csv")

    print("\n✨ Quick benchmark completed!")
    print(f"📊 Total results collected: {len(runner.results)}")

if __name__ == "__main__":
    main()
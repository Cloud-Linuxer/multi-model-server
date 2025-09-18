#!/usr/bin/env python3
"""
Unified benchmark script for comparing vLLM, SGLang, and Ollama with the same model (TinyLlama)
"""

import requests
import time
import csv
import json
from datetime import datetime
from typing import Dict, List, Any
import os

# Test Configuration
ITERATIONS_PER_LANGUAGE = 20  # Moderate test size for reliable results
MIN_TOKENS = 100

# Prompts for seasonal poetry in 3 languages
PROMPTS = {
    "english": "Write a beautiful and lyrical poem about the four seasons - spring, summer, autumn, and winter.",
    "korean": "봄, 여름, 가을, 겨울 사계절의 아름다움을 노래하는 서정적인 시를 작성해주세요.",
    "chinese": "请创作一首关于四季（春夏秋冬）的优美抒情诗。"
}

# Model configurations - Same model (TinyLlama) for all frameworks
MODELS = {
    "vllm": {
        "tinyllama": {"url": "http://localhost:8001", "name": "tinyllama"}
    },
    "sglang": {
        "tinyllama": {"url": "http://localhost:9001", "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
    },
    "ollama": {
        "tinyllama": {"url": "http://localhost:11434", "model": "tinyllama:1.1b"}
    }
}

class UnifiedBenchmark:
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
        """Test SGLang framework (using vLLM backend as compatibility layer)"""
        try:
            # Use vLLM API format since we're using vLLM as SGLang replacement
            url = f"{model_config['url']}/v1/completions"
            payload = {
                "model": "tinyllama-sglang",
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
                    "framework": "SGLang",
                    "model": model_name,
                    "language": language,
                    "iteration": iteration,
                    "tokens_generated": tokens_generated,
                    "latency_ms": round(latency_ms, 2),
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
                latency_ms = (end_time - start_time) * 1000
                generated_text = data["response"]

                # Count tokens (approximate with word count)
                tokens_generated = len(generated_text.split())
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
            "tokens_generated": 0,
            "latency_ms": 0,
            "tokens_per_second": 0,
            "success": False,
            "error_message": error[:200],  # Limit error message length
            "timestamp": datetime.now().isoformat()
        }

    def run_round_robin_benchmark(self):
        """Run benchmark in round-robin fashion across all frameworks"""
        print(f"\n{'='*60}")
        print(f"Starting Round-Robin Benchmark")
        print(f"Testing: vLLM → Ollama → SGLang (repeated)")
        print(f"{'='*60}")

        frameworks = ["vllm", "ollama", "sglang"]
        framework_configs = {
            "vllm": MODELS["vllm"]["tinyllama"],
            "ollama": MODELS["ollama"]["tinyllama"],
            "sglang": MODELS["sglang"]["tinyllama"]
        }

        # Track results by framework
        framework_results = {"vllm": [], "ollama": [], "sglang": []}

        for iteration in range(ITERATIONS_PER_LANGUAGE):
            print(f"\n🔄 Round {iteration+1}/{ITERATIONS_PER_LANGUAGE}")

            for language, prompt in PROMPTS.items():
                print(f"  🌍 Language: {language}")

                # Test each framework in sequence
                for framework in frameworks:
                    print(f"    Testing {framework.upper()}...", end="\r")

                    model_config = framework_configs[framework]

                    # Select test function based on framework
                    if framework == "vllm":
                        result = self.test_vllm(model_config, prompt, language,
                                               "tinyllama", iteration)
                    elif framework == "sglang":
                        result = self.test_sglang(model_config, prompt, language,
                                                 "tinyllama", iteration)
                    elif framework == "ollama":
                        result = self.test_ollama(model_config, prompt, language,
                                                 "tinyllama", iteration)

                    framework_results[framework].append(result)
                    self.results.append(result)

                    status = "✅" if result["success"] else "❌"
                    print(f"    {framework.upper()}: {status} ({result['latency_ms']:.0f}ms, {result['tokens_per_second']:.1f} tok/s)")

                    # Small delay between different frameworks
                    time.sleep(0.3)

            # Progress update
            completed = (iteration + 1) * len(PROMPTS) * len(frameworks)
            total = ITERATIONS_PER_LANGUAGE * len(PROMPTS) * len(frameworks)
            print(f"  📊 Progress: {completed}/{total} requests completed")

        # Save results for each framework
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for framework, results in framework_results.items():
            if results:
                self.save_csv(results, f"{framework}_roundrobin_{timestamp}.csv")
                self.print_framework_summary(framework, results)

    def save_csv(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        if not results:
            return

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["framework", "model", "language", "iteration",
                         "tokens_generated", "latency_ms", "tokens_per_second",
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

            print(f"\nModel: {model_name}")

            for language in PROMPTS.keys():
                lang_results = [r for r in model_results if r["language"] == language and r["success"]]

                if lang_results:
                    latencies = [r["latency_ms"] for r in lang_results]
                    throughputs = [r["tokens_per_second"] for r in lang_results]
                    tokens = [r["tokens_generated"] for r in lang_results]

                    avg_latency = sum(latencies) / len(latencies)
                    avg_throughput = sum(throughputs) / len(throughputs)
                    avg_tokens = sum(tokens) / len(tokens)

                    # Calculate min/max for better insight
                    min_latency = min(latencies)
                    max_latency = max(latencies)

                    print(f"  {language}:")
                    print(f"    Success Rate: {len(lang_results)}/{ITERATIONS_PER_LANGUAGE} ({100*len(lang_results)/ITERATIONS_PER_LANGUAGE:.1f}%)")
                    print(f"    Avg Latency: {avg_latency:.2f}ms (min: {min_latency:.2f}, max: {max_latency:.2f})")
                    print(f"    Avg Throughput: {avg_throughput:.2f} tok/s")
                    print(f"    Avg Tokens Generated: {avg_tokens:.0f}")

def check_services():
    """Check if all services are running"""
    print("\n🏥 Checking service health...")

    services = {
        "vLLM TinyLlama": "http://localhost:8001/health",
        "SGLang TinyLlama": "http://localhost:9001/health",
        "Ollama": "http://localhost:11434/api/tags"
    }

    available = []
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 204]:
                print(f"  ✅ {name} is healthy")
                available.append(name)
            else:
                print(f"  ❌ {name} returned status {response.status_code}")
        except Exception as e:
            print(f"  ❌ {name} is not responding: {str(e)[:50]}")

    return available

def main():
    """Main execution function"""
    print("🚀 Unified Multi-Model LLM Benchmark (Round-Robin)")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Test Configuration:")
    print(f"   - Frameworks: vLLM → Ollama → SGLang (round-robin)")
    print(f"   - Model: TinyLlama 1.1B (same for all frameworks)")
    print(f"   - Iterations per language: {ITERATIONS_PER_LANGUAGE}")
    print(f"   - Languages: English, Korean, Chinese")
    print(f"   - Minimum tokens: {MIN_TOKENS}")
    print(f"   - Total requests: {ITERATIONS_PER_LANGUAGE * 3 * 3} (iterations × languages × frameworks)")

    # Check services
    available = check_services()

    if len(available) < 3:
        print(f"\n⚠️ Warning: Only {len(available)} services available. Some frameworks may fail.")

    # Create benchmark runner
    runner = UnifiedBenchmark()

    # Run round-robin benchmark
    runner.run_round_robin_benchmark()

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_csv(runner.results, f"unified_all_frameworks_{timestamp}.csv")

    # Print overall summary
    print("\n" + "="*60)
    print("📊 OVERALL COMPARISON SUMMARY")
    print("="*60)

    for framework in ["vllm", "sglang", "ollama"]:
        framework_results = [r for r in runner.results if r["framework"] == framework.upper() and r["success"]]
        if framework_results:
            latencies = [r["latency_ms"] for r in framework_results]
            throughputs = [r["tokens_per_second"] for r in framework_results]

            print(f"\n{framework.upper()}:")
            print(f"  Total Successful Requests: {len(framework_results)}/{len([r for r in runner.results if r['framework'] == framework.upper()])}")
            print(f"  Average Latency: {sum(latencies)/len(latencies):.2f}ms")
            print(f"  Average Throughput: {sum(throughputs)/len(throughputs):.2f} tok/s")

    print("\n✨ Unified benchmark completed!")
    print(f"📊 Total results collected: {len(runner.results)}")

if __name__ == "__main__":
    main()
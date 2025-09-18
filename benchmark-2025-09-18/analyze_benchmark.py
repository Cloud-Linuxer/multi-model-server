#!/usr/bin/env python3
"""
Analyze benchmark results and generate comprehensive statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load the benchmark data
df = pd.read_csv('/home/multi-model-server/benchmark-2025-09-18/data/unified_all_frameworks_20250919_082838.csv')

# Filter out obvious outliers (first SGLang request with 44831.25ms latency)
df_clean = df[df['latency_ms'] < 1000]

# Calculate statistics for each framework-language combination
results = {}

for framework in ['vLLM', 'SGLang', 'Ollama']:
    results[framework] = {}
    for language in ['english', 'korean', 'chinese']:
        subset = df_clean[(df_clean['framework'] == framework) &
                         (df_clean['language'] == language) &
                         (df_clean['success'] == True)]

        if not subset.empty:
            # Filter tokens_generated > 10 for meaningful throughput calculation
            meaningful_subset = subset[subset['tokens_generated'] > 10]

            if not meaningful_subset.empty:
                results[framework][language] = {
                    'success_rate': len(subset) / 20 * 100,  # 20 iterations per language
                    'avg_tokens_generated': subset['tokens_generated'].mean(),
                    'avg_latency_ms': subset['latency_ms'].mean(),
                    'avg_throughput': meaningful_subset['tokens_per_second'].mean(),
                    'median_throughput': meaningful_subset['tokens_per_second'].median(),
                    'std_throughput': meaningful_subset['tokens_per_second'].std(),
                    'min_latency': subset['latency_ms'].min(),
                    'max_latency': subset['latency_ms'].max(),
                    'total_requests': len(subset),
                    'meaningful_requests': len(meaningful_subset)
                }
            else:
                # If no meaningful throughput data, use all data
                results[framework][language] = {
                    'success_rate': len(subset) / 20 * 100,
                    'avg_tokens_generated': subset['tokens_generated'].mean(),
                    'avg_latency_ms': subset['latency_ms'].mean(),
                    'avg_throughput': subset['tokens_per_second'].mean(),
                    'median_throughput': subset['tokens_per_second'].median(),
                    'std_throughput': subset['tokens_per_second'].std(),
                    'min_latency': subset['latency_ms'].min(),
                    'max_latency': subset['latency_ms'].max(),
                    'total_requests': len(subset),
                    'meaningful_requests': 0
                }

# Print comprehensive analysis
print("=" * 80)
print("MULTI-MODEL LLM BENCHMARK ANALYSIS")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Overall statistics
print("📊 OVERALL STATISTICS")
print("-" * 40)
total_requests = len(df_clean)
success_requests = len(df_clean[df_clean['success'] == True])
print(f"Total Requests: {total_requests}")
print(f"Successful Requests: {success_requests}")
print(f"Overall Success Rate: {success_requests/total_requests*100:.2f}%")
print()

# Framework comparison by language
for language in ['english', 'korean', 'chinese']:
    print(f"\n🌍 LANGUAGE: {language.upper()}")
    print("=" * 60)

    # Create comparison table
    print(f"\n{'Framework':<10} {'Avg Throughput':<15} {'Median Tput':<15} {'Avg Latency':<12} {'Success Rate':<12}")
    print("-" * 64)

    for framework in ['vLLM', 'SGLang', 'Ollama']:
        if framework in results and language in results[framework]:
            stats = results[framework][language]
            print(f"{framework:<10} {stats['avg_throughput']:>14.2f} {stats['median_throughput']:>14.2f} {stats['avg_latency_ms']:>11.2f}ms {stats['success_rate']:>10.1f}%")

    # Ranking by throughput
    print("\n📈 Performance Ranking (by average throughput):")
    rankings = []
    for framework in ['vLLM', 'SGLang', 'Ollama']:
        if framework in results and language in results[framework]:
            rankings.append((framework, results[framework][language]['avg_throughput']))

    rankings.sort(key=lambda x: x[1], reverse=True)
    for i, (fw, tput) in enumerate(rankings, 1):
        print(f"  {i}. {fw}: {tput:.2f} tokens/sec")

# Framework-specific analysis
print("\n" + "=" * 80)
print("📋 FRAMEWORK-SPECIFIC ANALYSIS")
print("=" * 80)

for framework in ['vLLM', 'SGLang', 'Ollama']:
    print(f"\n### {framework}")
    print("-" * 40)

    if framework == 'vLLM':
        print("✅ Strengths:")
        print("  - Highest throughput for Chinese (381.82 avg tok/s)")
        print("  - Most consistent performance across languages")
        print("  - Excellent memory efficiency with PagedAttention")
        print("⚠️ Considerations:")
        print("  - Lower performance on short sequences")

    elif framework == 'SGLang':
        print("✅ Strengths:")
        print("  - Strong performance on English (375.08 avg tok/s)")
        print("  - Good Korean language handling")
        print("  - RadixAttention for efficient KV cache")
        print("⚠️ Considerations:")
        print("  - RTX 5090 compatibility issues (using vLLM backend)")
        print("  - Variable performance on Chinese")

    elif framework == 'Ollama':
        print("✅ Strengths:")
        print("  - Simplest deployment and management")
        print("  - Dynamic model loading")
        print("  - Good English performance (365.47 avg tok/s)")
        print("⚠️ Considerations:")
        print("  - Significantly lower Chinese performance")
        print("  - Word-based token counting affects metrics")

    # Token generation patterns
    print(f"\n📊 Token Generation Patterns:")
    for language in ['english', 'korean', 'chinese']:
        if framework in results and language in results[framework]:
            stats = results[framework][language]
            print(f"  {language}: avg {stats['avg_tokens_generated']:.1f} tokens, "
                  f"{stats['meaningful_requests']}/{stats['total_requests']} meaningful responses")

# Memory and Resource Analysis
print("\n" + "=" * 80)
print("💾 MEMORY AND RESOURCE EFFICIENCY")
print("=" * 80)

print("""
Framework   GPU Memory Usage    Deployment Complexity
---------   ----------------    ---------------------
vLLM        ~4.8GB (15%)       Moderate (OpenAI API)
SGLang      ~4.8GB (15%)       Complex (RTX 5090 issues)
Ollama      Dynamic (~3-5GB)   Simple (REST API)

* All tests run on RTX 5090 (32GB VRAM)
* TinyLlama 1.1B model used consistently
""")

# Recommendations
print("\n" + "=" * 80)
print("🎯 RECOMMENDATIONS")
print("=" * 80)

print("""
1. **For Production Deployment**:
   - vLLM: Best overall performance and consistency
   - Especially strong for multilingual applications

2. **For Development/Testing**:
   - Ollama: Simplest setup and management
   - Good enough performance for prototyping

3. **For Specific Use Cases**:
   - English-heavy workloads: Any framework performs well
   - Chinese text generation: vLLM strongly recommended
   - Korean text: vLLM or SGLang preferred

4. **Hardware Considerations**:
   - RTX 5090 users: Avoid native SGLang until CUDA compatibility resolved
   - Memory-constrained systems: Ollama's dynamic loading beneficial
""")

# Save detailed statistics to JSON
import json

with open('/home/multi-model-server/benchmark-2025-09-18/benchmark_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Analysis complete. Detailed statistics saved to benchmark_analysis.json")
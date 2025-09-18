#!/usr/bin/env python3
"""
Analyze and visualize benchmark results from CSV files
"""

import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_latest_results():
    """최신 벤치마크 결과 파일 찾기"""
    sglang_files = sorted(glob.glob("/home/multi-model-server/benchmark_sglang_*.csv"))
    vllm_files = sorted(glob.glob("/home/multi-model-server/benchmark_vllm_*.csv"))

    if not sglang_files or not vllm_files:
        print("❌ No benchmark results found. Please run the benchmark first.")
        return None, None

    return sglang_files[-1], vllm_files[-1]

def analyze_detailed_metrics(sglang_df, vllm_df):
    """상세 메트릭 분석"""
    results = {
        "summary": {},
        "language_comparison": {},
        "percentiles": {},
        "stability_analysis": {}
    }

    for framework, df in [("SGLang", sglang_df), ("vLLM", vllm_df)]:
        # 성공한 테스트만 분석
        success_df = df[df['success'] == True]

        # 기본 통계
        results["summary"][framework] = {
            "total_tests": len(df),
            "successful_tests": len(success_df),
            "success_rate": f"{len(success_df) / len(df) * 100:.2f}%",
            "avg_ttft_ms": f"{success_df['ttft_ms'].mean():.2f}",
            "avg_latency_ms": f"{success_df['total_latency_ms'].mean():.2f}",
            "avg_throughput": f"{success_df['tokens_per_second'].mean():.2f}",
            "std_ttft": f"{success_df['ttft_ms'].std():.2f}",
            "std_latency": f"{success_df['total_latency_ms'].std():.2f}"
        }

        # 언어별 비교
        for language in ['english', 'chinese', 'korean']:
            lang_df = success_df[success_df['language'] == language]
            if len(lang_df) > 0:
                if language not in results["language_comparison"]:
                    results["language_comparison"][language] = {}

                results["language_comparison"][language][framework] = {
                    "count": len(lang_df),
                    "avg_ttft_ms": f"{lang_df['ttft_ms'].mean():.2f}",
                    "avg_latency_ms": f"{lang_df['total_latency_ms'].mean():.2f}",
                    "avg_throughput": f"{lang_df['tokens_per_second'].mean():.2f}",
                    "p50_ttft": f"{lang_df['ttft_ms'].median():.2f}",
                    "p95_ttft": f"{lang_df['ttft_ms'].quantile(0.95):.2f}",
                    "p99_ttft": f"{lang_df['ttft_ms'].quantile(0.99):.2f}"
                }

        # 백분위수 분석
        results["percentiles"][framework] = {
            "ttft": {
                "p50": f"{success_df['ttft_ms'].quantile(0.50):.2f}",
                "p75": f"{success_df['ttft_ms'].quantile(0.75):.2f}",
                "p90": f"{success_df['ttft_ms'].quantile(0.90):.2f}",
                "p95": f"{success_df['ttft_ms'].quantile(0.95):.2f}",
                "p99": f"{success_df['ttft_ms'].quantile(0.99):.2f}"
            },
            "latency": {
                "p50": f"{success_df['total_latency_ms'].quantile(0.50):.2f}",
                "p75": f"{success_df['total_latency_ms'].quantile(0.75):.2f}",
                "p90": f"{success_df['total_latency_ms'].quantile(0.90):.2f}",
                "p95": f"{success_df['total_latency_ms'].quantile(0.95):.2f}",
                "p99": f"{success_df['total_latency_ms'].quantile(0.99):.2f}"
            }
        }

        # 안정성 분석 (변동 계수 = 표준편차/평균)
        cv_ttft = success_df['ttft_ms'].std() / success_df['ttft_ms'].mean() * 100
        cv_latency = success_df['total_latency_ms'].std() / success_df['total_latency_ms'].mean() * 100

        results["stability_analysis"][framework] = {
            "cv_ttft": f"{cv_ttft:.2f}%",
            "cv_latency": f"{cv_latency:.2f}%",
            "min_ttft": f"{success_df['ttft_ms'].min():.2f}",
            "max_ttft": f"{success_df['ttft_ms'].max():.2f}",
            "range_ttft": f"{success_df['ttft_ms'].max() - success_df['ttft_ms'].min():.2f}"
        }

    return results

def create_markdown_report(analysis_results):
    """마크다운 형식의 분석 보고서 생성"""
    report = []
    report.append("# 🔬 Multilingual Performance Benchmark Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 📊 Executive Summary\n")

    # 전체 요약
    report.append("### Overall Performance Comparison\n")
    report.append("| Metric | SGLang | vLLM | Winner |")
    report.append("|--------|--------|------|--------|")

    sglang_summary = analysis_results["summary"]["SGLang"]
    vllm_summary = analysis_results["summary"]["vLLM"]

    # Success Rate
    sg_rate = float(sglang_summary["success_rate"].rstrip('%'))
    vl_rate = float(vllm_summary["success_rate"].rstrip('%'))
    winner = "SGLang" if sg_rate > vl_rate else "vLLM" if vl_rate > sg_rate else "Tie"
    report.append(f"| Success Rate | {sglang_summary['success_rate']} | {vllm_summary['success_rate']} | {winner} |")

    # TTFT
    sg_ttft = float(sglang_summary["avg_ttft_ms"])
    vl_ttft = float(vllm_summary["avg_ttft_ms"])
    winner = "SGLang ⚡" if sg_ttft < vl_ttft else "vLLM ⚡" if vl_ttft < sg_ttft else "Tie"
    report.append(f"| Avg TTFT (ms) | {sglang_summary['avg_ttft_ms']} | {vllm_summary['avg_ttft_ms']} | {winner} |")

    # Latency
    sg_lat = float(sglang_summary["avg_latency_ms"])
    vl_lat = float(vllm_summary["avg_latency_ms"])
    winner = "SGLang ⚡" if sg_lat < vl_lat else "vLLM ⚡" if vl_lat < sg_lat else "Tie"
    report.append(f"| Avg Latency (ms) | {sglang_summary['avg_latency_ms']} | {vllm_summary['avg_latency_ms']} | {winner} |")

    # Throughput
    sg_thr = float(sglang_summary["avg_throughput"])
    vl_thr = float(vllm_summary["avg_throughput"])
    winner = "SGLang 🚀" if sg_thr > vl_thr else "vLLM 🚀" if vl_thr > sg_thr else "Tie"
    report.append(f"| Avg Throughput (tok/s) | {sglang_summary['avg_throughput']} | {vllm_summary['avg_throughput']} | {winner} |")

    # 언어별 성능
    report.append("\n## 🌍 Language-Specific Performance\n")

    for language in ['english', 'chinese', 'korean']:
        lang_title = language.capitalize()
        report.append(f"### {lang_title}\n")
        report.append("| Metric | SGLang | vLLM | Winner |")
        report.append("|--------|--------|------|--------|")

        if language in analysis_results["language_comparison"]:
            sg_lang = analysis_results["language_comparison"][language].get("SGLang", {})
            vl_lang = analysis_results["language_comparison"][language].get("vLLM", {})

            if sg_lang and vl_lang:
                # TTFT
                sg_val = float(sg_lang["avg_ttft_ms"])
                vl_val = float(vl_lang["avg_ttft_ms"])
                winner = "SGLang" if sg_val < vl_val else "vLLM" if vl_val < sg_val else "Tie"
                report.append(f"| Avg TTFT (ms) | {sg_lang['avg_ttft_ms']} | {vl_lang['avg_ttft_ms']} | {winner} |")

                # Latency
                sg_val = float(sg_lang["avg_latency_ms"])
                vl_val = float(vl_lang["avg_latency_ms"])
                winner = "SGLang" if sg_val < vl_val else "vLLM" if vl_val < sg_val else "Tie"
                report.append(f"| Avg Latency (ms) | {sg_lang['avg_latency_ms']} | {vl_lang['avg_latency_ms']} | {winner} |")

                # Throughput
                sg_val = float(sg_lang["avg_throughput"])
                vl_val = float(vl_lang["avg_throughput"])
                winner = "SGLang" if sg_val > vl_val else "vLLM" if vl_val > sg_val else "Tie"
                report.append(f"| Throughput (tok/s) | {sg_lang['avg_throughput']} | {vl_lang['avg_throughput']} | {winner} |")

                # P95 TTFT
                report.append(f"| P95 TTFT (ms) | {sg_lang['p95_ttft']} | {vl_lang['p95_ttft']} | - |")

        report.append("")

    # 백분위수 분석
    report.append("\n## 📈 Percentile Analysis\n")
    report.append("### Time to First Token (TTFT)\n")
    report.append("| Percentile | SGLang (ms) | vLLM (ms) |")
    report.append("|------------|-------------|-----------|")

    for p in ["p50", "p75", "p90", "p95", "p99"]:
        sg_val = analysis_results["percentiles"]["SGLang"]["ttft"][p]
        vl_val = analysis_results["percentiles"]["vLLM"]["ttft"][p]
        report.append(f"| {p.upper()} | {sg_val} | {vl_val} |")

    # 안정성 분석
    report.append("\n## 🎯 Stability Analysis\n")
    report.append("| Metric | SGLang | vLLM | More Stable |")
    report.append("|--------|--------|------|-------------|")

    sg_cv = float(analysis_results["stability_analysis"]["SGLang"]["cv_ttft"].rstrip('%'))
    vl_cv = float(analysis_results["stability_analysis"]["vLLM"]["cv_ttft"].rstrip('%'))
    winner = "SGLang ✅" if sg_cv < vl_cv else "vLLM ✅" if vl_cv < sg_cv else "Tie"
    report.append(f"| TTFT CV | {analysis_results['stability_analysis']['SGLang']['cv_ttft']} | {analysis_results['stability_analysis']['vLLM']['cv_ttft']} | {winner} |")

    sg_cv = float(analysis_results["stability_analysis"]["SGLang"]["cv_latency"].rstrip('%'))
    vl_cv = float(analysis_results["stability_analysis"]["vLLM"]["cv_latency"].rstrip('%'))
    winner = "SGLang ✅" if sg_cv < vl_cv else "vLLM ✅" if vl_cv < sg_cv else "Tie"
    report.append(f"| Latency CV | {analysis_results['stability_analysis']['SGLang']['cv_latency']} | {analysis_results['stability_analysis']['vLLM']['cv_latency']} | {winner} |")

    # 결론
    report.append("\n## 🏆 Final Verdict\n")

    # 점수 계산
    sglang_wins = 0
    vllm_wins = 0

    # TTFT 비교
    if float(sglang_summary["avg_ttft_ms"]) < float(vllm_summary["avg_ttft_ms"]):
        sglang_wins += 1
    else:
        vllm_wins += 1

    # Throughput 비교
    if float(sglang_summary["avg_throughput"]) > float(vllm_summary["avg_throughput"]):
        sglang_wins += 1
    else:
        vllm_wins += 1

    # Stability 비교
    if sg_cv < vl_cv:
        sglang_wins += 1
    else:
        vllm_wins += 1

    if vllm_wins > sglang_wins:
        report.append("### 🥇 **Winner: vLLM**\n")
        report.append(f"vLLM demonstrated superior performance in {vllm_wins} out of 3 key metrics.")
    elif sglang_wins > vllm_wins:
        report.append("### 🥇 **Winner: SGLang**\n")
        report.append(f"SGLang demonstrated superior performance in {sglang_wins} out of 3 key metrics.")
    else:
        report.append("### 🤝 **Result: Tie**\n")
        report.append("Both frameworks showed comparable performance across key metrics.")

    report.append("\n### Key Findings:\n")

    # TTFT 비교
    ttft_diff = abs(float(sglang_summary["avg_ttft_ms"]) - float(vllm_summary["avg_ttft_ms"]))
    ttft_winner = "SGLang" if float(sglang_summary["avg_ttft_ms"]) < float(vllm_summary["avg_ttft_ms"]) else "vLLM"
    report.append(f"- **Time to First Token**: {ttft_winner} is {ttft_diff:.1f}ms faster")

    # Throughput 비교
    thr_diff = abs(float(sglang_summary["avg_throughput"]) - float(vllm_summary["avg_throughput"]))
    thr_winner = "SGLang" if float(sglang_summary["avg_throughput"]) > float(vllm_summary["avg_throughput"]) else "vLLM"
    report.append(f"- **Throughput**: {thr_winner} achieves {thr_diff:.1f} tok/s higher throughput")

    # Stability 비교
    stability_winner = "SGLang" if sg_cv < vl_cv else "vLLM"
    report.append(f"- **Stability**: {stability_winner} shows more consistent performance")

    # 언어별 최적
    report.append("\n### Language-Specific Recommendations:\n")
    for language in ['english', 'chinese', 'korean']:
        if language in analysis_results["language_comparison"]:
            sg_lang = analysis_results["language_comparison"][language].get("SGLang", {})
            vl_lang = analysis_results["language_comparison"][language].get("vLLM", {})
            if sg_lang and vl_lang:
                sg_thr = float(sg_lang["avg_throughput"])
                vl_thr = float(vl_lang["avg_throughput"])
                winner = "SGLang" if sg_thr > vl_thr else "vLLM"
                report.append(f"- **{language.capitalize()}**: {winner} performs better ({max(sg_thr, vl_thr):.1f} tok/s)")

    report.append("\n---\n")
    report.append(f"*Benchmark conducted on RTX 5090 (32GB) with TinyLlama 1.1B*")
    report.append(f"*Total tests per framework: 1800 (120 iterations × 15 prompts)*")

    return "\n".join(report)

def main():
    print("📊 Analyzing benchmark results...")

    # 최신 결과 파일 찾기
    sglang_file, vllm_file = load_latest_results()

    if not sglang_file or not vllm_file:
        print("Waiting for benchmark results...")
        return

    print(f"Found SGLang results: {sglang_file}")
    print(f"Found vLLM results: {vllm_file}")

    # 데이터 로드
    sglang_df = pd.read_csv(sglang_file)
    vllm_df = pd.read_csv(vllm_file)

    print(f"SGLang tests: {len(sglang_df)}")
    print(f"vLLM tests: {len(vllm_df)}")

    # 상세 분석
    analysis = analyze_detailed_metrics(sglang_df, vllm_df)

    # JSON 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"/home/multi-model-server/detailed_analysis_{timestamp}.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"💾 Detailed analysis saved to {analysis_file}")

    # 마크다운 보고서 생성
    report = create_markdown_report(analysis)
    report_file = f"/home/multi-model-server/BENCHMARK_REPORT_{timestamp}.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"📝 Markdown report saved to {report_file}")

    # 콘솔에도 출력
    print("\n" + "="*80)
    print(report)
    print("="*80)

if __name__ == "__main__":
    main()
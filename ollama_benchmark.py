#!/usr/bin/env python3
import requests
import time
import csv
from datetime import datetime
import json

# 테스트 설정
NUM_TESTS = 100
BASE_URL = "http://localhost:11434/api/generate"

# 프롬프트 설정
prompts = {
    "english": "Sing a beautiful song about spring with blooming flowers.",
    "chinese": "唱一首关于春天百花盛开之美的歌曲。",
    "korean": "봄에 피어나는 꽃들의 아름다움을 노래해주세요."
}

def test_ollama(model, prompt, language, iteration):
    """Ollama API 테스트"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100,
            "temperature": 0.7
        }
    }

    start_time = time.time()
    try:
        response = requests.post(BASE_URL, json=payload, timeout=30)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            total_tokens = len(response_text.split())  # 간단한 토큰 계산

            latency = (end_time - start_time) * 1000  # ms
            throughput = total_tokens / (end_time - start_time) if end_time > start_time else 0

            return {
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "language": language,
                "model": model,
                "prompt": prompt[:50] + "...",
                "response_preview": response_text[:100] + "...",
                "latency_ms": round(latency, 2),
                "total_tokens": total_tokens,
                "throughput_tok_s": round(throughput, 2),
                "status": "success",
                "error": None
            }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "language": language,
            "model": model,
            "prompt": prompt[:50] + "...",
            "response_preview": None,
            "latency_ms": None,
            "total_tokens": 0,
            "throughput_tok_s": 0,
            "status": "failed",
            "error": str(e)
        }

def main():
    # CSV 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ollama_benchmark_{timestamp}.csv"

    # 결과 저장용 리스트
    results = []

    print(f"🚀 Ollama 멀티모델 벤치마크 시작")
    print(f"📊 테스트 수: {NUM_TESTS}개 (언어별)")
    print(f"💾 결과 파일: {csv_filename}\n")

    # 테스트할 모델들
    models = ["tinyllama:1.1b", "qwen2.5:3b", "yi:6b"]

    for model in models:
        print(f"\n🔬 {model} 테스트 중...")
        model_results = []

        for lang, prompt in prompts.items():
            print(f"  📝 {lang} 테스트 중...")
            for i in range(1, NUM_TESTS + 1):
                result = test_ollama(model, prompt, lang, i)
                results.append(result)
                model_results.append(result)

                if i % 10 == 0:
                    success_count = sum(1 for r in model_results if r['status'] == 'success')
                    avg_latency = sum(r['latency_ms'] for r in model_results if r['latency_ms']) / len([r for r in model_results if r['latency_ms']])
                    print(f"    진행: {i}/{NUM_TESTS} - 성공률: {success_count}/{len(model_results)} - 평균 지연: {avg_latency:.2f}ms")

    # CSV 파일 저장
    if results:
        fieldnames = results[0].keys()
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ 테스트 완료! 결과가 {csv_filename}에 저장되었습니다.")

        # 통계 출력
        print("\n📊 전체 통계:")
        for model in models:
            model_results = [r for r in results if r['model'] == model]
            if model_results:
                success_results = [r for r in model_results if r['status'] == 'success']
                if success_results:
                    avg_latency = sum(r['latency_ms'] for r in success_results) / len(success_results)
                    avg_throughput = sum(r['throughput_tok_s'] for r in success_results) / len(success_results)
                    success_rate = (len(success_results) / len(model_results)) * 100

                    print(f"\n{model}:")
                    print(f"  성공률: {success_rate:.1f}%")
                    print(f"  평균 지연시간: {avg_latency:.2f}ms")
                    print(f"  평균 처리량: {avg_throughput:.2f} tok/s")

if __name__ == "__main__":
    main()
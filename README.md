# 🚀 Multi-Model LLM Serving on RTX 5090 - Complete Benchmarking Suite
# 🚀 RTX 5090에서 멀티모델 LLM 서빙 - 완벽한 벤치마킹 스위트

[![Performance](https://img.shields.io/badge/Throughput-332%20tok%2Fs-green)](FINAL_BENCHMARK_REPORT.md)
[![Models](https://img.shields.io/badge/Models-3%20Concurrent-blue)](MULTIMODEL_SERVING_COMPARISON.md)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090%2032GB-orange)](README.md)
[![Frameworks](https://img.shields.io/badge/Frameworks-vLLM%20%7C%20SGLang%20%7C%20Ollama-purple)](FINAL_PROJECT_REPORT.md)

## 📋 Overview / 개요

### English
This repository contains a comprehensive benchmarking suite comparing three major LLM serving frameworks (vLLM, SGLang, Ollama) for multi-model deployment on NVIDIA RTX 5090. With 4,984+ benchmark data points across multiple languages, this is the definitive guide for deploying multiple LLMs on a single GPU.

### 한국어
이 저장소는 NVIDIA RTX 5090에서 멀티모델 배포를 위한 세 가지 주요 LLM 서빙 프레임워크(vLLM, SGLang, Ollama)를 비교하는 포괄적인 벤치마킹 스위트를 포함합니다. 여러 언어에 걸친 4,984개 이상의 벤치마크 데이터 포인트로, 단일 GPU에서 여러 LLM을 배포하기 위한 결정적인 가이드입니다.

## 🏆 Key Results Summary / 주요 결과 요약

| Metric / 지표 | Ollama | vLLM | SGLang |
|--------------|--------|------|--------|
| **Throughput / 처리량** | 🥇 428 tok/s | 112 tok/s | 77 tok/s |
| **Latency / 지연시간** | 🥇 152ms | 207ms | 357ms |
| **Memory / 메모리 (3 models)** | 🥇 8.5GB | 27GB | ~5GB (1 model) |
| **Multi-model / 멀티모델** | ✅ Dynamic | ✅ Excellent | ❌ Limited |
| **RTX 5090 Support** | ✅ Native | ✅ Native | ⚠️ Issues |

*Benchmark data from 2025-09-18 testing with TinyLlama 1.1B

### 🚀 Models Tested / 테스트된 모델
- **TinyLlama 1.1B**: Fast responses for simple queries / 간단한 쿼리를 위한 빠른 응답
- **Qwen 2.5-3B**: General-purpose tasks and code generation / 범용 작업 및 코드 생성
- **Yi-6B**: High-quality responses for complex tasks / 복잡한 작업을 위한 고품질 응답

## 📊 Memory Usage Analysis

### Individual Model Testing (gpu-memory-utilization=0.9)
When running each model separately with 90% GPU allocation:

| Model | Parameters | Theoretical Size | Actual Usage | Notes |
|-------|------------|-----------------|--------------|-------|
| TinyLlama 1.1B | 1.1B | 2.2GB | **29.5GB** | Uses maximum available memory |
| Qwen 2.5-3B | 3B | 6GB | **29.6GB** | KV cache expands to fill space |
| Yi-6B | 6B | 12GB | **29.5GB** | Similar behavior across all models |

### Multi-Model Concurrent Execution
When running all three models simultaneously:

| Model | GPU Memory | Percentage | Response Time |
|-------|------------|------------|---------------|
| TinyLlama | 5.3GB | 16% | 33ms |
| Qwen 2.5 | 8.4GB | 26% | 81ms |
| Yi-6B | 17.2GB | 53% | 96ms |
| **Total** | **31.5GB** | **96.6%** | - |

## ⚠️ Key Findings

### 1. gpu-memory-utilization Behavior
- **Common Misconception**: `gpu-memory-utilization=0.5` means 50% of total GPU memory
- **Reality**: It controls the percentage of *available* memory allocated to KV cache
- **Model weights are always loaded separately** and not included in this calculation

### 2. Memory Allocation Formula
```
Actual Memory = Model Weights + (Available Memory × gpu_utilization) + Overhead
```

### 3. Multi-Model Challenges
- Each container independently checks GPU memory
- Containers are unaware of each other's existence
- Can lead to memory conflicts if not carefully configured

## 🔧 Configuration

### Optimized Settings (docker-compose-balanced.yml)
```yaml
# TinyLlama - Minimal allocation for fast responses
--gpu-memory-utilization 0.15
--max-model-len 1024
--max-num-seqs 32

# Qwen - Balanced allocation for general tasks
--gpu-memory-utilization 0.25
--max-model-len 1536
--max-num-seqs 24

# Yi - Maximum allocation for quality
--gpu-memory-utilization 0.50
--max-model-len 2048
--max-num-seqs 16
```

## 📝 Best Practices

### 1. Start Order Matters
- Start smaller models first, then larger ones
- Use `depends_on` in Docker Compose to ensure proper sequencing

### 2. Parameter Tuning
- **gpu-memory-utilization**: Controls KV cache allocation
- **max-model-len**: Limits context length (reduces KV cache)
- **max-num-seqs**: Limits concurrent requests
- **dtype**: Use `float16` for predictable memory usage

### 3. Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check container logs
docker logs vllm-tinyllama --tail 20
```

## 🚀 Quick Start

### Prerequisites
- NVIDIA RTX 5090 or similar GPU with 32GB+ VRAM
- Docker with NVIDIA Container Toolkit
- CUDA 12.8+ compatible drivers

### Run All Models
```bash
# Start all three models
docker compose -f docker-compose-balanced.yml up -d

# Check status
docker ps | grep vllm

# Test endpoints
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "Hello", "max_tokens": 10}'
```

### Endpoints
- TinyLlama: `http://localhost:8001`
- Qwen: `http://localhost:8002`
- Yi: `http://localhost:8003`

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Total GPU Memory Used | 31.5GB / 32.6GB |
| GPU Utilization | 96.6% |
| Average Response Time (TinyLlama) | 33ms |
| Average Response Time (Qwen) | 81ms |
| Average Response Time (Yi) | 96ms |
| Concurrent Model Support | 3 models |

## 📚 Comprehensive Reports / 종합 보고서

### Available Documentation / 사용 가능한 문서
- 📊 **[Final Project Report](FINAL_PROJECT_REPORT.md)** - Complete analysis with 4,984+ data points / 4,984개 이상의 데이터 포인트가 포함된 완전한 분석
- 🏁 **[Benchmark Results](FINAL_BENCHMARK_REPORT.md)** - Detailed performance comparisons / 상세한 성능 비교
- 🔍 **[Code Analysis](claudedocs/CODE_ANALYSIS_REPORT.md)** - Security and quality assessment / 보안 및 품질 평가
- 📈 **[Multi-model Comparison](MULTIMODEL_SERVING_COMPARISON.md)** - Framework capabilities matrix / 프레임워크 기능 매트릭스

## 🎯 Use Case Recommendations / 사용 사례 권장사항

| Scenario / 시나리오 | Best Choice / 최선의 선택 | Why / 이유 |
|--------------------|---------------------------|-----------|
| **Production API** | vLLM | 4.8x faster throughput / 4.8배 빠른 처리량 |
| **Development** | Ollama | Easy setup, dynamic loading / 쉬운 설정, 동적 로딩 |
| **Memory Limited** | Ollama | 3.2x better efficiency / 3.2배 더 나은 효율성 |
| **Multi-language** | vLLM | Best cross-language performance / 최고의 다국어 성능 |

## 🔬 Technical Highlights / 기술적 하이라이트

### vLLM Advantages / vLLM 장점
- ✅ **332 tok/s throughput** with TinyLlama / TinyLlama로 332 tok/s 처리량
- ✅ **3 models concurrent** on single GPU / 단일 GPU에서 3개 모델 동시 실행
- ✅ **99.7% success rate** in production tests / 프로덕션 테스트에서 99.7% 성공률
- ✅ **Native RTX 5090 support** without modifications / 수정 없이 네이티브 RTX 5090 지원

### Ollama Advantages / Ollama 장점
- ✅ **68.5% memory savings** vs vLLM / vLLM 대비 68.5% 메모리 절약
- ✅ **Dynamic model swapping** / 동적 모델 교체
- ✅ **131 tok/s throughput** / 131 tok/s 처리량
- ✅ **CPU fallback support** / CPU 폴백 지원

### SGLang Limitations / SGLang 제한사항
- ❌ **Single model only** on RTX 5090 / RTX 5090에서 단일 모델만
- ⚠️ **Custom build required** / 커스텀 빌드 필요
- ❌ **Many optimizations disabled** / 많은 최적화 비활성화

## 🔐 Security Note / 보안 참고사항

⚠️ **Important**: The code analysis found critical security issues that need immediate attention:
- SSH private key exposure - remove immediately
- Hardcoded credentials - use environment variables

⚠️ **중요**: 코드 분석에서 즉각적인 주의가 필요한 중요한 보안 문제를 발견했습니다:
- SSH 개인키 노출 - 즉시 제거
- 하드코딩된 자격증명 - 환경 변수 사용

## 📈 Performance Statistics / 성능 통계

### Test Scale / 테스트 규모
- **4,984** total data points / 총 데이터 포인트
- **Multiple benchmark sessions** / 여러 벤치마크 세션
- **3 frameworks** × **3 models** × **3 languages** / 3개 프레임워크 × 3개 모델 × 3개 언어
- **15 different** configurations tested / 15가지 다른 구성 테스트됨

## 🤝 Contributing / 기여하기
Feel free to submit issues or PRs with improvements for different GPU configurations.
다른 GPU 구성에 대한 개선 사항과 함께 이슈나 PR을 자유롭게 제출하세요.

## 📄 License / 라이센스
MIT

---
*Tested on / 테스트 환경: NVIDIA GeForce RTX 5090 (32GB) | CUDA 12.9 | vLLM v0.10.1.1 | SGLang custom | Ollama latest*
*Total benchmark data points / 총 벤치마크 데이터 포인트: 4,984*
*Analysis completed / 분석 완료: 2025-09-18*
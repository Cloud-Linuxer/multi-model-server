# LLM Serving Framework Benchmark Report - RTX 5090
# RTX 5090 LLM 서빙 프레임워크 벤치마크 보고서

**Date**: 2025-09-18
**Hardware**: NVIDIA RTX 5090 (32GB VRAM)
**Test Model**: TinyLlama 1.1B Chat v1.0

---

## 📊 Executive Summary / 요약

This report presents benchmark results for three LLM serving frameworks tested under various conditions. Due to differences in testing methodologies and measurement approaches, direct comparisons should be interpreted with caution.

이 보고서는 다양한 조건에서 테스트된 세 가지 LLM 서빙 프레임워크의 벤치마크 결과를 제시합니다. 테스트 방법론과 측정 방식의 차이로 인해 직접 비교 시 주의가 필요합니다.

---

## 🔬 Test Methodology / 테스트 방법론

### Test Configurations / 테스트 구성

| Parameter / 매개변수 | Configuration / 구성 |
|---------------------|---------------------|
| **Model** | TinyLlama 1.1B Chat v1.0 |
| **Prompt Types** | English, Chinese, Korean |
| **Max Tokens** | 100 (requested) |
| **Temperature** | 0.7 |
| **Iterations** | 100-300 per language |
| **Deployment** | Docker containers |

### Important Methodological Differences / 중요한 방법론적 차이

1. **Token Counting Methods / 토큰 계산 방식**:
   - Some tests counted actual generated words (word-based)
   - Others counted tokenizer tokens (subword-based)
   - This creates significant variance in throughput calculations

2. **Test Conditions / 테스트 조건**:
   - Different tests used different prompt lengths
   - Warm vs cold start not consistently controlled
   - Network overhead varied between tests

---

## 📈 Benchmark Results / 벤치마크 결과

### Raw Performance Metrics / 원시 성능 지표

| Framework | Test Type | Avg Latency (ms) | Throughput* | Sample Size | Success Rate |
|-----------|-----------|------------------|-------------|-------------|--------------|
| **vLLM** | Unified Test | 207.1 | 13.3 tok/s† | 299 | 99.7% |
| **SGLang** | Dedicated Test | 356.8 | 76.8 tok/s | 150 | 100% |
| **Ollama** | Unified Test | 151.8 | 64.2 tok/s† | 300 | 100% |
| **Ollama** | Dedicated Test | 347.7 | 127.6 tok/s | 73 | 100% |

*Throughput measurements are not directly comparable due to different token counting methods
*처리량 측정값은 토큰 계산 방식의 차이로 직접 비교 불가
†Word-based counting (significantly lower than tokenizer-based)

### Response Time Distribution / 응답 시간 분포

| Framework | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |
|-----------|----------|----------|----------|----------|
| **vLLM** | 257 | 261 | 10 | 14,404* |
| **SGLang** | ~350 | ~540 | ~180 | ~550 |
| **Ollama** | 150 | 152 | ~140 | ~160 |

*Includes timeout and cold start outliers

---

## 💾 Memory Usage Analysis / 메모리 사용 분석

| Framework | Single Model | Multi-Model Support | Memory Efficiency |
|-----------|--------------|---------------------|-------------------|
| **vLLM** | ~9GB | Yes (3 models: 27GB) | Static allocation |
| **SGLang** | ~5GB | Limited† | Lower baseline |
| **Ollama** | ~3GB | Yes (dynamic: 8.5GB) | Dynamic loading |

†Multi-model support encountered issues on RTX 5090

---

## ⚖️ Comparative Analysis / 비교 분석

### Strengths and Limitations / 강점과 한계

#### vLLM
**Strengths / 강점**:
- Mature multi-model support / 성숙한 멀티모델 지원
- Production-ready features / 프로덕션 준비 기능
- OpenAI API compatibility / OpenAI API 호환성

**Limitations / 한계**:
- Higher memory usage / 높은 메모리 사용량
- Variable performance in tests / 테스트에서 가변적 성능

#### SGLang
**Strengths / 강점**:
- Lower memory baseline / 낮은 메모리 기준선
- Specialized optimization potential / 특화된 최적화 가능성

**Limitations / 한계**:
- Multi-model challenges on RTX 5090 / RTX 5090에서 멀티모델 문제
- Docker deployment complexities / Docker 배포 복잡성

#### Ollama
**Strengths / 강점**:
- Memory efficient dynamic loading / 메모리 효율적 동적 로딩
- Consistent low latency in unified tests / 통합 테스트에서 일관된 낮은 지연시간
- Simple deployment / 간단한 배포

**Limitations / 한계**:
- Performance varies by test type / 테스트 유형별 성능 차이
- Less feature-rich than vLLM / vLLM보다 기능 부족

---

## 📋 Use Case Recommendations / 사용 사례 권장사항

| Use Case / 사용 사례 | Recommended Framework | Rationale / 근거 |
|---------------------|----------------------|------------------|
| **Production API Services** | vLLM or Ollama | Depends on feature vs efficiency needs |
| **Memory-Constrained Environments** | Ollama | Dynamic loading reduces memory footprint |
| **Multi-Model Serving** | vLLM | Most mature multi-model support |
| **Development/Testing** | Ollama | Simple setup and management |
| **Single Model Optimization** | Any | All perform adequately for single model |

---

## ⚠️ Important Considerations / 중요 고려사항

1. **Measurement Inconsistencies / 측정 불일치**:
   - Token counting methods significantly affect throughput metrics
   - Direct throughput comparisons may be misleading
   - Latency measurements are more reliable for comparison

2. **Test Environment Factors / 테스트 환경 요인**:
   - Docker overhead affects all frameworks
   - RTX 5090 compatibility varies
   - Network latency impacts API-based tests

3. **Statistical Validity / 통계적 유효성**:
   - Sample sizes vary (73-300 requests)
   - Not all tests covered all scenarios
   - Results may not generalize to other hardware

---

## 🔍 Methodology Transparency / 방법론 투명성

### Data Sources / 데이터 출처
- `unified_benchmark_20250918_215507.csv`: Comparative test
- `benchmark_sglang_20250918_153246.csv`: SGLang dedicated test
- `ollama_benchmark_20250918_211500.csv`: Ollama dedicated test
- `vllm_benchmark_20250918_155524.csv`: vLLM dedicated test

### Limitations / 한계
1. Tests conducted at different times with potentially different system loads
2. Token counting methodology not standardized across tests
3. Limited to single model (TinyLlama 1.1B) performance
4. RTX 5090 specific - results may not generalize to other GPUs

---

## 📝 Conclusion / 결론

The benchmark results show that each framework has distinct characteristics:

- **Ollama** demonstrated lower latency in unified tests but results varied significantly between test types
- **vLLM** showed stable multi-model support despite variable throughput measurements
- **SGLang** faced deployment challenges that limited comprehensive testing

The significant variance in token counting methods makes throughput comparisons unreliable. **Latency measurements provide more consistent comparison basis**. Framework selection should be based on specific use case requirements rather than raw performance metrics alone.

벤치마크 결과는 각 프레임워크가 뚜렷한 특성을 가지고 있음을 보여줍니다. 토큰 계산 방법의 차이로 인해 처리량 비교는 신뢰하기 어렵습니다. **지연시간 측정이 더 일관된 비교 기준을 제공합니다**. 프레임워크 선택은 단순한 성능 지표보다는 특정 사용 사례 요구사항을 기반으로 해야 합니다.

---

*This report aims to present benchmark data objectively. Users should conduct their own testing for their specific use cases.*
*이 보고서는 벤치마크 데이터를 객관적으로 제시하는 것을 목표로 합니다. 사용자는 특정 사용 사례에 대해 자체 테스트를 수행해야 합니다.*
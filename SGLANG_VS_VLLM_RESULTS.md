# SGLang vs vLLM 성능 비교 결과 (RTX 5090)

## 📊 테스트 환경
- **GPU**: NVIDIA RTX 5090 (32GB)
- **CUDA**: 12.9
- **날짜**: 2025-09-18
- **테스트 모델**: TinyLlama 1.1B, Qwen 2.5-3B, Yi-6B
- **벤치마크 수행**: 2025-09-18 15:09

## ✅ 실행 가능성 비교

### vLLM
| 모델 | 상태 | GPU 메모리 설정 | 실행 가능 |
|------|------|-----------------|-----------|
| TinyLlama 1.1B | ✅ | 0.15 | 성공 |
| Qwen 2.5-3B | ✅ | 0.25 | 성공 |
| Yi-6B | ✅ | 0.45 | 성공 |
| **3개 동시 실행** | ✅ | 총 0.85 | **성공** |

### SGLang
| 모델 | 상태 | 메모리 설정 | 실행 가능 |
|------|------|-------------|-----------|
| TinyLlama 1.1B | ✅ | 0.85 | 성공 (단독) |
| Qwen 2.5-3B | ❌ | - | 메모리 부족 |
| Yi-6B | ❌ | - | 메모리 부족 |
| **3개 동시 실행** | ❌ | - | **불가능** |

## ⚡ 성능 벤치마크 결과 (TinyLlama 1.1B)

### 처리량 (Throughput)
| 메트릭 | SGLang | vLLM |
|--------|--------|------|
| 평균 처리량 | 77.0 tokens/s | **177.5 tokens/s** |
| 최대 처리량 | 202.7 tokens/s | **381.2 tokens/s** |
| **승자** | - | **vLLM (2.3x 빠름)** |

### 지연시간 (Latency)
| 메트릭 | SGLang | vLLM |
|--------|--------|------|
| 평균 지연시간 | **0.131s** | 1.161s |
| P50 지연시간 | 0.025s | **0.009s** |
| P95 지연시간 | **0.543s** | 15.841s |

### 안정성
- **SGLang**: 일관된 지연시간 (0.01s ~ 0.56s)
- **vLLM**: 변동성 큼 (0.01s ~ 28.25s), 하지만 대부분 빠른 응답

## 🔍 핵심 발견사항

### 1. 멀티모델 서빙 능력
- **vLLM**: ✅ 3개 모델 동시 실행 가능
  - 각 컨테이너가 독립적으로 메모리 관리
  - gpu-memory-utilization 파라미터로 정밀 제어

- **SGLang**: ❌ 단일 모델만 실행 가능
  - mem-fraction-static이 전체 GPU 기준
  - 컨테이너 간 메모리 공유 불가
  - 두 번째 모델 시작 시 "Not enough memory" 오류

### 2. RTX 5090 호환성

#### vLLM
```bash
--enforce-eager  # JIT 컴파일 비활성화로 RTX 5090 지원
```
- 공식 이미지로 바로 실행 가능
- 안정적 작동

#### SGLang
```bash
--disable-cuda-graph \
--disable-custom-all-reduce \
--disable-flashinfer \
--disable-radix-cache \
--attention-backend torch_native
```
- 커스텀 빌드 필요 (PyTorch nightly + CUDA 12.8)
- 많은 최적화 기능 비활성화 필요
- RadixAttention (핵심 기능) 사용 불가

### 3. 성능 특성

| 항목 | vLLM | SGLang |
|------|------|--------|
| **처리량 (Throughput)** | 우수 (177.5 tok/s) | 보통 (77.0 tok/s) |
| **지연시간 일관성** | 변동성 큼 | 일관됨 |
| **메모리 효율성** | 우수 (3개 모델) | 제한적 (1개 모델) |
| **설정 복잡도** | 간단 | 복잡 |
| **RTX 5090 지원** | 공식 지원 | 커스텀 빌드 |
| **안정성** | 높음 | 중간 |
| **최적화 기능** | 대부분 사용 가능 | 대부분 비활성화 |

## 💡 권장사항

### RTX 5090 사용자

#### 멀티모델 서빙이 필요한 경우
**vLLM 선택** ✅
- 3개 이상 모델 동시 실행 필요
- 안정성이 중요한 프로덕션 환경
- 간단한 설정 선호

#### 단일 모델 + 특수 기능이 필요한 경우
**SGLang 고려** ⚠️
- 구조화된 출력 (JSON, 코드) 필요
- 실험적 기능 사용 의향
- 커스텀 빌드 가능한 환경

### RTX 4090 이하 사용자
SGLang의 RadixAttention 등 고급 기능 활용 가능
- 반복적인 프롬프트 처리에 유리
- 더 나은 호환성

## 📝 실제 배포 예시

### vLLM 3개 모델 동시 배포 (성공)
```yaml
services:
  tinyllama:
    gpu-memory-utilization: 0.15
    max-model-len: 1024

  qwen:
    gpu-memory-utilization: 0.25
    max-model-len: 1536

  yi:
    gpu-memory-utilization: 0.45
    max-model-len: 2048

# 총 GPU 사용: 85%, 여유: 15%
```

### SGLang 배포 시도 (실패)
```bash
# TinyLlama만 실행 가능
--mem-fraction-static 0.85  # 전체 GPU의 85% 사용

# Qwen 시작 시도 → 실패
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

## 🎯 결론

**RTX 5090에서 멀티모델 서빙:**
- **vLLM**: ✅ 완벽 지원, 3개 모델 동시 실행 성공
- **SGLang**: ❌ 제한적, 단일 모델만 가능

**성능 벤치마크 결과:**
- **vLLM**: 평균 177.5 tokens/s (최대 381.2 tokens/s)
- **SGLang**: 평균 77.0 tokens/s (최대 202.7 tokens/s)
- **승자**: vLLM이 약 2.3배 빠른 처리량 달성

**선택 기준:**
1. 멀티모델 필요 → **vLLM**
2. 높은 처리량 필요 → **vLLM**
3. 안정성 중요 → **vLLM**
4. RTX 5090 사용 → **vLLM**
5. 지연시간 일관성 중요 → SGLang 고려
6. 단일 모델 + 실험적 → SGLang 고려

---

*테스트 완료: 2025-09-18 15:09*
*환경: RTX 5090 (32GB) | CUDA 12.9 | Docker*
*벤치마크: 40개 프롬프트 × 4개 출력 길이 = 160개 테스트*
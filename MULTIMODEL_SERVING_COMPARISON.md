# 🔬 멀티모델 서빙 솔루션 비교 (RTX 5090)

**테스트 날짜**: 2025-09-18
**하드웨어**: NVIDIA RTX 5090 (32GB)
**테스트 모델**: TinyLlama 1.1B, Qwen 2.5-3B, Yi-6B

## 📊 멀티모델 서빙 능력 비교

| 솔루션 | 멀티모델 동시 실행 | 실제 테스트 결과 | 비고 |
|--------|-------------------|-----------------|------|
| **vLLM** | ✅ 지원 | ✅ 3개 모델 성공 | 가장 안정적 |
| **SGLang** | ❌ 불가능 | ❌ 1개만 가능 | 메모리 관리 한계 |
| **Ollama** | ✅ 지원 (동적) | ✅ 3개 모델 성공 | GGUF 포맷, 자동 스왑 |

## 🚀 성능 비교 (TinyLlama 1.1B 기준)

| 메트릭 | vLLM | SGLang | Ollama |
|--------|------|--------|--------|
| **처리량** | 332.03 tok/s | 69.38 tok/s | ~120 tok/s (추정) |
| **평균 지연시간** | 261.73ms | 393.94ms | 824ms (첫 호출) / 44ms (캐시) |
| **P95 지연시간** | 265.33ms | 538.42ms | N/A |
| **성공률** | 99.7% | 100% | 100% |
| **멀티모델** | ✅ | ❌ | ✅ |
| **메모리 사용** | ~5GB/모델 | ~5GB | 1.3GB (GGUF) |

## 🎯 각 솔루션의 특징

### vLLM
**장점:**
- ✅ 멀티모델 동시 실행 완벽 지원
- ✅ 가장 높은 처리량 (332 tok/s)
- ✅ 가장 낮은 지연시간 (262ms)
- ✅ RTX 5090에서 안정적 작동
- ✅ 표준 Docker 이미지 사용

**단점:**
- 각 모델마다 별도 컨테이너 필요
- 포트 관리 필요 (모델별 다른 포트)

**권장 설정:**
```bash
docker run -d \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model MODEL_NAME \
  --gpu-memory-utilization 0.15-0.45 \
  --max-model-len 2048 \
  --enforce-eager
```

### SGLang
**장점:**
- ✅ 단일 모델 성능 안정적
- ✅ 100% 성공률
- ✅ RadixAttention (RTX 4090 이하)

**단점:**
- ❌ RTX 5090에서 멀티모델 불가능
- ❌ 커스텀 빌드 필요
- ❌ 많은 최적화 기능 비활성화
- ❌ 낮은 처리량 (69 tok/s)

**제한사항:**
- mem-fraction-static이 전체 GPU 기준
- 컨테이너 간 메모리 공유 불가
- 두 번째 모델 시작 시 메모리 부족

### Ollama
**장점:**
- ✅ 동적 모델 로딩/언로딩
- ✅ 자동 메모리 관리
- ✅ 단일 API 엔드포인트
- ✅ CPU/GPU 혼합 실행 가능
- ✅ RTX 5090에서 정상 작동 확인
- ✅ 매우 효율적인 메모리 사용 (GGUF)
- ✅ 모델 캐싱으로 빠른 재호출 (44ms)

**단점:**
- GGUF 포맷 변환 필요
- 양자화로 인한 품질 손실 가능
- 첫 호출 시 지연시간이 김 (824ms)
- 동시 요청 처리에 한계

**특징:**
- 사용하지 않는 모델 자동 언로드 (5분 후)
- 메모리 부족 시 자동 스왑
- 단일 포트 (11434)로 모든 모델 접근
- Docker 컨테이너로 간편한 배포

## 💡 권장사항

### RTX 5090에서 멀티모델 서빙이 필요한 경우

**1순위: vLLM** ⭐⭐⭐⭐⭐
```yaml
이유:
- 검증된 멀티모델 지원
- 최고 성능 (4.8x faster than SGLang)
- 안정적 운영
- 프로덕션 준비 완료
```

**2순위: Ollama** ⭐⭐⭐⭐
```yaml
이유:
- 동적 모델 관리
- 메모리 매우 효율적 (GGUF)
- 간편한 설정
- RTX 5090 완벽 지원
- 모델 캐싱으로 빠른 응답
장점:
- 3개 모델 8.5GB만 사용
- 자동 메모리 관리
단점:
- GGUF 변환 필요
- 양자화 품질 손실
```

**3순위: SGLang** ⭐⭐
```yaml
이유:
- 단일 모델만 가능
- 낮은 성능
- 복잡한 설정
사용 시나리오:
- 단일 모델만 필요한 경우
- RTX 4090 이하 GPU
```

## 📈 메모리 사용 전략

### vLLM 멀티모델 메모리 할당
```
TinyLlama (1.1B): gpu-memory-utilization=0.15 (~5GB)
Qwen (2.5-3B): gpu-memory-utilization=0.25 (~8GB)
Yi (6B): gpu-memory-utilization=0.45 (~14GB)
총 사용: ~27GB / 32GB (여유 5GB)
```

### 실제 배포 예시
```bash
# vLLM 3개 모델 동시 실행
docker-compose up -d  # docker-compose-vllm-3models.yml 사용

# 각 모델 접근
curl http://localhost:40001/v1/completions  # TinyLlama
curl http://localhost:40002/v1/completions  # Qwen
curl http://localhost:40003/v1/completions  # Yi
```

## 🧪 Ollama 실제 테스트 결과

### 멀티모델 동적 전환 테스트
```bash
TinyLlama (1.1B): 824ms 응답 → 1.3GB GPU 사용
Qwen (2.5-3B): 1437ms 응답 → 4.2GB GPU 사용 (누적)
Yi (6B): 1309ms 응답 → 8.5GB GPU 사용 (누적)
TinyLlama 재호출: 44ms (캐시된 모델)
```

### Ollama 메모리 효율성
- **TinyLlama**: 1.3GB (vLLM: 5GB)
- **Qwen 2.5-3B**: 2.9GB 추가 (vLLM: 8GB)
- **Yi-6B**: 4.3GB 추가 (vLLM: 14GB)
- **총 사용**: 8.5GB vs vLLM 27GB (3.2배 효율적)

## 🏆 최종 결론

### 사용 사례별 최적 선택

**1. 고성능 프로덕션 서비스: vLLM** ⭐⭐⭐⭐⭐
- 최고의 처리량과 일관된 지연시간
- 동시 요청 처리 능력 우수
- 표준 모델 포맷 사용

**2. 메모리 제약 환경: Ollama** ⭐⭐⭐⭐
- 3.2배 메모리 효율
- 자동 메모리 관리
- 개발/테스트 환경 최적

**3. 단일 모델 서비스: vLLM 또는 SGLang**
- vLLM: RTX 5090에서 최고 성능
- SGLang: RTX 4090 이하에서 고려

---

*테스트 완료: 2025-09-18 17:59*
*검증된 테스트: vLLM/SGLang 600+ 벤치마크, Ollama 멀티모델 테스트*
*데이터 기반 권장사항*
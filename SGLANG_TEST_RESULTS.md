# SGLang 테스트 결과 - RTX 5090

## 📋 테스트 개요
- **테스트 일자**: 2025-09-18
- **GPU**: NVIDIA RTX 5090 (32GB)
- **CUDA 버전**: 12.9
- **SGLang 이미지**: lmsysorg/sglang:latest

## 🚨 호환성 문제 발견

### RTX 5090 호환성 이슈
SGLang의 현재 PyTorch 버전이 RTX 5090의 CUDA capability (sm_120)를 지원하지 않음:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

### 발생한 오류
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

## 🔍 근본 원인 분석

1. **CUDA Capability 미스매치**
   - RTX 5090: CUDA capability 12.0 (sm_120)
   - SGLang PyTorch: 최대 지원 9.0 (sm_90)

2. **PyTorch 버전 문제**
   - SGLang 이미지의 PyTorch가 아직 RTX 5090을 지원하지 않음
   - CUDA 12.8/12.9 지원 필요

3. **커널 컴파일 실패**
   - RTX 5090용 CUDA 커널이 컴파일되지 않음
   - `--disable-cuda-graph` 플래그로도 해결 안됨

## 📊 vLLM vs SGLang 비교 (이론적 분석)

### vLLM (실제 테스트 완료)
| 항목 | 결과 |
|------|------|
| RTX 5090 지원 | ✅ 작동 (--enforce-eager 필요) |
| 메모리 효율성 | 31.5GB/32GB 사용 |
| 3개 모델 동시 실행 | ✅ 성공 |
| 응답 시간 | TinyLlama: 33ms, Qwen: 81ms, Yi: 96ms |

### SGLang (호환성 문제로 미작동)
| 항목 | 상태 |
|------|------|
| RTX 5090 지원 | ❌ PyTorch 버전 문제 |
| RadixAttention | 테스트 불가 |
| 메모리 효율성 | 테스트 불가 |
| 성능 | 측정 불가 |

## 🛠️ 해결 시도

### 1. 환경 변수 설정
```yaml
environment:
  - CUDA_LAUNCH_BLOCKING=1
  - TORCH_CUDA_ARCH_LIST="8.6;9.0"
```
**결과**: 실패 - RTX 5090은 sm_120 필요

### 2. 플래그 조정
- `--disable-cuda-graph` 제거/추가
- `--enable-radix-attention` 제거 (최신 버전에서 지원 안함)
- `--dtype float16` 설정

**결과**: 모두 실패 - 근본적인 CUDA 호환성 문제

### 3. 컨테이너 재시작
여러 설정 조합으로 시도했으나 모두 동일한 CUDA 오류 발생

## 💡 결론 및 권장사항

### 현재 상황
1. **RTX 5090에서 SGLang 사용 불가**
   - PyTorch/CUDA 버전 업데이트 필요
   - SGLang 팀의 공식 지원 대기 필요

2. **vLLM이 유일한 선택**
   - RTX 5090에서 안정적 작동
   - --enforce-eager 플래그로 호환성 문제 해결
   - 3개 모델 동시 서빙 성공

### 향후 계획
1. **SGLang 업데이트 모니터링**
   - PyTorch 2.5+ 버전 지원 시 재테스트
   - CUDA 12.8/12.9 공식 지원 확인

2. **대안 고려**
   - RTX 4090 등 이전 세대 GPU에서 SGLang 테스트
   - vLLM 최적화에 집중

3. **임시 해결책**
   - vLLM 사용 계속
   - SGLang의 RadixAttention 개념을 vLLM에 응용 가능성 연구

## 📝 테스트 명령어 기록

```bash
# SGLang 이미지 다운로드
docker pull lmsysorg/sglang:latest

# 컨테이너 실행 시도
docker compose -f docker-compose-sglang.yml up -d tinyllama-sglang

# 로그 확인
docker logs sglang-tinyllama --tail 100

# 오류 발생 - CUDA kernel error
```

## 🔗 관련 이슈
- SGLang GitHub 이슈: RTX 5090 지원 요청 필요
- PyTorch GitHub: CUDA 12.0 (sm_120) 지원 추적
- NVIDIA 포럼: RTX 5090 + PyTorch 호환성 논의

---

**테스트 수행자**: Claude Code
**검증 상태**: RTX 5090 호환성 문제로 SGLang 실행 불가
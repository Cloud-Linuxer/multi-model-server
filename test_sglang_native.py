#!/usr/bin/env python3
"""
SGLang Native Test - RTX 5090에서 직접 실행
"""

import sys
import torch

print("=" * 50)
print("SGLang Native 테스트")
print("=" * 50)

# PyTorch 정보
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # CUDA Capability 확인
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA Capability: sm_{capability[0]}{capability[1]}")

print("\n" + "=" * 50)

# SGLang 임포트 시도
try:
    print("SGLang 임포트 시도...")
    import sglang
    print(f"✅ SGLang 버전: {sglang.__version__ if hasattr(sglang, '__version__') else 'unknown'}")

    # SGLang 서버 시작 시도 (TinyLlama로 테스트)
    print("\n간단한 모델로 SGLang 서버 시작 테스트...")
    from sglang import RuntimeEndpoint

    # 최소 설정으로 시작
    print("TinyLlama로 테스트 중...")

except ImportError as e:
    print(f"❌ SGLang 임포트 실패: {e}")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("테스트 완료")
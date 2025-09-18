#!/usr/bin/env python3

import json
import requests
import sys

# Prometheus API
prometheus_url = "http://localhost:9091"

# Get targets
targets_response = requests.get(f"{prometheus_url}/api/v1/targets")
targets_data = targets_response.json()

print("📊 실제 Prometheus 타겟 상태")
print("=" * 40)

up_count = 0
down_count = 0

for target in targets_data['data']['activeTargets']:
    job = target['labels']['job']
    health = target['health']
    status = "✅" if health == "up" else "❌"
    print(f"{status} {job:15} : {health}")

    if health == "up":
        up_count += 1
    else:
        down_count += 1

print(f"\n총 {len(targets_data['data']['activeTargets'])}개 타겟:")
print(f"  ✅ {up_count}개 정상 작동")
print(f"  ❌ {down_count}개 다운")

# Check actual metrics
print("\n📈 수집된 메트릭 확인:")
print("-" * 40)

# Query API metrics
query_response = requests.get(f"{prometheus_url}/api/v1/query",
                            params={"query": "api_requests_total"})
query_data = query_response.json()

if query_data['status'] == 'success' and query_data['data']['result']:
    print(f"✅ API 요청 메트릭: {len(query_data['data']['result'])}개 시리즈")
    total_requests = sum(float(r['value'][1]) for r in query_data['data']['result'])
    print(f"   총 {int(total_requests)}개 요청 처리")
else:
    print("❌ API 메트릭 수집 안됨")

# System metrics
for metric in ['system_cpu_percent', 'system_memory_percent', 'system_disk_percent']:
    query_response = requests.get(f"{prometheus_url}/api/v1/query",
                                params={"query": metric})
    query_data = query_response.json()

    if query_data['status'] == 'success' and query_data['data']['result']:
        value = float(query_data['data']['result'][0]['value'][1])
        print(f"✅ {metric}: {value:.1f}%")
    else:
        print(f"❌ {metric}: 데이터 없음")
# vLLM Multi-Model Serving on RTX 5090

## 📋 Overview
This repository demonstrates how to run multiple LLM models simultaneously using vLLM on a single NVIDIA RTX 5090 GPU (32GB VRAM) with Docker containers.

## 🚀 Models Deployed
- **TinyLlama 1.1B**: Fast responses for simple queries
- **Qwen 2.5-3B**: General-purpose tasks and code generation
- **Yi-6B**: High-quality responses for complex tasks

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

## 🔬 Technical Details

### vLLM Memory Management
1. vLLM uses PagedAttention for efficient KV cache management
2. Memory is allocated in blocks, causing some overhead
3. Each model process cannot share CUDA context with others
4. Actual memory usage = Model weights × 2-2.5x (including KV cache and overhead)

### RTX 5090 Compatibility
- Requires `--enforce-eager` flag to avoid CUDA compilation issues
- Works best with vLLM latest version (not v0.6.3.post1)
- CUDA 12.8/12.9 compatible

## 📚 Lessons Learned

1. **gpu-memory-utilization is relative, not absolute**: The same setting results in different memory usage depending on what's already allocated
2. **Manual optimization beats automation**: Each GPU/model combination requires testing
3. **Monitor and adjust**: Real-world usage patterns may require tweaking
4. **Memory overhead is significant**: Models use 2-2.5x their weight size in practice

## 🤝 Contributing
Feel free to submit issues or PRs with improvements for different GPU configurations.

## 📄 License
MIT

---
*Tested on: NVIDIA GeForce RTX 5090 (32GB) | CUDA 12.9 | vLLM v0.10.1.1*
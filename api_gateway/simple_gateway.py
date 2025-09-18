from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import httpx
import time
import logging
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'model'])
request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
active_requests = Gauge('api_active_requests', 'Number of active requests')
model_requests = Counter('model_requests_total', 'Total requests per model', ['model'])
error_count = Counter('api_errors_total', 'Total API errors', ['error_type'])

# System metrics
cpu_usage = Gauge('system_cpu_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_percent', 'Memory usage percentage')
disk_usage = Gauge('system_disk_percent', 'Disk usage percentage')

# Model endpoints
MODEL_ENDPOINTS = {
    "qwen2p5_3b": "http://mock-model:8001",
    "llama32_3b": "http://mock-model:8002",
    "gemma2_2b": "http://mock-model:8003"
}

app = FastAPI(
    title="Multi-Model API Gateway",
    description="Simplified gateway for multiple LLM models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    request_count.labels(method='GET', endpoint='/', model='none').inc()
    return {
        "message": "Multi-Model API Gateway",
        "available_models": list(MODEL_ENDPOINTS.keys()),
        "endpoints": {
            "models": "/v1/models",
            "completions": "/v1/completions",
            "chat": "/v1/chat/completions",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Check health of the gateway"""
    return {
        "status": "healthy",
        "models": list(MODEL_ENDPOINTS.keys()),
        "timestamp": time.time()
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "system"}
            for model in MODEL_ENDPOINTS.keys()
        ]
    }

@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completion requests"""
    start_time = time.time()
    active_requests.inc()

    try:
        data = await request.json()
        model_name = data.get("model", "qwen2p5_3b")

        # Track metrics
        request_count.labels(method='POST', endpoint='/v1/completions', model=model_name).inc()
        model_requests.labels(model=model_name).inc()

        # Mock response for testing
        response = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": f"This is a mock response from {model_name}",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
        }

        # Record duration
        duration = time.time() - start_time
        request_duration.labels(method='POST', endpoint='/v1/completions').observe(duration)

        return response
    except Exception as e:
        error_count.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        active_requests.dec()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests"""
    start_time = time.time()
    active_requests.inc()

    try:
        data = await request.json()
        model_name = data.get("model", "qwen2p5_3b")

        # Track metrics
        request_count.labels(method='POST', endpoint='/v1/chat/completions', model=model_name).inc()
        model_requests.labels(model=model_name).inc()

        # Mock response for testing
        response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"This is a mock chat response from {model_name}"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
        }

        # Record duration
        duration = time.time() - start_time
        request_duration.labels(method='POST', endpoint='/v1/chat/completions').observe(duration)

        return response
    except Exception as e:
        error_count.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        active_requests.dec()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update system metrics
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
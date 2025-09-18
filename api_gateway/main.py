from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import asyncio
from typing import Dict, Any, Optional, List
import json
import time
import redis
import hashlib
from contextlib import asynccontextmanager
import logging
from prometheus_client import Counter, Histogram, generate_latest
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('/app/configs/model_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Metrics
request_counter = Counter('model_requests_total', 'Total requests per model', ['model'])
request_duration = Histogram('model_request_duration_seconds', 'Request duration', ['model'])
cache_hits = Counter('cache_hits_total', 'Cache hits')
cache_misses = Counter('cache_misses_total', 'Cache misses')

# Model endpoints
MODEL_ENDPOINTS = {
    "qwen2p5_3b": "http://qwen-model:8001",
    "llama32_3b": "http://llama-model:8002",
    "gemma2_2b": "http://gemma-model:8003"
}

# Redis client for caching
redis_client = None
if config['caching']['enabled']:
    try:
        redis_client = redis.Redis(
            host=config['caching']['redis_host'],
            port=config['caching']['redis_port'],
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis caching enabled")
    except:
        logger.warning("Redis connection failed, caching disabled")
        redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("API Gateway starting up...")
    yield
    # Shutdown
    logger.info("API Gateway shutting down...")

app = FastAPI(
    title="Multi-Model API Gateway",
    description="Unified gateway for multiple LLM models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelRouter:
    def __init__(self):
        self.current_model = 0
        self.model_list = list(MODEL_ENDPOINTS.keys())
        self.weights = config['load_balancing']['weights']
        self.weighted_list = []
        for model, weight in self.weights.items():
            self.weighted_list.extend([model] * weight)

    def get_next_model(self) -> str:
        """Get next model using weighted round-robin"""
        model = self.weighted_list[self.current_model % len(self.weighted_list)]
        self.current_model += 1
        return model

    def get_model_endpoint(self, model_name: Optional[str] = None) -> str:
        """Get endpoint for specific model or next in rotation"""
        if model_name and model_name in MODEL_ENDPOINTS:
            return MODEL_ENDPOINTS[model_name]
        return MODEL_ENDPOINTS[self.get_next_model()]

router = ModelRouter()

def generate_cache_key(request_data: Dict[str, Any]) -> str:
    """Generate cache key from request data"""
    data_str = json.dumps(request_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

async def forward_request(
    endpoint: str,
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    stream: bool = False
):
    """Forward request to model server"""
    if stream:
        async with httpx.AsyncClient(timeout=300.0) as client:
            url = f"{endpoint}{path}"
            async with client.stream("POST", url, json=data) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            url = f"{endpoint}{path}"

            if method == "GET":
                response = await client.get(url)
            elif method == "POST":
                response = await client.post(url, json=data)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")

            response.raise_for_status()
            return response.json()

@app.get("/")
async def root():
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
    """Check health of all model servers"""
    statuses = {}
    for model_name, endpoint in MODEL_ENDPOINTS.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/health")
                statuses[model_name] = response.status_code == 200
        except:
            statuses[model_name] = False

    all_healthy = all(statuses.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": statuses,
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
    data = await request.json()
    model_name = data.get("model")

    # Check cache if enabled
    if redis_client and not data.get("stream", False):
        cache_key = f"completion:{generate_cache_key(data)}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            cache_hits.inc()
            return json.loads(cached_response)
        else:
            cache_misses.inc()

    # Get endpoint
    endpoint = router.get_model_endpoint(model_name)
    model = model_name or router.get_next_model()

    # Track metrics
    request_counter.labels(model=model).inc()
    start_time = time.time()

    try:
        # Stream handling
        if data.get("stream", False):
            return StreamingResponse(
                forward_request(endpoint, "POST", "/v1/completions", data, stream=True),
                media_type="text/event-stream"
            )

        # Regular request
        response = await forward_request(endpoint, "POST", "/v1/completions", data)

        # Cache response if enabled
        if redis_client:
            cache_key = f"completion:{generate_cache_key(data)}"
            redis_client.setex(
                cache_key,
                config['caching']['ttl'],
                json.dumps(response)
            )

        # Track duration
        request_duration.labels(model=model).observe(time.time() - start_time)

        return response

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests"""
    data = await request.json()
    model_name = data.get("model")

    # Check cache if enabled
    if redis_client and not data.get("stream", False):
        cache_key = f"chat:{generate_cache_key(data)}"
        cached_response = redis_client.get(cache_key)
        if cached_response:
            cache_hits.inc()
            return json.loads(cached_response)
        else:
            cache_misses.inc()

    # Get endpoint
    endpoint = router.get_model_endpoint(model_name)
    model = model_name or router.get_next_model()

    # Track metrics
    request_counter.labels(model=model).inc()
    start_time = time.time()

    try:
        # Stream handling
        if data.get("stream", False):
            return StreamingResponse(
                forward_request(endpoint, "POST", "/v1/chat/completions", data, stream=True),
                media_type="text/event-stream"
            )

        # Regular request
        response = await forward_request(endpoint, "POST", "/v1/chat/completions", data)

        # Cache response if enabled
        if redis_client:
            cache_key = f"chat:{generate_cache_key(data)}"
            redis_client.setex(
                cache_key,
                config['caching']['ttl'],
                json.dumps(response)
            )

        # Track duration
        request_duration.labels(model=model).observe(time.time() - start_time)

        return response

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models/{model_name}/completions")
async def model_specific_completions(model_name: str, request: Request):
    """Handle model-specific completion requests"""
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    data = await request.json()
    data["model"] = model_name
    return await completions(request)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
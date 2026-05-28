import os 
import asyncio
from typing import Any 

import httpx 
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(override=False)

MODEL_BACKENDS = {'RuadaptQwen3-4B-simpo':"http://127.0.0.1:8001",
                  'RuadaptQwen3-4B-sft-pro': "http://127.0.0.1:8002",
                  'RuadaptQwen3-4B-sft-stance': "http://127.0.0.1:8003",
                  'RuadaptQwen3-32B-Instruct_v2_stance': "http://127.0.0.1:8004",
                  'RuadaptQwen3-32B-Instruct_v2_stance_cons_tox_classifier': "http://127.0.0.1:8005",
                  'RuadaptQwen3-32B-Instruct_v2_tox_classifier': "http://127.0.0.1:8006",
                  'RuadaptQwen3-32B-Instruct_v2_tox_stance_classifier': "http://127.0.0.1:8007",
                  'RuadaptQwen3-32B-Instruct_v2_tox_target_classifier': "http://127.0.0.1:8008"}

ROUTER_API_KEY = os.getenv("ADDITIONAL_VLLM_API_KEY", "")
BACKEND_API_KEY = os.getenv("ADDITIONAL_VLLM_API_KEY", "")
MODEL_LIST_TIMEOUT = float(os.getenv("MODEL_LIST_TIMEOUT", "5"))
REQUEST_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))

app = FastAPI()


def check_auth(authorization: str | None):
    if ROUTER_API_KEY and authorization != f"Bearer {ROUTER_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
    return headers


async def fetch_backend_models(backend_url: str) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=MODEL_LIST_TIMEOUT) as client:
            response = await client.get(f"{backend_url.rstrip('/')}/v1/models", headers=build_headers())
        if response.status_code != 200:
            return []
        return response.json().get("data", [])
    except httpx.HTTPError:
        return []


def build_model_info(model_name: str, backend_models: list[dict[str, Any]]) -> dict[str, Any]:
    backend_model = next((item for item in backend_models if item.get("id") == model_name), None)
    if backend_model is None and len(backend_models) == 1:
        backend_model = backend_models[0]

    status = "offloaded"
    if backend_model is not None:
        status = backend_model.get("status") or "spawned_additional"

    return {
        "id": model_name,
        "created": backend_model.get("created") if backend_model else None,
        "object": backend_model.get("object") if backend_model else None,
        "owned_by": backend_model.get("owned_by") if backend_model else None,
        "max_model_len": backend_model.get("max_model_len") if backend_model else None,
        "status": status,
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    check_auth(authorization)

    data = []
    model_names = list(MODEL_BACKENDS)
    backend_model_lists = await asyncio.gather(
        *(fetch_backend_models(MODEL_BACKENDS[model_name]) for model_name in model_names)
    )

    for model_name, backend_models in zip(model_names, backend_model_lists):
        data.append(build_model_info(model_name, backend_models))
        
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    check_auth(authorization)

    body = await request.json()
    model = body.get("model")
    if model not in MODEL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}. Available: {list(MODEL_BACKENDS)}")
    
    backend_url = MODEL_BACKENDS[model]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{backend_url.rstrip('/')}/v1/chat/completions",
            json=body,
            headers=build_headers(),
        )

    return JSONResponse(content=response.json(), status_code=response.status_code)

import os 
import asyncio
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from string import Formatter
from typing import Any 

import httpx 
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(override=False)

ROUTER_API_KEY = os.getenv("ADDITIONAL_VLLM_API_KEY", "")
BACKEND_API_KEY = os.getenv("ADDITIONAL_VLLM_API_KEY", "")
MODEL_LIST_TIMEOUT = float(os.getenv("MODEL_LIST_TIMEOUT", "5"))
REQUEST_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
ON_DEMAND_ENABLED = os.getenv("ON_DEMAND_VLLM_ENABLED", "1").lower() not in ("0", "false", "no")
ON_DEMAND_PORT_START = int(os.getenv("ON_DEMAND_PORT_START", "8001"))
ON_DEMAND_HOST = os.getenv("ON_DEMAND_HOST", "127.0.0.1")
ON_DEMAND_CUDA_DEVICES = [
    item.strip()
    for item in os.getenv("ON_DEMAND_CUDA_DEVICES", "0,1").split(",")
    if item.strip()
]
ON_DEMAND_CUDA_DEVICE_GROUPS = [
    item.strip()
    for item in os.getenv("ON_DEMAND_CUDA_DEVICE_GROUPS", "").split(";")
    if item.strip()
] or ON_DEMAND_CUDA_DEVICES
ON_DEMAND_MAX_ACTIVE_MODELS = int(os.getenv("ON_DEMAND_MAX_ACTIVE_MODELS", str(max(1, len(ON_DEMAND_CUDA_DEVICE_GROUPS)))))
ON_DEMAND_START_WAIT_SECONDS = float(os.getenv("ON_DEMAND_START_WAIT_SECONDS", "300"))
ON_DEMAND_HEALTH_INTERVAL_SECONDS = float(os.getenv("ON_DEMAND_HEALTH_INTERVAL_SECONDS", "2"))
ON_DEMAND_STOP_TIMEOUT_SECONDS = float(os.getenv("ON_DEMAND_STOP_TIMEOUT_SECONDS", "30"))
ON_DEMAND_VLLM_COMMAND = os.getenv(
    "ON_DEMAND_VLLM_COMMAND",
    "vllm serve {model_path} --served-model-name {model} --host {host} --port {port}",
)

app = FastAPI()


@dataclass
class ModelState:
    process: subprocess.Popen | None = None
    status: str = "offloaded"
    last_used: float = field(default_factory=time.monotonic)
    inflight: int = 0
    error: str | None = None
    cuda_visible_devices: str = ""


MODEL_STATES: dict[str, ModelState] = {}
MODEL_LOCKS: dict[str, asyncio.Lock] = {}
SCHEDULER_LOCK = asyncio.Lock()


def env_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value).upper()


def parse_csv_env(*names: str) -> list[str]:
    result: list[str] = []
    seen = set()
    for name in names:
        for item in os.getenv(name, "").split(","):
            value = item.strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
    return result


def build_model_backends() -> dict[str, str]:
    model_names = parse_csv_env(
        "FINE_TUNED_GENERATION_MODELS",
        "FINE_TUNED_COMPARISON_MODELS",
        "FINE_TUNED_MODERATION_MODELS",
    )
    if not model_names:
        model_names = [
            "RuadaptQwen3-4B-simpo",
            "RuadaptQwen3-4B-sft-pro",
            "RuadaptQwen3-4B-sft-stance",
            "RuadaptQwen3-32B-Instruct_v2_stance",
            "RuadaptQwen3-32B-Instruct_v2_stance_cons_tox_classifier",
            "RuadaptQwen3-32B-Instruct_v2_tox_classifier",
            "RuadaptQwen3-32B-Instruct_v2_tox_stance_classifier",
            "RuadaptQwen3-32B-Instruct_v2_tox_target_classifier",
        ]

    return {
        model_name: f"http://{ON_DEMAND_HOST}:{ON_DEMAND_PORT_START + idx}"
        for idx, model_name in enumerate(model_names)
    }


MODEL_BACKENDS = build_model_backends()


def get_model_lock(model_name: str) -> asyncio.Lock:
    if model_name not in MODEL_LOCKS:
        MODEL_LOCKS[model_name] = asyncio.Lock()
    return MODEL_LOCKS[model_name]


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


async def is_backend_ready(model_name: str) -> bool:
    backend_models = await fetch_backend_models(MODEL_BACKENDS[model_name])
    return any(item.get("id") == model_name for item in backend_models) or len(backend_models) == 1


def build_model_info(model_name: str, backend_models: list[dict[str, Any]]) -> dict[str, Any]:
    backend_model = next((item for item in backend_models if item.get("id") == model_name), None)
    if backend_model is None and len(backend_models) == 1:
        backend_model = backend_models[0]

    status = "offloaded"
    if backend_model is not None:
        status = backend_model.get("status") or "spawned_additional"
    elif model_name in MODEL_STATES:
        status = MODEL_STATES[model_name].status

    return {
        "id": model_name,
        "created": backend_model.get("created") if backend_model else None,
        "object": backend_model.get("object") if backend_model else None,
        "owned_by": backend_model.get("owned_by") if backend_model else None,
        "max_model_len": backend_model.get("max_model_len") if backend_model else None,
        "status": status,
    }


def render_command(template: str, *, model_name: str, backend_url: str, cuda_visible_devices: str) -> list[str]:
    port = backend_url.rstrip("/").rsplit(":", 1)[-1]
    model_path = os.getenv(f"MODEL_PATH_{env_key(model_name)}") or os.getenv(f"MODEL_PATH_{model_name}") or model_name
    values = {
        "model": model_name,
        "served_model_name": model_name,
        "model_path": model_path,
        "host": ON_DEMAND_HOST,
        "port": port,
        "backend_url": backend_url,
        "cuda_visible_devices": cuda_visible_devices,
    }
    fields = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    return shlex.split(template.format(**{key: values[key] for key in fields}))


def active_managed_models() -> list[str]:
    active = []
    for model_name, state in MODEL_STATES.items():
        if state.process is not None and state.process.poll() is None and state.status in ("loading", "spawned_additional"):
            active.append(model_name)
    return active


async def stop_model(model_name: str):
    state = MODEL_STATES.get(model_name)
    if state is None or state.process is None or state.process.poll() is not None:
        return

    state.status = "offloading"
    try:
        os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.to_thread(state.process.wait, ON_DEMAND_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(state.process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await asyncio.to_thread(state.process.wait)
    finally:
        state.process = None
        state.status = "offloaded"
        state.error = None


async def reserve_model_slot(model_name: str) -> str:
    if not ON_DEMAND_CUDA_DEVICE_GROUPS:
        return ""

    if ON_DEMAND_MAX_ACTIVE_MODELS <= 0:
        used_devices = {
            state.cuda_visible_devices
            for state in MODEL_STATES.values()
            if state.process is not None and state.process.poll() is None and state.cuda_visible_devices
        }
        return next((device for device in ON_DEMAND_CUDA_DEVICE_GROUPS if device not in used_devices), ON_DEMAND_CUDA_DEVICE_GROUPS[0])

    async with SCHEDULER_LOCK:
        active = active_managed_models()
        if model_name in active or len(active) < ON_DEMAND_MAX_ACTIVE_MODELS:
            used_devices = {
                MODEL_STATES[name].cuda_visible_devices
                for name in active
                if MODEL_STATES[name].cuda_visible_devices
            }
            free_device = next((device for device in ON_DEMAND_CUDA_DEVICE_GROUPS if device not in used_devices), None)
            if free_device:
                return free_device

        candidates = [
            name
            for name in active
            if MODEL_STATES[name].inflight == 0 and MODEL_STATES[name].status != "loading"
        ]
        if not candidates:
            raise HTTPException(
                status_code=503,
                detail="All model slots are busy; retry after current requests finish.",
            )

        victim = min(candidates, key=lambda name: MODEL_STATES[name].last_used)
        cuda_visible_devices = MODEL_STATES[victim].cuda_visible_devices
        await stop_model(victim)
        return cuda_visible_devices or ON_DEMAND_CUDA_DEVICE_GROUPS[0]


def start_model_process(model_name: str, cuda_visible_devices: str) -> ModelState:
    state = MODEL_STATES.setdefault(model_name, ModelState())
    backend_url = MODEL_BACKENDS[model_name]

    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = render_command(
        ON_DEMAND_VLLM_COMMAND,
        model_name=model_name,
        backend_url=backend_url,
        cuda_visible_devices=cuda_visible_devices,
    )
    state.process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state.status = "loading"
    state.error = None
    state.cuda_visible_devices = cuda_visible_devices
    state.last_used = time.monotonic()
    return state


async def ensure_model_ready(model_name: str):
    if await is_backend_ready(model_name):
        state = MODEL_STATES.setdefault(model_name, ModelState())
        state.status = "spawned_additional"
        state.last_used = time.monotonic()
        return

    if not ON_DEMAND_ENABLED:
        raise HTTPException(status_code=503, detail=f"Model {model_name} is offloaded and on-demand loading is disabled.")

    lock = get_model_lock(model_name)
    async with lock:
        if await is_backend_ready(model_name):
            state = MODEL_STATES.setdefault(model_name, ModelState())
            state.status = "spawned_additional"
            state.last_used = time.monotonic()
            return

        state = MODEL_STATES.setdefault(model_name, ModelState())
        if state.process is None or state.process.poll() is not None:
            cuda_visible_devices = await reserve_model_slot(model_name)
            try:
                state = start_model_process(model_name, cuda_visible_devices)
            except Exception as e:
                state.status = "failed"
                state.error = repr(e)
                raise HTTPException(status_code=500, detail=f"Failed to start model {model_name}: {e}")

        deadline = time.monotonic() + ON_DEMAND_START_WAIT_SECONDS
        while time.monotonic() < deadline:
            if state.process is not None and state.process.poll() is not None:
                state.status = "failed"
                state.error = f"Process exited with code {state.process.returncode}"
                raise HTTPException(status_code=502, detail=f"Model {model_name} failed to start: {state.error}")
            if await is_backend_ready(model_name):
                state.status = "spawned_additional"
                state.last_used = time.monotonic()
                return
            await asyncio.sleep(ON_DEMAND_HEALTH_INTERVAL_SECONDS)

        raise HTTPException(
            status_code=503,
            detail=f"Model {model_name} is loading. Retry later.",
        )


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
    
    await ensure_model_ready(model)
    backend_url = MODEL_BACKENDS[model]
    state = MODEL_STATES.setdefault(model, ModelState(status="spawned_additional"))
    state.inflight += 1
    state.last_used = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{backend_url.rstrip('/')}/v1/chat/completions",
                json=body,
                headers=build_headers(),
            )
    finally:
        state.inflight = max(0, state.inflight - 1)
        state.last_used = time.monotonic()

    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.post("/v1/models/{model_name:path}/load")
async def load_model(model_name: str, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if model_name not in MODEL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}. Available: {list(MODEL_BACKENDS)}")
    await ensure_model_ready(model_name)
    return {"model": model_name, "status": MODEL_STATES.get(model_name, ModelState()).status}


@app.post("/v1/models/{model_name:path}/offload")
async def offload_model(model_name: str, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if model_name not in MODEL_BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}. Available: {list(MODEL_BACKENDS)}")
    await stop_model(model_name)
    return {"model": model_name, "status": MODEL_STATES.get(model_name, ModelState()).status}

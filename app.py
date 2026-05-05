import os
import json
import re
import ast
from typing import Dict, List, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prompts import CRITERION_V3

from dotenv import load_dotenv
load_dotenv(override=False)

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:6266")
API_KEY = os.getenv("VLLM_API_KEY", "")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))

app = FastAPI(title="LLM Ispras")


# Contracts
class RunRequest(BaseModel):
    model: str = Field(...)
    text: str = Field(..., min_length=1)
    criteria: Optional[List[str]] = Field(default=None)
    max_retries: int = Field(default=1, ge=0, le=2)

class CriterionResult(BaseModel):
    task_name: str
    verdict: str  # "pass" | "fail" or "ANTI"|"PRO"|"NEU" depends on the criteria 
    reason: str
    raw: str
    raw_repr: str

class RunResponse(BaseModel):
    model: str
    results: Dict[str, CriterionResult]


CRITERION = CRITERION_V3


def build_response_format(key: str) -> Dict[str, Any]:
    if key == "anti_russia":
        verdict_enum = ["ANTI", "PRO", "NEU"]
    else:
        verdict_enum = ["pass", "fail"]

    schema = {
        "name": f"{key}_result",
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": verdict_enum},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    return {"type": "json_schema", "json_schema": schema}


@app.get("/criteria")
async def get_criteria():
    items = [{"key": key, "title": cfg["title"]} for key, cfg in CRITERION.items()]
    return {"criteria": items}


@app.get("/models")
async def get_models():
    url = f"{BASE_URL}/v1/models"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)
    
    data = response.json()
    models = [{"id": m.get("id"), "status": m.get("status")} for m in data.get("data", []) if m.get("id") is not None]
    return {"models": models}

async def chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2,
               response_format: Optional[Dict[str, Any]] = None) -> str:
    
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req_body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if response_format is not None:
        req_body["response_format"] = response_format

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        result = await client.post(url, headers=headers, json=req_body)

    if result.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM chat error: {result.text}")
    
    data = result.json()
    try:
        return data['choices'][0]["message"]["content"]
    except:
        raise HTTPException(status_code=502, detail=f"Unexpected vLLM response: {data}")
    

def parse_model_reply(raw: str) -> Dict[str, Any]:
    if raw is None:
        raise ValueError("Empty model reply")

    def normalize_quotes(s: str) -> str:
        return s.translate(
            str.maketrans(
                {
                    "“": '"', "”": '"',
                    "‟": '"', "„": '"',
                    "’": "'", "‘": "'", "‚": "'",
                }
            )
        )

    def strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```\s*$", "", s)
        return s.strip()

    def extract_json_object(s: str) -> str | None:
        lb = s.find("{")
        rb = s.rfind("}")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb:rb + 1].strip()
        return None

    def newlines_in_strings(s: str) -> str:
        res = []
        in_str = False
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    res.append(ch)
                    esc = False
                    continue
                if ch == "\\":
                    res.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    res.append(ch)
                    in_str = False
                    continue
                if ch == "\n":
                    res.append("\\n")
                    continue
                if ch == "\r":
                    res.append("\\r")
                    continue
                res.append(ch)
            else:
                res.append(ch)
                if ch == '"':
                    in_str = True
                    esc = False
        return "".join(res)

    s = str(raw)
    s = strip_code_fences(s)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    s_norm = normalize_quotes(s)
    try:
        obj = json.loads(s_norm)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    for text in (s, normalize_quotes(s)):
        candidate = extract_json_object(text)
        if candidate:
            cand = newlines_in_strings(candidate)
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise ValueError("Unable to parse model reply as JSON")

def build_messages(cfg: Dict[str, Any], user_text: str) -> List[Dict[str, str]]:
    system = str(cfg.get('system_prompt', '')).strip()
    if not system:
        raise ValueError("Criterion config is missing non-empty 'system_prompt'")

    title = str(cfg.get('title', '')).strip()
    instruction = str(cfg.get('instruction', '')).strip()
    few_shot = cfg.get('few_shot', {})


    examples_block = ""
    if few_shot and isinstance(few_shot, dict):
        verdict_options = few_shot.keys()
        lines = []
        lines.append("**Примеры:**\n")
        for opt in verdict_options:
            for example in few_shot.get(opt, []):
                example_text = example['text'].strip()
                example_reason = example['reason'].strip()
                lines.append(f"Текст: {example_text}")
                lines.append(f"Ответ: {{\"verdict\": \"{opt}\", \"reason\": \"{example_reason}\"}}")
        
        examples_block = '\n'.join(lines) + "\n\n"
 
                             
    user = (
        f"**Критерий:** {title}\n"
        f"**Инструкция:** {instruction}\n\n"
        + examples_block + 
        f"**Текст для анализа:**\n\n{user_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def normalize_verdict(key: str, raw_verdict: str) -> str:
    verdict = raw_verdict.strip()
    if not verdict:
        return "UNDEFINED"

    if key == 'anti_russia':
        verdict = verdict.upper()

        if verdict=='ANTI':
            return "fail"
        if verdict in ("PRO", "NEU"):
            return "pass"
        return "UNDEFINED"
    
    verdict = verdict.lower()
    if verdict in ("pass", "fail"):
        return verdict 

    return "UNDEFINED"


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest):
    available_models = (await get_models())["models"]
    available_ids = [model['id'] for model in available_models]
    if request.model not in available_ids:
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not available")

    results: Dict[str, CriterionResult] = {}

    selected = request.criteria
    if not selected:
        keys_to_run = list(CRITERION.keys())
    else:
        keys_to_run = [key for key in CRITERION.keys() if key in set(selected)]

    for key in keys_to_run:
        cfg = CRITERION[key]
        last_raw = ""
        last_err: Optional[Exception] = None

        for attempt in range(request.max_retries + 1):
            raw = await chat(
                model=request.model,
                messages=build_messages(cfg, request.text),
                temperature=0.2 if attempt == 0 else 0.0,
                response_format=build_response_format(key),
            )
            #print(build_messages(cfg["title"], cfg["instruction"], request.text, cfg["few_shot"]))
            last_raw = raw

            try:
                parsed = parse_model_reply(raw)
                raw_verdict = str(parsed.get("verdict", "pass")).strip()
                verdict = normalize_verdict(key, raw_verdict)
                reason = str(parsed.get("reason", "")).strip()

                results[key] = CriterionResult(
                    task_name=cfg["title"],
                    verdict=verdict,
                    reason=reason,
                    raw=last_raw,
                    raw_repr=repr(last_raw),
                )
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None and key not in results:
            results[key] = CriterionResult(
                task_name=cfg["title"],
                verdict="UNDEFINED",
                reason=f"Couldn't parse model's answer as a JSON after {request.max_retries + 1} retries.",
                raw=last_raw,
                raw_repr=repr(last_raw),
            )

    return RunResponse(model=request.model, results=results)
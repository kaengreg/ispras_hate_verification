import os
import json
import re
import ast
from typing import Dict, List, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:6266")
API_KEY = os.getenv("VLLM_API_KEY", "")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

app = FastAPI(title="LLM Ispras")


# Contracts
class RunRequest(BaseModel):
    model: str = Field(...)
    text: str = Field(..., min_length=1)
    criteria: Optional[List[str]] = Field(...)

class CriterionResult(BaseModel):
    task_name: str
    verdict: str  # "pass" | "fail"
    reason: str
    raw: str

class RunResponse(BaseModel):
    model: str
    results: Dict[str, CriterionResult]


"""CRITERION_v1 = {
    "anti_russia": {
        "title": "Антироссийские высказывания",
        "instruction": (
            "Определи, содержит ли текст антироссийские высказывания: унижение России/россиян, "
            "призывы к дискриминации/вражде, дегуманизация. Обычная критика политики/событий без ненависти "
            "или призывов сама по себе не считается нарушением."
        ),
    },
    "obscene language": {
        "title": "Контроль мата",
        "instruction": "Определи, есть ли в тексте нецензурная лексика/мат, включая завуалированный мат (звёздочки, пропуски букв).",
    },
    "hate_speech": {
        "title": "Ненависть к народам, религиям, странам и социальным слоям",
        "instruction": (
            "Определи, есть ли в тексте высказывания ненависти/вражды к группам людей по национальным или "
            "религиозным признакам, по признаку страны происхождения или социального слоя, включая дегуманизацию и призывы к насилию/дискриминации."
        ),
    },
    "toxicity": {
        "title": "Токсичность по отношению к конкретному человеку",
        "instruction": (
            "Определи, есть ли в тексте персональные оскорбления, унижения, травля или токсичность, "
            "направленные на конкретного человека (по имени/явному адресату)."
        ),
    },
    "crime_approval": {
        "title": "Одобрение правонарушений и преступлений",
        "instruction": "Определи, содержит ли текст одобрение/поощрение преступлений или призывы совершать правонарушения.",
    },
    "drugs_instructions": {
        "title": "Рекомендации по запрещенным веществам",
        "instruction": (
            "Определи, содержит ли текст инструкции, рецепты, советы по изготовлению, употреблению, покупке, "
            "распространению запрещённых веществ либо советы по обходу проверок/законов."
        ),
    },
    "weapons_explosives": {
        "title": "Рекомендации по оружию, взрывчатым и опасным веществам",
        "instruction": (
            "Определи, содержит ли текст инструкции, рецепты или советы по изготовлению/модификации/применению оружия, "
            "взрывчатых или иных опасных веществ/устройств."
        ),
    },
    "cultural_historical": {
        "title": "Контроль культурных и исторических норм",
        "instruction": (
            "Определи, содержит ли текст явные оскорбления культурных/исторических символов и традиций, "
            "надругательство над памятными датами, либо провокационное унижение исторической памяти."
        ),
    },
}"""

CRITERION_V2 = {
    "anti_russia": {
        "title": "Антироссийские высказывания",
        "instruction": (
            "Определи, содержит ли текст антироссийские высказывания."

            "Считается нарушением:"
            "— унижение России или россиян как группы;"
            "— обесчеловечивание россиян;"
            "— призывы к дискриминации, вражде или насилию в отношении россиян;"
            " — обобщающие негативные утверждения о россиянах как о народе."

            "Не считается нарушением:"
            "— критика политики, власти, государственных решений без ненависти;"
            "— обсуждение санкций и международных отношений без ненависти;"

            "Оценивай только явное содержание текста, без домыслов о намерениях автора."
            "Если нарушение не выражено явно — выбирай отсутствие нарушения."
        ),
    },
    "obscene language": {
        "title": "Контроль мата",
        "instruction": ("Определи, содержит ли текст нецензурную лексику (мат)."

                        "Считается нарушением:"
                        "— прямое использование нецензурной лексики;"
                        "— завуалированный мат (звёздочки, пропуски букв, фонетические замены);"
                        "— устойчивые матерные выражения и ругательства."

                        "Не считается нарушением:"
                        "— разговорные грубости без мата;"
                        "— упоминание мата в описательном, учебном или аналитическом контексте."

                        "Учитывай язык текста и распространённые способы маскировки мата."),
    },
    "hate_speech": {
        "title": "Ненависть к народам, религиям, странам и социальным слоям",
        "instruction": (
            "Определи, содержит ли текст высказывания ненависти или вражды к группам людей."

            "Считается нарушением:"
            "— негативные или враждебные обобщения по признаку национальности, религии,"
            "страны происхождения или социального слоя;"
            "— обесчеловечивание групп людей;"
            "— призывы к насилию, дискриминации или исключению."

            "Если объектом является группа людей, используй этот критерий, а не персональную токсичность."
        ),
    },
    "toxicity": {
        "title": "Токсичность по отношению к конкретному человеку",
        "instruction": (
            "Определи, содержит ли текст токсичность, направленную на конкретного человека."

            "Считается нарушением:"
            "— персональные оскорбления, унижения или насмешки;"
            "— агрессивные или враждебные высказывания в адрес конкретного лица;"
            "— травля или систематическое унижение личности."

            "Не считается нарушением:"
            "— резкая, но аргументированная критика действий, решений или взглядов;"
            "— эмоциональные высказывания без конкретного адресата."
        ),
    },
    "crime_approval": {
        "title": "Одобрение правонарушений и преступлений",
        "instruction": ("Определи, содержит ли текст одобрение или поощрение правонарушений."

                        "Считается нарушением:"
                        "— прямое одобрение преступлений или насилия;"
                        "— призывы к совершению незаконных действий;"
                        "— оправдание краж, мошенничества, насилия и других правонарушений."

                        "Не считается нарушением:"
                        "— описание преступлений в новостном, художественном контексте;"
                        "— обсуждение последствий, наказаний или профилактики преступлений."),
    },
    "drugs_instructions": {
        "title": "Рекомендации по запрещенным веществам",
        "instruction": (
            "Определи, содержит ли текст инструкции или советы, связанные с запрещенными веществами."

            "Считается нарушением:"
            "— инструкции по изготовлению, употреблению, покупке или распространению;"
            "— советы по сокрытию, обходу проверок или нарушению законов."

            "Не считается нарушением:"
            "— медицинское, научное или общественно-профилактическое описание;"
            "— обсуждение вреда, зависимости и последствий употребления без практических рекомендаций."
        ),
    },
    "weapons_explosives": {
        "title": "Рекомендации по оружию, взрывчатым и опасным веществам",
        "instruction": (
            "Определи, содержит ли текст инструкции или советы по оружию или взрывчатым веществам."

            "Считается нарушением:"
            "— инструкции по изготовлению, модификации или применению оружия;"
            "— советы по созданию или использованию взрывчатых и других опасных устройств."

            "Не считается нарушением:"
            "— историческое, новостное или учебное описание;"
            "— обсуждение законодательства, контроля и мер безопасности без инструктивных деталей."
        ),
    },
    "cultural_historical": {
        "title": "Контроль культурных и исторических норм",
        "instruction": (
            "Определи, содержит ли текст оскорбления культурных или исторических ценностей."

            "Считается нарушением:"
            "— явные оскорбления или надругательство над культурными, религиозными"
            "или историческими символами;"
            "— провокационное унижение исторической памяти или памятных дат."

            "Не считается нарушением:"
            "— критическое, научное или публицистическое обсуждение истории и культуры;"
            "— выражение личного мнения без оскорбительного или уничижительного языка."
        ),
    },
}

CRITERION = CRITERION_V2


@app.get("/criteria")
async def get_criteria():
    items = [{"key": key, "title": cfg["title"]} for key, cfg in CRITERION.items()]
    return {"criteria": items}


@app.get("/models")
async def get_models():
    url = f"{BASE_URL}/v1/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)
    
    data = response.json()
    models =  [{"id": model["id"], "status": model["status"]} for model in data.get("data", []) if "id" in model]
    return {"models": models}

async def chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    req_body = {"model": model, "messages": messages, "temperature": temperature}

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

    s = str(raw).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    s = s.translate(str.maketrans({
        "“": '"', "”": '"',
        "«": '"', "»": '"',
        "’": "'", "‘": "'",
    }))

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    lb = s.find("{")
    rb = s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        candidate = s[lb:rb + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise ValueError("Unable to parse model reply as JSON")

def build_messages(title: str, instruction: str, user_text: str) -> List[Dict[str, str]]:
    system = (
        "Ты виртуальный ассистент для модерации текстов. Твоя задача — проверять текст по заданному критерию и выдавать однозначный вердикт.\n"
        "Отвечай строго в JSON без markdown и без лишних ключей.\n"
        "Ключи JSON: verdict, reason. Никаких других ключей.\n"
        "verdict может быть только: \"pass\" или \"fail\" (строго в нижнем регистре).\n"
        "Правило вынесения вердиктов: если по данному критерию в тексте ЕСТЬ нарушение — verdict=\"fail\". Если нарушения НЕТ — verdict=\"pass\".\n"
        "Если есть сомнение или недостаточно данных — выбирай verdict=\"pass\".\n"
        "Не цитируй мат/оскорбления/ненавистнические фразы дословно; при необходимости используй маскировку (например, ***)."
    )
    user = (
        f"Критерий: {title}\n"
        f"Инструкция: {instruction}\n\n"
        "Твоя задача: определить, есть ли в тексте нарушение по этому критерию.\n"
        "Верни строго JSON формата: {\"verdict\":\"pass\"|\"fail\",\"reason\":\"коротко почему (1-3 предложения)\"}.\n"
        "Важно: если нарушение есть — обязательно \"fail\". Если нарушения нет — обязательно \"pass\".\n\n"
        f"Текст:\n{user_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

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
        raw = await chat(
            model=request.model,
            messages=build_messages(cfg["title"], cfg["instruction"], request.text),
            temperature=0.2,
        )

        try:
            parsed = parse_model_reply(raw)
            verdict = str(parsed.get("verdict", "pass")).strip().lower()
            verdict = verdict if verdict in {"pass", "fail"} else "pass"
            reason = str(parsed.get("reason", "")).strip()
            results[key] = CriterionResult(task_name=cfg["title"], verdict=verdict, reason=reason, raw=raw)
        except Exception:
            results[key] = CriterionResult(
                task_name=cfg["title"],
                verdict="fail",
                reason="Не удалось разобрать ответ модели как JSON.",
                raw=raw,
            )

    return RunResponse(model=request.model, results=results)
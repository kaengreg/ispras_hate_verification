import gradio as gr
import httpx
from gradio import Request
import os
import html
import asyncio
import time
import base64
from pathlib import Path

from app import get_criteria as backend_get_criteria
from app import get_models as backend_get_models

DEFAULT_SYSTEM_PROMPT = ("Ты — полезный ассистент. Отвечай точно, связно и на языке пользователя.")

BASE_COMPARISON_MODELS = [
    item.strip()
    for item in (
        os.getenv("BASE_COMPARISON_MODELS")
        or os.getenv("BASE_COMPARISON_MODEL", "RuadaptQwen3-4B-Instruct")
    ).split(",")
    if item.strip()
]
FINE_TUNED_COMPARISON_MODELS = [
    item.strip()
    for item in os.getenv(
        "FINE_TUNED_COMPARISON_MODELS",
        "RuadaptQwen3-4B-Instruct_sft_pro,"
        "RuadaptQwen3-4B-Instruct_sft_stance,"
        "RuadaptQwen3-4B_simpo_stance_v4",
    ).split(",")
    if item.strip()
]
MODEL_CACHE_TTL_SECONDS = float(os.getenv("MODEL_CACHE_TTL_SECONDS", "300"))
MODEL_ITEMS_CACHE = {"items": None, "loaded_at": 0.0}

def get_logo_markup():
    logo_path = Path(__file__).resolve().parent / "logo.png"
    if not logo_path.exists():
        return '<div class="brand-logo-fallback">IS</div>'

    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f'<img class="brand-logo" src="data:image/png;base64,{encoded}" alt="ISPRAS logo">'

def get_api_base(req: Request) -> str:
    base = os.getenv("API_BASE_URL") or str(req.request.base_url).rstrip("/")
    return base.replace("http://0.0.0.0", "http://127.0.0.1")


def run_backend_async(coro):
    return asyncio.run(coro)


def clear_model_cache():
    MODEL_ITEMS_CACHE["items"] = None
    MODEL_ITEMS_CACHE["loaded_at"] = 0.0


def fetch_model_items(req: Request, force: bool = False):
    now = time.monotonic()
    cached_items = MODEL_ITEMS_CACHE["items"]
    cache_age = now - MODEL_ITEMS_CACHE["loaded_at"]
    if not force and cached_items is not None and cache_age < MODEL_CACHE_TTL_SECONDS:
        return cached_items

    base = get_api_base(req)
    try:
        response = httpx.get(f"{base}/models", timeout=20)
        response.raise_for_status()
        items = response.json().get("models", [])
    except httpx.HTTPError:
        items = run_backend_async(backend_get_models()).get("models", [])

    MODEL_ITEMS_CACHE["items"] = items
    MODEL_ITEMS_CACHE["loaded_at"] = now
    return items


def fetch_models(req: Request, force: bool = False):
    return build_model_choices(fetch_model_items(req, force=force))


def fetch_criteria(req: Request):
    base = get_api_base(req)
    try:
        response = httpx.get(f"{base}/criteria", timeout=20)
        response.raise_for_status()
        items = response.json().get("criteria", [])
    except httpx.HTTPError:
        items = run_backend_async(backend_get_criteria()).get("criteria", [])
    return [(item["title"], item["key"]) for item in items if item.get("key")]

def build_model_choices(models):
    choices = []
    models = sorted(models, key=lambda model: model.get("status", "") not in ("spawned","spawned_additional"))
    for model in models:
        status = model.get("status")
        if status == "spawned":
            icon = "🟢"
        elif status == "spawned_additional":
            icon = "🔵"
        else:
            icon = "❌"
        source = model.get("source", "unknown")
        model_id = model.get("model", model.get("id"))
        label = f"{icon} {model_id}"
        if model.get("duplicate"):
            label = f"{label} [{source}]"
        choices.append((label, model["id"]))
    return choices


def find_model(models, model_name, preferred_source=None):
    matches = [model for model in models if model.get("model") == model_name or model.get("id") == model_name]
    if not matches:
        return None

    preferred = None
    if preferred_source:
        preferred = next((model for model in matches if model.get("source") == preferred_source), None)
    selected = preferred or matches[0]
    return selected["id"]


def find_first_available_model(models, model_names, preferred_source=None):
    for model_name in model_names:
        model_id = find_model(models, model_name, preferred_source=preferred_source)
        if model_id is not None:
            return model_id, model_name
    return None, None


def build_comparison_model_choices(models):
    choices = []
    allowed = set(FINE_TUNED_COMPARISON_MODELS)
    for label, value in build_model_choices(models):
        model = next((item for item in models if item.get("id") == value), None)
        model_name = model.get("model") if model else value
        if model_name in allowed:
            choices.append((label, value))
    return choices


def run_pipeline(model, criteria_keys, text, req: Request):
    if text is None or not str(text).strip():
        yield gr.update(value="[Ошибка] Необходимо ввести текст перед запуском", visible=True)
        return

    yield gr.update(value="&nbsp;\n\n⏳ **Обработка...**\n\nПожалуйста, подождите — выполняется проверка по выбранным критериям.", visible=True)

    base = get_api_base(req)
    req_body = {"model": model, "text": text, "criteria": criteria_keys}

    try:
        resp = httpx.post(f"{base}/run", json=req_body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            j = e.response.json()
            detail = str(j.get("detail", j))
        except Exception:
            detail = e.response.text
        yield gr.update(
            value=f"Ошибка запроса к /run (HTTP {e.response.status_code}):\n{detail}",
            visible=True,
        )
        return
    except Exception as e:
        yield gr.update(value=f"Неожиданная ошибка: {e}", visible=True)
        return

    results = data.get("results", {})

    table_lines = [
        "| Критерий | Вердикт |",
        "|---|---|",
    ]

    detail_blocks = []
    for key, ans in results.items():
        task_name = ans.get("task_name", key)
        verdict_raw = str(ans.get("verdict", "")).lower().strip()
        verdict_icon = "✅" if verdict_raw == "pass" else "❌"
        verdict_text = (
            "Нарушений по данному критерию не выявлено"
            if verdict_raw == "pass"
            else "Обнаружены нарушения по данному критерию"
        )
        reason = str(ans.get("reason", ""))
        raw = str(ans.get("raw", ""))
        raw_repr = str(ans.get("raw_repr", ""))

        task_name_tbl = task_name.replace("|", "\\|")
        table_lines.append(f"| {task_name_tbl} | {verdict_icon} |")

        reason_md = reason.replace("|", "\\|")
        detail_blocks.append(
            f"### {task_name}\n"
            f"**Вердикт:** {verdict_text} {verdict_icon}\n\n"
            f"**Обоснование:** {reason_md}\n\n"
            f"<details>\n<summary><b>Полный ответ модели</b></summary>\n\n"
            f"<div><b>raw</b></div>\n<pre>{raw}</pre>\n\n"
            f"<div><b>repr(raw)</b></div>\n<pre>{raw_repr}</pre>\n"
            f"</details>\n"
        )

    md = "\n".join(table_lines) + "\n\n---\n\n" + "\n\n".join(detail_blocks)

    yield gr.update(value=md, visible=True)

def run_generation(model, system_prompt, user_prompt, temperature, req: Request):
    if user_prompt is None or not str(user_prompt).strip():
        yield gr.update(value="[Ошибка] Необходимо ввести запрос перед запуском", visible=True)
        return 
    
    yield gr.update(value="⏳ Генерация ответа...", visible=True)

    base = get_api_base(req)
    req_body = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
    }

    try:
        response = httpx.post(f"{base}/generate", json=req_body, timeout=120)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        try:
            detail = str(e.response.json().get("detail", e.response.text))
        except Exception:
            detail = e.response.text
        yield gr.update(
            value=f"Ошибка запроса к /generate (HTTP {e.response.status_code}):\n{detail}",
            visible=True,
        )
        return
    except Exception as e:
        yield gr.update(value=f"Неожиданная ошибка: {e}", visible=True)
        return

    model_id = html.escape(str(data.get("model", model)))
    answer = str(data.get("text", ""))
    yield gr.update(
        value=f"{answer}",
        visible=True,
    )

def run_comparison(model, system_prompt, user_prompt, temperature, req: Request):
    if user_prompt is None or not str(user_prompt).strip():
        yield gr.update(value="[Ошибка] Необходимо ввести запрос перед запуском ")
        return 
    
    if not model:
        yield gr.update(value="[Ошибка] Необходимо выбрать дообученную модель")
        return

    models = fetch_model_items(req)
    base_model, _ = find_first_available_model(models, BASE_COMPARISON_MODELS, preferred_source="lab")
    if base_model is None:
        expected = ", ".join(BASE_COMPARISON_MODELS)
        yield gr.update(value=f"[Ошибка] Исходная модель не найдена в /models. Ожидались: {expected}", visible=True)
        return 
    
    yield gr.update(value="⏳ Генерация ответов для сравнения...", visible=True)

    base = get_api_base(req)

    def request_generation(model_id):
        req_body = {
            "model": model_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature
        }
        response = httpx.post(f"{base}/generate", json=req_body, timeout=120)
        response.raise_for_status()
        return response.json()

    try:
        tuned_data = request_generation(model)
        base_data = request_generation(base_model) 
    except httpx.HTTPStatusError as e:
        try:
            detail = str(e.response.json().get("detail", e.response.text))
        except Exception:
            detail = e.response.text
        yield gr.update(
            value=f"Ошибка запроса к /generate (HTTP {e.response.status_code}):\n{detail}",
            visible=True,
        )
        return
    except Exception as e:
        yield gr.update(value=f"Неожиданная ошибка: {e}", visible=True)
        return

    tuned_model = html.escape(str(tuned_data.get("model", model)))
    base_model_label = html.escape(str(base_data.get("model", base_model)))
    tuned_answer = str(tuned_data.get("text", ""))
    base_answer = str(base_data.get("text", ""))

    md = (
        "<table style=\"width:100%; table-layout:fixed; border-collapse:collapse;\">"
        "<thead>"
        "<tr>"
        "<th style=\"width:50%; text-align:left; padding:8px; border-bottom:1px solid #ddd;\">Дообученная модель</th>"
        "<th style=\"width:50%; text-align:left; padding:8px; border-bottom:1px solid #ddd;\">Исходная модель</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        "<tr>"
        "<td style=\"vertical-align:top; padding:8px; border-right:1px solid #ddd;\">"
        f"<div><b>Модель:</b> <code>{tuned_model}</code></div>"
        f"<div style=\"white-space:pre-wrap; overflow-wrap:anywhere; margin-top:8px;\">{html.escape(tuned_answer)}</div>"
        "</td>"
        "<td style=\"vertical-align:top; padding:8px;\">"
        f"<div><b>Модель:</b> <code>{base_model_label}</code></div>"
        f"<div style=\"white-space:pre-wrap; overflow-wrap:anywhere; margin-top:8px;\">{html.escape(base_answer)}</div>"
        "</td>"
        "</tr>"
        "</tbody>"
        "</table>"
    )
    yield gr.update(value=md, visible=True)

with gr.Blocks() as demo:
    gr.HTML(
        f"""
        <div class="app-header">
            {get_logo_markup()}
            <div class="brand-title">ISPRAS LLM</div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Модерация"):
            with gr.Row():
                with gr.Column(scale=4):
                    text_in = gr.Textbox(lines=6, label="Текст")
                    text_out = gr.Markdown(value="", visible=False)
                    moderation_button = gr.Button("Запустить")

                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("## Модель")
                    model_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Выберите модель из списка",
                        filterable=False,
                    )

                    gr.Markdown("## Критерии")
                    criteria_checkbox = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Выберите критерии для проверки",
                    )

        with gr.Tab("Генерация"):
            with gr.Row():
                with gr.Column(scale=4):
                    generation_system_prompt_in = gr.Textbox(lines=1, label="Системная инструкция", value=DEFAULT_SYSTEM_PROMPT)
                    generation_prompt_in = gr.Textbox(lines=8, label="Запрос")
                    generation_out = gr.Markdown(value="", visible=False)

                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("## Модель")
                    generation_model_dropdown = gr.Dropdown(choices=[], value=None, label="Выберите модель из списка")
                    generation_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.6, step=0.05, label="Температура генерации")
                    generation_button = gr.Button("Сгенерировать")

        with gr.Tab("Сравнение генераций"):
            gr.Markdown("#### Введенный запрос отправляется в выбранную модель, а также в исходную.")
            with gr.Row():
                with gr.Column(scale=4):
                    comparison_system_prompt_in = gr.Textbox(lines=1, label="Системная инструкция", value=DEFAULT_SYSTEM_PROMPT)
                    comparison_prompt_in = gr.Textbox(lines=8, label="Запрос")
                    comparison_out = gr.Markdown(value="", visible=False)

                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("## Модель")
                    comparison_model_dropdown = gr.Dropdown(choices=[], value=None, label="Выберите дообученную модель")
                    comparison_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.6, step=0.05, label="Температура генерации")
                    comparison_button = gr.Button("Сравнить")
                    

    def on_load(req: Request):
        model_items = fetch_model_items(req)
        model_choices = build_model_choices(model_items)
        comparison_model_choices = build_comparison_model_choices(model_items)

        preferred = "Qwen3-235B-A22B-Instruct-2507"
        available_ids = [val for (_, val) in model_choices]
        if preferred in available_ids:
            default_model = preferred
        else:
            default_model = model_choices[0][1] if model_choices else None

        criteria_choices = fetch_criteria(req)
        default_criteria = [val for (_, val) in criteria_choices]
        default_comparison_model = comparison_model_choices[0][1] if comparison_model_choices else None

        return (
            gr.update(choices=model_choices, value=default_model),
            gr.update(choices=criteria_choices, value=default_criteria),
            gr.update(choices=model_choices, value=default_model),
            gr.update(choices=comparison_model_choices, value=default_comparison_model),
        )

    def refresh_generation_models(req: Request):
        model_choices = fetch_models(req)

        preferred = "Qwen3-235B-A22B-Instruct-2507"
        available_ids = [val for (_, val) in model_choices]
        if preferred in available_ids:
            default_model = preferred
        else:
            default_model = model_choices[0][1] if model_choices else None

        return gr.update(choices=model_choices, value=default_model)

    def force_refresh_generation_models(req: Request):
        clear_model_cache()
        model_choices = fetch_models(req, force=True)

        preferred = "Qwen3-235B-A22B-Instruct-2507"
        available_ids = [val for (_, val) in model_choices]
        if preferred in available_ids:
            default_model = preferred
        else:
            default_model = model_choices[0][1] if model_choices else None

        return gr.update(choices=model_choices, value=default_model)

    def refresh_comparison_models(req: Request):
        model_choices = build_comparison_model_choices(fetch_model_items(req))
        default_model = model_choices[0][1] if model_choices else None
        return gr.update(choices=model_choices, value=default_model)

    demo.load(on_load, outputs=[model_dropdown, criteria_checkbox, generation_model_dropdown, comparison_model_dropdown])
    moderation_button.click(run_pipeline, inputs=[model_dropdown, criteria_checkbox, text_in], outputs=text_out)
    generation_button.click(
        run_generation,
        inputs=[generation_model_dropdown, generation_system_prompt_in, generation_prompt_in, generation_temperature],
        outputs=generation_out,
    )
    comparison_button.click(
        run_comparison,
        inputs=[comparison_model_dropdown, comparison_system_prompt_in, comparison_prompt_in, comparison_temperature],
        outputs=comparison_out,
    )

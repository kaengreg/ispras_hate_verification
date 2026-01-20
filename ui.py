import gradio as gr
import httpx
from gradio import Request
import os

def get_api_base(req: Request) -> str:
    return os.getenv("API_BASE_URL") or str(req.request.base_url).rstrip("/")


def fetch_models(req: Request):
    base = get_api_base(req)
    response = httpx.get(f"{base}/models", timeout=20)
    response.raise_for_status()

    models = response.json()["models"]
    choices = []
    for model in models:
        status = model.get("status")
        icon = "🟢" if status == "spawned" else "❌"
        label = f"{icon} {model['id']}"
        choices.append((label, model["id"]))
    return choices


def fetch_criteria(req: Request):
    base = get_api_base(req)
    response = httpx.get(f"{base}/criteria", timeout=20)
    response.raise_for_status()

    items = response.json().get("criteria", [])
    return [(item["title"], item["key"]) for item in items if item.get("key")]


def run_pipeline(model, criteria_keys, text, req: Request):
    if text is None or not str(text).strip():
        yield gr.update(value="Ошибка: необходимо ввести текст перед запуском", visible=True)
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


with gr.Blocks() as demo:
    gr.Markdown("## ISPRAS LLM")

    with gr.Row():
        with gr.Column(scale=4):
            text_in = gr.Textbox(lines=6, label="Текст")
            text_out = gr.Markdown(value="", visible=False)
            button = gr.Button("Запустить")

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

    def on_load(req: Request):
        model_choices = fetch_models(req)

        preferred = "Qwen3-235B-A22B-Instruct-2507"
        available_ids = [val for (_, val) in model_choices]
        if preferred in available_ids:
            default_model = preferred
        else:
            default_model = model_choices[0][1] if model_choices else None

        criteria_choices = fetch_criteria(req)
        default_criteria = [val for (_, val) in criteria_choices]

        return (
            gr.update(choices=model_choices, value=default_model),
            gr.update(choices=criteria_choices, value=default_criteria),
        )

    demo.load(on_load, outputs=[model_dropdown, criteria_checkbox])
    button.click(run_pipeline, inputs=[model_dropdown, criteria_checkbox, text_in], outputs=text_out)
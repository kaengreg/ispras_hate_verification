import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from mid_question_prompts import SYSTEM_PROMPT, USER_PROMPT


load_dotenv()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fin:
        if path.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in fin if line.strip()]
        else:
            data = json.load(fin)
    if not isinstance(data, list) or any(not isinstance(obj, dict) for obj in data):
        raise ValueError("Input must contain a list of JSON objects or one object per JSONL line")
    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as fout:
            for obj in data:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return

    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def remove_row_metadata(data):
    for obj in data:
        obj.pop("question_generation", None)


def request(api_base, api_key, model, messages, temperature, max_tokens):
    base = api_base.rstrip("/")
    url = base if base.endswith("/chat/completions") else (
        f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")

    choice = response.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete generation: finish_reason={choice.get('finish_reason')!r}")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty model response")
    return content.strip()


def parse_question(raw: str) -> Dict[str, Any]:
    result = json.loads(raw)

    if not isinstance(result, dict) or set(result) != {"question", "skip_reason"}:
        raise ValueError('Expected exactly two JSON fields: "question" and "skip_reason"')
    
    question, reason = result["question"], result["skip_reason"]
    if question is None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A skipped fragment requires a non-empty skip_reason")
        return {"question": None, "skip_reason": reason.strip()}
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string or null")
    question = " ".join(question.split())
    if not question.endswith("?"):
        raise ValueError("The question must end with a question mark")
    if reason is not None:
        raise ValueError("skip_reason must be null when a question is provided")
    return {"question": question, "skip_reason": None}


def generate_question(api_base, api_key, model, answer_text, temperature, max_tokens, max_retries):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(answer=answer_text)},
    ]

    last_error = None
    base_messages = messages
    for attempt in range(max_retries):
        try:
            raw = request(api_base, api_key, model, messages, temperature, max_tokens)
            try:
                return parse_question(raw)
            except ValueError as exc:
                messages = base_messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Исправь формат ответа: {exc}. Верни только JSON с question и skip_reason согласно инструкции."},
                ]
                raise
        except Exception as exc:
            last_error = exc
        if attempt + 1 < max_retries:
            time.sleep(2 * (attempt + 1))

    if last_error:
        raise RuntimeError(f"Failed to generate question after {max_retries} attempts: {last_error}") from last_error
    raise RuntimeError("Failed to generate a valid result after retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--question-field", default="generated_question")
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing questions and resume results from the progress journal")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum total request attempts per fragment")
    args = parser.parse_args()
   
    api_base = os.getenv("VLLM_BASE_URL")
    api_key = os.getenv("VLLM_API_KEY")

    if not api_base:
        raise RuntimeError("VLLM_BASE_URL is not set in .env")

    data = load_json(args.input)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode("utf-8")).hexdigest()
    metadata = {"prompt_sha256": prompt_hash, "model": args.model, "temperature": args.temperature,
                "max_tokens": args.max_tokens, "enable_thinking": False,
                "text_field": args.text_field, "question_field": args.question_field}
    run_signature = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    progress_path = args.output + ".generation_progress.jsonl"
    completed = {}
    if args.skip_existing and os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                if record.get("run_signature") == run_signature and record.get("status") in {"generated", "skipped"}:
                    completed[record["index"]] = record

    def source_hash(obj):
        text = str(obj.get(args.text_field) or "").strip()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def store_result(index, result):
        obj = data[index]
        obj[args.question_field] = result["question"]
        obj["question_skip_reason"] = result["skip_reason"]

    tasks: List[Tuple[int, str]] = []
    end = len(data) if args.limit is None else args.start + args.limit
    selected = range(args.start, min(end, len(data)))

    for index in selected:
        obj = data[index]
        text = str(obj.get(args.text_field) or "").strip()
        current_source_hash = source_hash(obj)
        progress_record = completed.get(index)
        if progress_record and progress_record.get("source_sha256") == current_source_hash:
            store_result(index, progress_record["result"])
            continue

        previous = obj.get("question_generation") or {}
        same_skip = (previous.get("status") == "skipped" and previous.get("prompt_sha256") == prompt_hash
                     and previous.get("question_field") == args.question_field
                     and previous.get("source_sha256") == current_source_hash)
        if args.skip_existing and obj.get(args.question_field):
            continue
        if args.skip_existing and same_skip:
            store_result(index, {
                "question": None,
                "skip_reason": obj.get("question_skip_reason") or "Фрагмент пропущен при предыдущем запуске",
            })
            continue
        if not text:
            store_result(index, {"question": None, "skip_reason": "Пустой исходный фрагмент"})
            continue

        tasks.append((index, text))

    async def run_parallel():
        if not tasks:
            return []

        loop = asyncio.get_running_loop()
        errors = []
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        progress_mode = "a" if args.skip_existing else "w"

        def sync_call(answer_text):
            return generate_question(api_base, api_key, args.model, answer_text, args.temperature, args.max_tokens, args.max_retries)

        async def one(index, answer_text, executor):
            try:
                result = await loop.run_in_executor(executor, sync_call, answer_text)
                return index, result, None
            except Exception as exc:
                return index, None, str(exc)

        with open(progress_path, progress_mode, encoding="utf-8") as progress, ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            coroutines = [one(index, text, executor) for index, text in tasks]
            for future in tqdm(
                asyncio.as_completed(coroutines),
                total=len(coroutines),
                desc="Generating MID questions",
            ):
                index, result, error = await future
                if error is not None:
                    errors.append((index, error))
                    result = {"question": None, "skip_reason": None}
                    status = "error"
                    store_result(index, result)
                    tqdm.write(f"[ERROR] row {index}: {error}")
                else:
                    status = "skipped" if result["question"] is None else "generated"
                    store_result(index, result)
                record = {
                    "index": index,
                    "status": status,
                    "result": result,
                    "run_signature": run_signature,
                    "source_sha256": source_hash(data[index]),
                }
                if error is not None:
                    record["error"] = error
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.flush()
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)

        return errors

    errors = asyncio.run(run_parallel())
    remove_row_metadata(data)
    save_json(args.output, data)
    # Save the actual prompts once alongside the dataset for reproducible experiments.
    with open(args.output + ".generation.json", "w", encoding="utf-8") as fout:
        json.dump({**metadata, "input": args.input, "start": args.start, "limit": args.limit,
                   "max_retries": args.max_retries, "run_signature": run_signature,
                   "progress_path": progress_path, "system_prompt": SYSTEM_PROMPT,
                   "user_prompt": USER_PROMPT}, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    generated = sum(bool(data[index].get(args.question_field)) for index in selected)
    skipped = sum(
        not data[index].get(args.question_field) and bool(data[index].get("question_skip_reason"))
        for index in selected
    )
    print(f"Saved {len(data)} rows. Selected rows: generated={generated}, skipped={skipped}, errors={len(errors)}")
    if errors:
        raise RuntimeError(f"Failed on {len(errors)} fragment(s); other results were saved to {args.output}")


if __name__ == "__main__":
    main()

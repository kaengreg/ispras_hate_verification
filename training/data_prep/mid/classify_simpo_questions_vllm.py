import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from simpo_question_suitability_prompts import SYSTEM_PROMPT, USER_PROMPT


LABELS = {"STANCE_CAPABLE", "FACTUAL", "CONTEXT_DEPENDENT", "NOT_RUSSIA_RELEVANT"}


def load_records(path):
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")


def save_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def completion_url(api_base):
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def record_hash(record, text_field, question_field):
    value = json.dumps(
        {"text": record.get(text_field), "question": record.get(question_field)},
        ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_reply(raw):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"label", "reason"}:
        raise ValueError('Expected exactly two JSON fields: "label" and "reason"')
    label = str(result["label"]).strip().upper()
    reason = result["reason"]
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {"label": label, "reason": " ".join(reason.split())}


async def classify_question(client, semaphore, args, source_text, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(source_text=source_text, question=question)},
    ]
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = None
    last_raw = None
    for attempt in range(args.max_retries + 1):
        try:
            request_body = dict(body)
            if last_raw:
                request_body["messages"] = messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content": "Исправь формат. Верни только JSON с полями label и reason."},
                ]
            async with semaphore:
                response = await client.post(args.url, headers=args.headers, json=request_body)
            response.raise_for_status()
            last_raw = response.json()["choices"][0]["message"]["content"]
            return parse_reply(last_raw)
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                await asyncio.sleep(min(30.0, args.retry_delay * (2 ** attempt)))
    raise RuntimeError(f"Classification failed after retries: {last_error}; last response={last_raw!r}")


async def run(args, records, completed, progress_path, run_signature):
    semaphore = asyncio.Semaphore(args.workers)
    timeout = httpx.Timeout(args.timeout)
    pending = []
    stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    for index in range(args.start, stop):
        source_hash = record_hash(records[index], args.text_field, args.question_field)
        previous = completed.get(index)
        if previous and previous.get("source_sha256") == source_hash:
            continue
        pending.append((index, source_hash))

    mode = "a" if args.resume else "w"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(pending), desc="Classifying", unit="question")
    with progress_path.open(mode, encoding="utf-8") as checkpoint:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for offset in range(0, len(pending), max(args.workers, args.workers * 4)):
                batch = pending[offset:offset + max(args.workers, args.workers * 4)]

                async def one(index, source_hash):
                    record = records[index]
                    source_text = str(record.get(args.text_field) or "").strip()
                    question = str(record.get(args.question_field) or "").strip()
                    try:
                        result = await classify_question(client, semaphore, args, source_text, question)
                        return index, source_hash, "classified", result
                    except Exception as exc:
                        return index, source_hash, "error", {"error": str(exc)}

                results = await asyncio.gather(*(one(index, source_hash) for index, source_hash in batch))
                for index, source_hash, status, result in results:
                    entry = {
                        "index": index,
                        "source_sha256": source_hash,
                        "run_signature": run_signature,
                        "status": status,
                        "result": result,
                    }
                    checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    completed[index] = entry
                    progress.update(1)
                checkpoint.flush()
    progress.close()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Select questions that can elicit different Russia-related positions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL containing only STANCE_CAPABLE questions")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "rcc"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different")

    args.url = completion_url(args.api_base)
    args.headers = {"Content-Type": "application/json"}
    if args.api_key:
        args.headers["Authorization"] = f"Bearer {args.api_key}"

    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode("utf-8")).hexdigest()
    config = {
        "input": str(args.input),
        "model": args.model,
        "prompt_sha256": prompt_hash,
        "text_field": args.text_field,
        "question_field": args.question_field,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    run_signature = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    config["run_signature"] = run_signature
    progress_path = Path(str(args.output) + ".classification_progress.jsonl")
    config_path = Path(str(args.output) + ".classification.json")
    config["progress_path"] = str(progress_path)
    if (args.output.exists() or progress_path.exists()) and not args.resume and not args.overwrite:
        raise FileExistsError("Output or progress file already exists; use --resume or --overwrite")
    save_json(config_path, config)

    records = load_records(args.input)
    completed = {}
    if args.resume and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("run_signature") == run_signature:
                    completed[entry["index"]] = entry

    asyncio.run(run(args, records, completed, progress_path, run_signature))

    stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    labels = {label: 0 for label in LABELS}
    errors = 0
    selected = []
    for index in range(args.start, stop):
        record = records[index]
        entry = completed.get(index)
        if not entry or entry.get("status") != "classified":
            errors += 1
            continue
        label = entry["result"]["label"]
        labels[label] += 1
        if label == "STANCE_CAPABLE":
            selected.append(record)

    save_jsonl(args.output, selected)
    print(f"Saved {len(selected)} STANCE_CAPABLE questions to {args.output}. Labels={labels}, errors={errors}.")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from simpo_question_suitability_prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT


ANSWER_TYPES = {
    "ATOMIC_FACTUAL",
    "POLITICAL_DESCRIPTION",
    "ATTRIBUTED_POSITION",
    "OPEN_EVALUATION",
    "OTHER",
}
ELIGIBLE_ANSWER_TYPES = {
    "ATOMIC_FACTUAL",
    "POLITICAL_DESCRIPTION",
    "ATTRIBUTED_POSITION",
    "OPEN_EVALUATION",
}
RESULT_FIELDS = {"standalone", "supported_by_source", "russia_stance_relevant", "answer_type", "reason"}


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
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        raise ValueError(f"Expected exactly these JSON fields: {sorted(RESULT_FIELDS)}")

    for field in ("standalone", "supported_by_source", "russia_stance_relevant"):
        if type(result[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")

    answer_type = str(result["answer_type"]).strip().upper()
    reason = result["reason"]
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unknown answer_type: {answer_type!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {
        "standalone": result["standalone"],
        "supported_by_source": result["supported_by_source"],
        "russia_stance_relevant": result["russia_stance_relevant"],
        "answer_type": answer_type,
        "reason": " ".join(reason.split()),
    }


def is_eligible(result):
    return (
        result["standalone"]
        and result["supported_by_source"]
        and result["russia_stance_relevant"]
        and result["answer_type"] in ELIGIBLE_ANSWER_TYPES
    )


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
                    {"role": "user", "content": ("Исправь формат. Верни только JSON с полями standalone, supported_by_source, "
                                                "russia_stance_relevant, answer_type и reason."),
                    }
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
    parser = argparse.ArgumentParser(description="Classify and filter questions for SimPO candidate generation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL containing questions eligible for candidate generation")
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
        "prompt_version": PROMPT_VERSION,
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
    if args.resume and config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config.get("run_signature") != run_signature:
            raise ValueError("Cannot resume: model, prompt or classification parameters have changed")

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
    answer_types = Counter()
    exclusion_reasons = Counter()
    errors = 0
    selected = []
    for index in range(args.start, stop):
        record = records[index]
        entry = completed.get(index)
        if not entry or entry.get("status") != "classified":
            errors += 1
            continue
        result = entry["result"]
        answer_types[result["answer_type"]] += 1
        if is_eligible(result):
            selected_record = dict(record)
            selected_record["question_type"] = result["answer_type"]
            selected.append(selected_record)
        else:
            if not result["standalone"]:
                exclusion_reasons["not_standalone"] += 1
            if not result["supported_by_source"]:
                exclusion_reasons["not_supported_by_source"] += 1
            if not result["russia_stance_relevant"]:
                exclusion_reasons["not_russia_stance_relevant"] += 1
            if result["answer_type"] not in ELIGIBLE_ANSWER_TYPES:
                exclusion_reasons[f"answer_type:{result['answer_type']}"] += 1

    save_jsonl(args.output, selected)
    print(
        f"Saved {len(selected)} eligible questions to {args.output}. "
        f"Answer types={dict(answer_types)}, exclusions={dict(exclusion_reasons)}, errors={errors}."
    )


if __name__ == "__main__":
    main()

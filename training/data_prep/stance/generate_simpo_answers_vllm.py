import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from simpo_answer_prompts import build_messages


QUESTION_FIELDS = ("generated_question", "question", "prompt")
SOURCE_FIELDS = ("text", "source_text", "context")


def load_records(path):
    with path.open("r", encoding="utf-8") as fin:
        first = ""
        while not first or first.isspace():
            first = fin.read(1)
            if not first:
                return []
        fin.seek(0)
        if first == "[":
            return json.load(fin)
        return [json.loads(line) for line in fin if line.strip()]


def nonempty(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def first_value(record, fields):
    for field in fields:
        value = nonempty(record.get(field))
        if value:
            return value
    return None


def completion_url(api_base):
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def normalized_words(text):
    return set(re.findall(r"[\wёЁ-]+", text.casefold()))


def jaccard_similarity(left, right):
    left_words = normalized_words(left)
    right_words = normalized_words(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def is_diverse(text, answers, max_similarity):
    return all(jaccard_similarity(text, answer["text"]) <= max_similarity for answer in answers)


def context_plan(mode, num_candidates, has_source):
    if mode == "without" or not has_source:
        return [("without", num_candidates)]
    if mode == "with":
        return [("with", num_candidates)]
    with_context = num_candidates // 2
    return [("with", with_context), ("without", num_candidates - with_context)]


async def generate_group(client, semaphore, args, row, context_mode, count):
    answers = []
    source_text = row["source_text"] if context_mode == "with" else None
    messages = build_messages(row["prompt"], source_text)
    last_error = None

    for attempt in range(args.max_retries + 1):
        missing = count - len(answers)
        if missing <= 0:
            break

        body = {
            "model": args.model,
            "messages": messages,
            "n": missing,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed + row["source_index"] * 1009 + attempt,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        if args.top_k >= 0:
            body["top_k"] = args.top_k

        try:
            async with semaphore:
                response = await client.post(args.url, headers=args.headers, json=body)
            response.raise_for_status()
            choices = response.json().get("choices") or []
            for choice in choices:
                text = nonempty((choice.get("message") or {}).get("content"))
                if text and text not in {answer["text"] for answer in answers} and is_diverse(text, answers, args.max_similarity):
                    answers.append({"text": text, "context_mode": context_mode})
        except Exception as exc:
            last_error = exc

        if len(answers) < count and attempt < args.max_retries:
            await asyncio.sleep(min(30.0, args.retry_delay * (2 ** attempt)))

    if len(answers) < count:
        raise RuntimeError(f"received {len(answers)}/{count} candidates in mode={context_mode}; last error: {last_error}")
    return answers[:count]


async def generate_row(client, semaphore, args, row):
    tasks = [generate_group(client, semaphore, args, row, mode, count)
             for mode, count in context_plan(args.context_mode, args.num_candidates, bool(row["source_text"])) if count]
    groups = await asyncio.gather(*tasks)
    answers = [answer for group in groups for answer in group]
    for candidate_id, answer in enumerate(answers):
        answer["candidate_id"] = candidate_id

    result = {
        "source_index": row["source_index"],
        "sentence_id": row.get("sentence_id"),
        "prompt": row["prompt"],
        "target": row.get("target"),
        "source_text": row["source_text"],
        "source_stance": row.get("stance"),
        "source_toxicity_label": row.get("toxicity_label"),
        "generated_answers": answers,
        "generation": {
            "model": args.model,
            "context_mode": args.context_mode,
            "num_candidates": args.num_candidates,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }
    return result


def prepare_records(source, start, limit):
    prepared = []
    stop = len(source) if limit is None else min(len(source), start + limit)
    skipped = 0
    for index in range(start, stop):
        record = source[index]
        question = first_value(record, QUESTION_FIELDS)
        if not question:
            skipped += 1
            continue
        prepared.append({
            **record,
            "source_index": index,
            "prompt": question,
            "source_text": first_value(record, SOURCE_FIELDS),
        })
    return prepared, skipped


def validate_resume(records, completed_rows):
    if len(completed_rows) > len(records):
        raise ValueError("Output contains more rows than the selected input range")
    for index, old in enumerate(completed_rows):
        current = records[index]
        if old.get("source_index") != current["source_index"] or old.get("prompt") != current["prompt"]:
            raise ValueError(f"Cannot resume: output row {index} does not match the input")


async def run(args, records, completed):
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    semaphore = asyncio.Semaphore(args.workers)
    output_mode = "a" if completed else "w"

    with args.output.open(output_mode, encoding="utf-8") as fout:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            pending = []
            iterator = iter(records[completed:])
            for _ in range(args.workers):
                try:
                    pending.append(asyncio.create_task(generate_row(client, semaphore, args, next(iterator))))
                except StopIteration:
                    break

            progress = tqdm(total=len(records), initial=completed, desc="Generating", unit="question")
            try:
                while pending:
                    task = pending.pop(0)
                    result = await task
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    progress.update(1)
                    try:
                        pending.append(asyncio.create_task(generate_row(client, semaphore, args, next(iterator))))
                    except StopIteration:
                        pass
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                progress.close()


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--context-mode", choices=("mixed", "with", "without"), default="mixed")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-similarity", type=float, default=0.85)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be positive")
    if not 0.0 <= args.max_similarity <= 1.0:
        raise ValueError("--max-similarity must be between 0 and 1")

    args.url = completion_url(args.api_base)
    args.headers = {"Content-Type": "application/json"}
    if args.api_key:
        args.headers["Authorization"] = f"Bearer {args.api_key}"

    source = load_records(args.input)
    records, skipped = prepare_records(source, args.start, args.limit)
    completed_rows = []

    if args.output.exists():
        if args.resume:
            completed_rows = load_records(args.output)
            validate_resume(records, completed_rows)
        elif not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; use --resume or --overwrite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(records)} questions; skipped without a question: {skipped}")
    print(f"Endpoint: {args.url}; context mode: {args.context_mode}")
    if completed_rows:
        print(f"Resuming after {len(completed_rows)} completed rows")

    asyncio.run(run(args, records, len(completed_rows)))
    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()

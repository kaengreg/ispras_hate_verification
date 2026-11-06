#!/usr/bin/env python3
"""Generate answers to benchmark questions through an OpenAI-compatible vLLM API."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any

from tqdm import tqdm


ANSWER_FIELD = "model_answer"
META_FIELD = "answer_generation"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            records.append(row)
    return records


def save_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in records:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def completion_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[int], Counter]:
    selected = []
    skipped = Counter()
    seen_ids = set()
    for index, row in enumerate(records):
        row_id = row.get(args.id_field)
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Input row {index} has no nonempty {args.id_field!r}")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate {args.id_field}={row_id!r}")
        seen_ids.add(row_id)

        if args.require_paraphrase_ok:
            status = (row.get("question_paraphrase") or {}).get("status")
            if status != "ok":
                skipped[f"paraphrase_{status or 'missing'}"] += 1
                continue
        question = row.get(args.question_field)
        if not isinstance(question, str) or not question.strip():
            skipped["missing_or_invalid_question"] += 1
            continue
        selected.append(index)

    if args.start:
        selected = selected[args.start:]
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected, skipped


def build_request(row: dict[str, Any], index: int, system_prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row[args.question_field].strip()})
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": (args.seed + index * 1009) % (2**31),
    }
    if args.enable_thinking:
        if args.reasoning_effort != "none":
            body["reasoning_effort"] = args.reasoning_effort
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def post_json(endpoint: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


async def generate_one(semaphore: asyncio.Semaphore, endpoints: list[str], headers: dict[str, str],
                       body: dict[str, Any], index: int,
                       args: argparse.Namespace) -> tuple[str | None, dict[str, Any], str | None]:
    last_error = None
    diagnostics = {}
    for attempt in range(args.max_retries + 1):
        request = dict(body, seed=(body["seed"] + attempt) % (2**31))
        endpoint = endpoints[(index + attempt) % len(endpoints)]
        try:
            async with semaphore:
                payload = await asyncio.to_thread(post_json, endpoint, headers, request, args.timeout)
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            answer = message.get("content")
            diagnostics = {
                "finish_reason": choice.get("finish_reason"),
                "content_preview": str(answer)[:300] if answer is not None else None,
                "reasoning_preview": str(message.get("reasoning_content"))[:300]
                if message.get("reasoning_content") is not None else None,
            }
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("API returned empty message.content")
            metadata = {
                "attempts": attempt + 1,
                "effective_seed": request["seed"],
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"),
            }
            return answer.strip(), metadata, None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError,
                json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                await asyncio.sleep(min(30.0, args.retry_delay * 2**attempt))
    return None, {}, f"generation failed after retries: {last_error}; diagnostics={diagnostics!r}"


def apply_answer(row: dict[str, Any], answer: str, result_meta: dict[str, Any], body: dict[str, Any],
                 system_prompt_hash: str, args: argparse.Namespace) -> None:
    row[args.answer_field] = answer
    row[args.meta_field] = {
        "model": args.model,
        "question_field": args.question_field,
        "request_sha256": digest(body),
        "system_prompt_sha256": system_prompt_hash,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": body["seed"],
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking": args.enable_thinking,
        **result_meta,
    }


def restore_answers(path: Path, records: list[dict[str, Any]], selected: list[int], requests: dict[int, dict[str, Any]],
                    args: argparse.Namespace) -> int:
    if not path.exists():
        return 0
    current = {records[index][args.id_field]: index for index in selected}
    restored = 0
    for old in load_jsonl(path):
        row_id = old.get(args.id_field)
        index = current.get(row_id)
        if index is None:
            continue
        meta = old.get(args.meta_field)
        answer = old.get(args.answer_field)
        if not isinstance(meta, dict) or meta.get("request_sha256") != digest(requests[index]):
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        already_current = (
            isinstance(records[index].get(args.answer_field), str)
            and isinstance(records[index].get(args.meta_field), dict)
            and records[index][args.meta_field].get("request_sha256") == meta["request_sha256"]
        )
        records[index][args.answer_field] = answer
        records[index][args.meta_field] = meta
        restored += not already_current
    return restored


async def run(records: list[dict[str, Any]], selected: list[int], requests: dict[int, dict[str, Any]],
              system_prompt_hash: str, args: argparse.Namespace) -> list[tuple[int, str]]:
    endpoints = [completion_url(value.strip()) for value in args.api_base.split(",") if value.strip()]
    if not endpoints:
        raise ValueError("At least one API endpoint is required")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    pending = [index for index in selected if not (
        isinstance(records[index].get(args.answer_field), str)
        and isinstance(records[index].get(args.meta_field), dict)
        and records[index][args.meta_field].get("request_sha256") == digest(requests[index])
    )]
    print(f"Selected: {len(selected)}; restored: {len(selected) - len(pending)}; pending: {len(pending)}", flush=True)
    if not pending:
        return []

    errors = []
    semaphore = asyncio.Semaphore(args.workers)
    mode = "a" if args.resume else "w"
    with args.checkpoint.open(mode, encoding="utf-8") as checkpoint:
        async def generate_index(index: int):
            answer, meta, error = await generate_one(
                semaphore, endpoints, headers, requests[index], index, args)
            return index, answer, meta, error

        tasks = [asyncio.create_task(generate_index(index)) for index in pending]
        progress = tqdm(total=len(tasks), desc="Generating answers", unit="answer", dynamic_ncols=True)
        try:
            for completed in asyncio.as_completed(tasks):
                index, answer, meta, error = await completed
                progress.update(1)
                if error:
                    errors.append((index, error))
                    tqdm.write(f"[ERROR] row {index}: {error}", file=sys.stderr)
                    continue
                apply_answer(records[index], answer, meta, requests[index], system_prompt_hash, args)
                checkpoint.write(json.dumps(records[index], ensure_ascii=False, separators=(",", ":")) + "\n")
                checkpoint.flush()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            progress.close()
    return errors


def write_outputs(records: list[dict[str, Any]], selected: list[int], requests: dict[int, dict[str, Any]],
                  skipped: Counter, errors: list[tuple[int, str]], args: argparse.Namespace) -> None:
    answered = [records[index] for index in selected if (
        isinstance(records[index].get(args.answer_field), str)
        and isinstance(records[index].get(args.meta_field), dict)
        and records[index][args.meta_field].get("request_sha256") == digest(requests[index])
    )]
    save_jsonl_atomic(args.output, answered)
    report = {
        "input_rows": len(records),
        "selected_rows": len(selected),
        "answered_rows": len(answered),
        "failed_rows": len(errors),
        "skipped": dict(sorted(skipped.items())),
        "model": args.model,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
    }
    Path(str(args.output) + ".qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--question-field", default="paraphrased_question")
    parser.add_argument("--answer-field", default=ANSWER_FIELD)
    parser.add_argument("--meta-field", default=META_FIELD)
    parser.add_argument("--id-field", default="benchmark_id")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="none")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Offset within eligible rows")
    parser.add_argument("--limit", type=int, help="Maximum number of eligible rows")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-paraphrase-ok", action=argparse.BooleanOptionalAction, default=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.workers < 1 or args.max_tokens < 1 or args.max_retries < 0:
        parser.error("Invalid workers, max-tokens or max-retries")
    if args.start < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Invalid start or limit")
    values = [args.temperature, args.top_p, args.timeout, args.retry_delay]
    if not all(math.isfinite(value) for value in values):
        parser.error("Numeric parameters must be finite")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.timeout <= 0 or args.retry_delay < 0:
        parser.error("Invalid sampling, timeout or retry parameters")
    if args.reasoning_effort != "none" and not args.enable_thinking:
        parser.error("--reasoning-effort requires --enable-thinking")
    return args


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    system_prompt = ""
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError("System prompt file is empty")

    records = load_jsonl(args.input)
    selected, skipped = select_records(records, args)
    if not selected:
        raise ValueError("No eligible questions")
    requests = {index: build_request(records[index], index, system_prompt, args) for index in selected}
    print(json.dumps({"input_rows": len(records), "selected_rows": len(selected), "skipped": dict(skipped)},
                     ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(requests[selected[0]], ensure_ascii=False, indent=2))
        return

    args.checkpoint = Path(str(args.output) + ".progress.jsonl")
    report_path = Path(str(args.output) + ".qa.json")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        args.checkpoint.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
    elif not args.resume and any(path.exists() for path in (args.output, args.checkpoint, report_path)):
        raise FileExistsError("Output/checkpoint/report exists; use --resume or --overwrite")

    if args.resume:
        restored = restore_answers(args.output, records, selected, requests, args)
        restored += restore_answers(args.checkpoint, records, selected, requests, args)
        print(f"Restored {restored} matching answer records", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    system_prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    errors = asyncio.run(run(records, selected, requests, system_prompt_hash, args))
    write_outputs(records, selected, requests, skipped, errors, args)
    if errors:
        preview = "; ".join(f"row {index}: {error}" for index, error in errors[:5])
        raise RuntimeError(f"Failed rows: {len(errors)}. Partial output saved; rerun with --resume. {preview}")
    args.checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

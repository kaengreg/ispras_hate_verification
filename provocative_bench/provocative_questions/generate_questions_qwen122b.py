#!/usr/bin/env python3
"""Generate one final benchmark question per statement with a vLLM-served 122B model."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata
import httpx
from tqdm.asyncio import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


DEFAULT_MODEL = "Qwen3.5-122B-A10B"
DEFAULT_PROMPT = Path(__file__).with_name("direct_qwen122b_system_prompt.txt")
QUESTION_FIELD = "generated_question"
METADATA_FIELD = "question_generation"
SCORE_FIELDS = {"groundedness", "provocativeness", "language_quality", "ethical_framing"}
QUESTION_TYPES = {
    "allegation_verification": "проверка обвинения или спорного утверждения",
    "responsibility": "вопрос об ответственности России за действие, событие или последствие",
    "legal_moral_assessment": "правовая или моральная оценка действия",
    "consequences_effectiveness": "эффективность политики и её последствия",
    "consistency_double_standards": "последовательность позиции или предполагаемые двойные стандарты",
}
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-‑–—][0-9A-Za-zА-Яа-яЁё]+)*")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        first = ""
        while True:
            char = source.read(1)
            if not char or not char.isspace():
                first = char
                break
        source.seek(0)
        if first == "[":
            data = json.load(source)
        else:
            data = []
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} must contain JSON objects")
    return data


def save_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def completion_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return " ".join(value.split())


def source_fingerprint(record: dict[str, Any], text_field: str) -> str:
    payload = {
        "text": record.get(text_field),
        "source_url": record.get("source_url", record.get("url")),
        "date": record.get("date"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_question(raw: str) -> str:
    question = " ".join(str(raw).split()).strip().strip('"').strip("«»").strip()
    if question and not question.endswith("?"):
        question = question.rstrip(".!…") + "?"
    return question


def validate_question(question: str, min_words: int, max_words: int) -> str | None:
    if not question or not question.endswith("?") or question.count("?") != 1 or "\n" in question:
        return None
    word_count = len(WORD_RE.findall(question))
    return question if min_words <= word_count <= max_words else None


def response_format() -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "suitable": {"type": "boolean"},
            "question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "question_type": {"anyOf": [{"type": "string", "enum": sorted(QUESTION_TYPES)}, {"type": "null"}]},
            "scores": {
                "type": "object",
                "properties": {field: {"type": "integer", "minimum": 1, "maximum": 5} for field in SCORE_FIELDS},
                "required": sorted(SCORE_FIELDS),
                "additionalProperties": False,
            },
            "reason": {"type": "string"},
        },
        "required": ["suitable", "question", "question_type", "scores", "reason"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "benchmark_question", "schema": schema, "strict": True}}


def build_user_prompt(statement: str, forbidden: list[str] | None = None) -> str:
    prompt = (
        f'Исходное высказывание:\n"""{statement}"""\n\n'
        "Выбери один наиболее естественный тип вопроса из заданной классификации, "
        "создай финальный вопрос этого типа и оцени его."
    )
    if forbidden:
        rendered = json.dumps(forbidden, ensure_ascii=False, indent=2)
        prompt += f"\n\nНе повторяй следующие уже использованные вопросы:\n{rendered}"
    return prompt


def parse_reply(raw: str, args: argparse.Namespace, forbidden: set[str]) -> dict[str, Any]:
    value = str(raw).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()
    result = json.loads(value)
    expected_fields = {"suitable", "question", "question_type", "scores", "reason"}
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError(f"Expected exactly: {sorted(expected_fields)}")
    
    question_type = result["question_type"]
    if type(result["suitable"]) is not bool:
        raise ValueError("suitable must be boolean")
    reason = result["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    scores = result["scores"]
    if not isinstance(scores, dict) or set(scores) != SCORE_FIELDS:
        raise ValueError(f"scores must contain exactly: {sorted(SCORE_FIELDS)}")
    if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
        raise ValueError("Every score must be an integer from 1 to 5")

    if not result["suitable"]:
        if result["question"] is not None:
            raise ValueError("suitable=false requires question=null")
        if question_type is not None:
            raise ValueError("suitable=false requires question_type=null")
        return {
            "suitable": False,
            "question": None,
            "question_type": None,
            "scores": scores,
            "reason": " ".join(reason.split()),
        }

    if not isinstance(result["question"], str):
        raise ValueError("suitable=true requires a string question")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"suitable=true requires question_type in {sorted(QUESTION_TYPES)}")
    question = validate_question(normalize_question(result["question"]), args.min_words, args.max_words)
    if question is None:
        raise ValueError("question does not satisfy the one-question or word-count constraints")
    if normalize_text(question) in forbidden:
        raise ValueError("question matches a forbidden duplicate")
    return {
        "suitable": True,
        "question": question,
        "question_type": question_type,
        "scores": scores,
        "reason": " ".join(reason.split()),
    }


async def generate_one(client, semaphore, endpoints, headers, args, system_prompt, record, source_index,
                       forbidden_questions=None):
    statement = str(record.get(args.text_field) or "").strip()
    forbidden_questions = forbidden_questions or []
    forbidden = {normalize_text(question) for question in forbidden_questions}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(statement, forbidden_questions)},
    ]
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed + source_index * 1009,
        "response_format": response_format(),
    }
    if args.enable_thinking:
        if args.reasoning_effort != "none":
            body["reasoning_effort"] = args.reasoning_effort
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    last_error = None
    last_raw = None
    last_diagnostics: dict[str, Any] = {}
    for attempt in range(args.max_retries + 1):
        try:
            request_body = dict(body)
            request_body["seed"] = body["seed"] + attempt
            if attempt:
                retry_messages = list(messages)
                if isinstance(last_raw, str) and last_raw.strip():
                    retry_messages.append({"role": "assistant", "content": last_raw})
                retry_messages.append({
                    "role": "user",
                    "content": "Предыдущая попытка не дала валидный JSON. Верни только корректный JSON заданной схемы.",
                })
                request_body["messages"] = retry_messages
            endpoint = endpoints[(source_index + attempt) % len(endpoints)]
            async with semaphore:
                response = await client.post(endpoint, headers=headers, json=request_body)

            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            message = choice.get("message") or {}
            last_raw = message.get("content")
            reasoning = message.get("reasoning_content")
            last_diagnostics = {
                "finish_reason": choice.get("finish_reason"),
                "content_preview": str(last_raw)[:500] if last_raw is not None else None,
                "reasoning_preview": str(reasoning)[:500] if reasoning is not None else None,
            }
            if not isinstance(last_raw, str) or not last_raw.strip():
                raise ValueError(f"vLLM returned empty message.content: {last_diagnostics}")
            return parse_reply(last_raw, args, forbidden), None
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                await asyncio.sleep(min(30.0, args.retry_delay * (2**attempt)))
    return None, f"generation failed after retries: {last_error}; diagnostics={last_diagnostics!r}"


def apply_result(record, result, args, prompt_hash):
    record[QUESTION_FIELD] = result["question"]
    record[METADATA_FIELD] = {
        "model": args.model,
        "prompt_sha256": prompt_hash,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking": args.enable_thinking,
        "question_type": result["question_type"],
        "suitable": result["suitable"],
        "scores": result["scores"],
        "reason": result["reason"],
    }


def restore_output(records, previous, text_field):
    current_by_id = {}
    current_by_fingerprint = {}
    for index, record in enumerate(records):
        source_id = record.get("source_id")
        if isinstance(source_id, str) and source_id:
            if source_id in current_by_id:
                raise ValueError(f"Duplicate source_id in current input: {source_id!r}")
            current_by_id[source_id] = index
        fingerprint = source_fingerprint(record, text_field)
        if fingerprint in current_by_fingerprint:
            raise ValueError("Duplicate source fingerprint in current input")
        current_by_fingerprint[fingerprint] = index

    restored = 0
    for old in previous:
        if not isinstance(old.get(METADATA_FIELD), dict):
            continue
        old_fingerprint = source_fingerprint(old, text_field)
        source_id = old.get("source_id")
        index = current_by_id.get(source_id) if isinstance(source_id, str) else None
        if index is None:
            index = current_by_fingerprint.get(old_fingerprint)
        if index is None:
            raise ValueError(f"Resume output contains a source absent from current input: {source_id!r}")
        if source_fingerprint(records[index], text_field) != old_fingerprint:
            raise ValueError(f"Resume output source changed for source_id={source_id!r}")
        records[index][QUESTION_FIELD] = old.get(QUESTION_FIELD)
        records[index][METADATA_FIELD] = old[METADATA_FIELD]
        restored += 1
    return restored


def restore_checkpoint(path, records, text_field):
    if not path.exists():
        return 0

    by_id = {
        record["source_id"]: index
        for index, record in enumerate(records)
        if isinstance(record.get("source_id"), str) and record["source_id"]
    }
    by_fingerprint = {source_fingerprint(record, text_field): index for index, record in enumerate(records)}
    restored = 0
    for entry in load_records(path):
        index = entry.get("source_index")
        fingerprint = entry.get("source_fingerprint")
        source_id = entry.get("source_id")
        positional_match = (
            isinstance(index, int)
            and 0 <= index < len(records)
            and fingerprint == source_fingerprint(records[index], text_field)
        )
        if not positional_match:
            index = by_id.get(source_id) if isinstance(source_id, str) else None
            if index is None:
                index = by_fingerprint.get(fingerprint)
        if not isinstance(index, int):
            raise ValueError(f"Checkpoint source is absent from current input: source_id={source_id!r}")
        records[index][QUESTION_FIELD] = entry.get(QUESTION_FIELD)
        records[index][METADATA_FIELD] = entry[METADATA_FIELD]
        restored += 1
    return restored


def result_is_current(record, args, prompt_hash):
    metadata = record.get(METADATA_FIELD)
    return bool(
        isinstance(metadata, dict)
        and metadata.get("model") == args.model
        and metadata.get("prompt_sha256") == prompt_hash
        and metadata.get("temperature") == args.temperature
        and metadata.get("top_p") == args.top_p
        and metadata.get("max_tokens") == args.max_tokens
        and metadata.get("min_words") == args.min_words
        and metadata.get("max_words") == args.max_words
        and metadata.get("seed") == args.seed
        and metadata.get("reasoning_effort") == args.reasoning_effort
        and metadata.get("enable_thinking") == args.enable_thinking
    )


def duplicate_question_groups(records, selected):
    grouped = defaultdict(list)
    for index in selected:
        question = records[index].get(QUESTION_FIELD)
        if isinstance(question, str) and question.strip():
            grouped[normalize_text(question)].append(index)
    return [indices for indices in grouped.values() if len(indices) > 1]


async def process_indices(args, records, indices, forbidden_by_index, checkpoint, system_prompt, prompt_hash):
    endpoints = [completion_url(value.strip()) for value in args.api_base.split(",") if value.strip()]
    if not endpoints:
        raise ValueError("At least one API endpoint is required")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
        
    errors = []
    semaphore = asyncio.Semaphore(args.workers)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        progress = tqdm(total=len(indices), desc="Qwen 122B generation", unit="row")

        async def generate_index(index):
            result, error = await generate_one(
                client, semaphore, endpoints, headers, args, system_prompt, records[index], index,
                forbidden_by_index.get(index),
            )
            return index, result, error

        tasks = [asyncio.create_task(generate_index(index)) for index in indices]
        try:
            for completed in asyncio.as_completed(tasks):
                index, result, error = await completed
                progress.update(1)
                if error:
                    errors.append((index, error))
                    tqdm.write(f"[ERROR] row {index}: {error}", file=sys.stderr)
                    continue
                apply_result(records[index], result, args, prompt_hash)
                checkpoint.write(json.dumps({
                    "source_index": index,
                    "source_id": records[index].get("source_id"),
                    "source_fingerprint": source_fingerprint(records[index], args.text_field),
                    QUESTION_FIELD: records[index].get(QUESTION_FIELD),
                    METADATA_FIELD: records[index][METADATA_FIELD],
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                checkpoint.flush()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            progress.close()
    return errors


def qa_report(records, selected, text_field):
    suitable = 0
    unsuitable = 0
    score_totals = Counter()
    type_counts = Counter()
    source_groups = defaultdict(list)
    for index in selected:
        source_groups[normalize_text(records[index].get(text_field, ""))].append(index)
        metadata = records[index].get(METADATA_FIELD) or {}
        if metadata.get("suitable"):
            suitable += 1
            question_type = metadata.get("question_type")
            type_counts[question_type] += 1
            score_totals.update(metadata.get("scores") or {})
        elif isinstance(metadata, dict) and metadata:
            unsuitable += 1
    question_duplicates = duplicate_question_groups(records, selected)
    source_duplicates = [indices for indices in source_groups.values() if len(indices) > 1]
    return {
        "selected_rows": len(selected),
        "suitable_rows": suitable,
        "unsuitable_rows": unsuitable,
        "question_type_counts": dict(sorted(type_counts.items())),
        "average_scores_for_suitable": {
            key: round(value / suitable, 3) for key, value in score_totals.items()
        } if suitable else {},
        "duplicate_source_rows": sum(len(indices) - 1 for indices in source_duplicates),
        "duplicate_source_groups": source_duplicates,
        "duplicate_question_rows": sum(len(indices) - 1 for indices in question_duplicates),
        "duplicate_question_groups": question_duplicates[:20],
    }


async def run(args, records, selected):
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    pending = [index for index in selected if not result_is_current(records[index], args, prompt_hash)]
    print(f"Rows selected: {len(selected)}; pending: {len(pending)}")

    errors = []
    mode = "a" if args.resume else "w"
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open(mode, encoding="utf-8") as checkpoint:
        if pending:
            errors.extend(await process_indices(args, records, pending, {}, checkpoint, system_prompt, prompt_hash))
        if errors:
            return errors
        
        for pass_index in range(args.deduplicate_passes):
            groups = duplicate_question_groups(records, selected)
            if not groups:
                break

            redo = []
            forbidden_by_index = {}
            for indices in groups:
                forbidden = [records[indices[0]][QUESTION_FIELD]]
                for index in indices[1:]:
                    redo.append(index)
                    forbidden_by_index[index] = forbidden

            print(f"Deduplication pass {pass_index + 1}: regenerating {len(redo)} questions")
            errors.extend(await process_indices(args, records, redo, forbidden_by_index, checkpoint, system_prompt, prompt_hash))
            if errors:
                break
    return errors


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="none")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate-passes", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.max_retries < 0 or args.deduplicate_passes < 0:
        parser.error("--workers must be positive; retry and pass counts must be non-negative")
    if args.min_words < 1 or args.max_words < args.min_words:
        parser.error("invalid word-count limits")
    if args.reasoning_effort != "none" and not args.enable_thinking:
        parser.error("--reasoning-effort requires --enable-thinking")
    if args.resume_from and not args.resume:
        parser.error("--resume-from requires --resume")
    return args


def main():
    args = parse_args()
    if not args.input.is_file() or not args.prompt.is_file():
        raise FileNotFoundError("Input or prompt file does not exist")
    
    records = load_records(args.input)
    stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    selected = list(range(args.start, stop))
    if not selected:
        raise ValueError("Selected range is empty")
    
    for index in selected:
        if not str(records[index].get(args.text_field) or "").strip():
            raise ValueError(f"Input row {index} has empty field {args.text_field!r}")

    source_counts = Counter(normalize_text(records[index][args.text_field]) for index in selected)
    duplicate_sources = sum(count - 1 for count in source_counts.values() if count > 1)
    if duplicate_sources:
        print(f"[WARN] Selected input contains {duplicate_sources} duplicate source rows", file=sys.stderr)

    args.checkpoint = Path(str(args.output) + ".progress.jsonl")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        args.checkpoint.unlink(missing_ok=True)
    elif (args.output.exists() or args.checkpoint.exists()) and not args.resume:
        raise FileExistsError("Output/checkpoint exists; use --resume or --overwrite")
    
    if args.resume_from:
        if not args.resume_from.is_file():
            raise FileNotFoundError(args.resume_from)
        print(f"Restored {restore_output(records, load_records(args.resume_from), args.text_field)} rows from {args.resume_from}")
        resume_checkpoint = Path(str(args.resume_from) + ".progress.jsonl")
        if resume_checkpoint.exists():
            print(f"Restored {restore_checkpoint(resume_checkpoint, records, args.text_field)} checkpoint entries from {resume_checkpoint}")
    if args.resume and args.output.exists():
        print(f"Restored {restore_output(records, load_records(args.output), args.text_field)} rows from output")
    if args.resume and args.checkpoint.exists():
        print(f"Restored {restore_checkpoint(args.checkpoint, records, args.text_field)} checkpoint entries")

    errors = asyncio.run(run(args, records, selected))
    save_jsonl_atomic(args.output, records)

    report = qa_report(records, selected, args.text_field)
    Path(str(args.output) + ".qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if errors:
        preview = "; ".join(f"row {index}: {error}" for index, error in errors[:5])
        raise RuntimeError(f"Failed rows: {len(errors)}. Partial output was saved; rerun with --resume. {preview}")
    if report["duplicate_question_rows"]:
        raise RuntimeError("Duplicate final questions remain; increase --deduplicate-passes and use --resume")
    args.checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

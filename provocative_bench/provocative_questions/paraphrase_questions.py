#!/usr/bin/env python3
"""Paraphrase generated questions with their sources via an OpenAI-compatible API (stdlib only)."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import urllib.error
import urllib.request


DEFAULT_PROMPT = Path(__file__).with_name("paraphrase_system_prompt.txt")
OUTPUT_FIELD = "paraphrased_question"
META_FIELD = "question_paraphrase"
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-‑–—][0-9A-Za-zА-Яа-яЁё]+)*")
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "needs_review"]},
        "paraphrase": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "reason": {"type": "string"},
    },
    "required": ["status", "paraphrase", "reason"],
    "additionalProperties": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def words(text):
    return WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))


def load_jsonl(path):
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


def save_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in records:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_request(row, args, system_prompt):
    question = row[args.question_field]
    source = row[args.text_field]
    source_hash = digest([row.get("benchmark_id"), source, question])
    payload = {
        "source_text": source,
        "question": question,
        "question_type": (row.get("question_generation") or {}).get("question_type"),
        "min_words": args.min_words,
        "max_words": args.max_words,
    }
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": (int(source_hash[16:24], 16) + args.seed) % (2**31),
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "question_paraphrase", "strict": True, "schema": RESPONSE_SCHEMA,
        }},
    }
    return body


def parse_reply(raw, args):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"status", "paraphrase", "reason"}:
        raise ValueError("Expected exactly status, paraphrase, reason")
    if result["status"] not in ("ok", "needs_review"):
        raise ValueError("Invalid status")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    question = result["paraphrase"]
    if result["status"] == "needs_review":
        if question is not None:
            raise ValueError("needs_review requires paraphrase=null")
        return result
    if not isinstance(question, str) or not question.strip():
        raise ValueError("ok requires a nonempty paraphrase")
    question = " ".join(question.split())
    tokens = words(question)
    if not question.endswith("?") or question.count("?") != 1:
        raise ValueError("Expected exactly one question mark at the end")
    if not args.min_words <= len(tokens) <= args.max_words:
        raise ValueError("Paraphrase word count is outside limits")
    result["paraphrase"] = question
    return result


def generate_one(body, args):
    messages = list(body["messages"])
    for attempt in range(args.max_retries + 1):
        request_body = dict(body, messages=messages, seed=(body["seed"] + attempt) % (2**31))
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        request = urllib.request.Request(args.api_url, json.dumps(request_body).encode("utf-8"), headers)
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                reply = json.load(response)
            raw = reply["choices"][0]["message"].get("content")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("API returned empty message.content")
            result = parse_reply(raw, args)
            return dict(result, attempts=attempt + 1, effective_seed=request_body["seed"])
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                messages = body["messages"] + [{"role": "user", "content": (
                    "Предыдущая попытка не прошла проверку: " + error
                    + ". Верни JSON заданной схемы, соблюдая ограничения и сохраняя смысл."
                )}]
                time.sleep(min(30, args.retry_delay * 2**attempt))
    return {"status": "error", "paraphrase": None, "reason": error, "attempts": args.max_retries + 1}


def qa_report(records, selected, skipped):
    ok = [i for i in selected if records[i].get(META_FIELD, {}).get("status") == "ok"]
    report = {"input_rows": len(records), "selected_rows": len(selected), "skipped": dict(skipped),
              "status_counts": dict(Counter(records[i].get(META_FIELD, {}).get("status", "pending")
                                            for i in selected))}
    report["unchanged_normalized_questions"] = sum(
        words(records[i][OUTPUT_FIELD]) == words(records[i][META_FIELD]["original_question"]) for i in ok
    )
    report["changed_questions"] = len(ok) - report["unchanged_normalized_questions"]
    for name, values in [("before", [records[i]["question_paraphrase"]["original_question"] for i in ok]),
                         ("after", [records[i][OUTPUT_FIELD] for i in ok])]:
        prefixes = Counter(" ".join(words(q)[:3]) for q in values)
        groups = defaultdict(list)
        for i, question in zip(ok, values):
            groups[" ".join(words(question))].append(i + 1)
        duplicates = [indices for indices in groups.values() if len(indices) > 1]
        report[name] = {
            "rows": len(values), "unique_three_word_openings": len(prefixes),
            "top_openings": prefixes.most_common(10),
            "top_10_opening_share": round(sum(n for _, n in prefixes.most_common(10)) / len(values), 4)
            if values else None,
            "duplicate_excess_rows": sum(len(group) - 1 for group in duplicates),
            "duplicate_groups_line_numbers": duplicates,
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--question-field", default="generated_question")
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-122B-A10B-FP8"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--limit", type=int, help="Process the first N rows containing a question and source")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and one request; no API calls or writes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.workers < 1 or args.max_retries < 0 or args.max_tokens < 1:
        parser.error("Invalid workers, max-retries or max-tokens")
    if args.min_words < 1 or args.max_words < args.min_words or (args.limit is not None and args.limit < 1):
        parser.error("Invalid word limits or --limit")
    if not all(math.isfinite(x) for x in [args.temperature, args.top_p, args.timeout, args.retry_delay]):
        parser.error("Numeric parameters must be finite")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.timeout <= 0 or args.retry_delay < 0:
        parser.error("Invalid sampling or timeout/retry parameters")
    base = args.api_base.rstrip("/")
    args.api_url = base if base.endswith("/chat/completions") else base + (
        "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    if not args.api_url.startswith(("http://", "https://")):
        parser.error("--api-base must be an HTTP(S) URL")
    return args


def main():
    args = parse_args()
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError("Empty system prompt")
    records = load_jsonl(args.input)
    selected, skipped = [], Counter()
    for index, row in enumerate(records):
        question, source = row.get(args.question_field), row.get(args.text_field)
        if question is None or question == "":
            skipped["missing_question"] += 1
        elif not isinstance(question, str) or not question.strip():
            skipped["invalid_question"] += 1
        elif not isinstance(source, str) or not source.strip():
            skipped["invalid_source"] += 1
        else:
            selected.append(index)
    eligible_count = len(selected)
    if args.limit is not None:
        selected = selected[:args.limit]
    print(json.dumps({"input_rows": len(records), "eligible_rows": eligible_count,
                      "selected_rows": len(selected), "skipped": dict(skipped)}, ensure_ascii=False))
    if not selected:
        raise ValueError("No eligible questions")
    requests = {i: build_request(records[i], args, system_prompt) for i in selected}
    if args.dry_run:
        print(json.dumps(requests[selected[0]], ensure_ascii=False, indent=2))
        return
    checkpoint_path = Path(str(args.output) + ".progress.jsonl")
    report_path = Path(str(args.output) + ".qa.json")
    if not args.resume and not args.overwrite and any(p.exists() for p in [args.output, checkpoint_path, report_path]):
        raise FileExistsError("Output/checkpoint/report exists; use --resume or --overwrite")
    if not args.resume and any(OUTPUT_FIELD in row or META_FIELD in row for row in records):
        raise ValueError("Input already contains paraphrases; use the original input with --resume")
    # A checkpoint can outlive output after interruption. Match the entire request, including source and prompt.
    previous = {}
    if args.resume:
        for path in [args.output, checkpoint_path]:
            if path.exists():
                for old in load_jsonl(path):
                    meta = old.get(META_FIELD) or {}
                    if meta.get("status") in ("ok", "needs_review"):
                        previous[meta["request_sha256"]] = old
    pending = []
    for i, body in requests.items():
        old = previous.get(digest(body))
        if old is None:
            records[i].pop(OUTPUT_FIELD, None)
            records[i].pop(META_FIELD, None)
            pending.append(i)
        else:
            records[i][OUTPUT_FIELD], records[i][META_FIELD] = old[OUTPUT_FIELD], old[META_FIELD]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Restored: {len(selected) - len(pending)}; pending: {len(pending)}", flush=True)
    with checkpoint_path.open("a" if args.resume else "w", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_one, requests[i], args): i
                       for i in pending}
            for count, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                body = requests[i]
                result = future.result()
                records[i][OUTPUT_FIELD] = result.pop("paraphrase")
                records[i][META_FIELD] = dict(
                    result, request_sha256=digest(body),
                    prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(), model=args.model,
                    seed=body["seed"],
                    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                    enable_thinking=False, original_question=records[i][args.question_field],
                )
                checkpoint.write(json.dumps({OUTPUT_FIELD: records[i][OUTPUT_FIELD],
                                             META_FIELD: records[i][META_FIELD]}, ensure_ascii=False) + "\n")
                checkpoint.flush()
                if count % 25 == 0 or count == len(pending):
                    print(f"Completed: {count}/{len(pending)}", flush=True)
    save_jsonl(args.output, records)
    report = qa_report(records, selected, skipped)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status_counts"].get("error"):
        raise RuntimeError("Some requests failed; partial results saved. Rerun with --resume")
    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

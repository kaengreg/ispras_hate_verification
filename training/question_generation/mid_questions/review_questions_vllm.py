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

from mid_question_review_prompts import (
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_PROMPT)


load_dotenv()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fin:
        if path.lower().endswith(".jsonl"):
            data = [json.loads(line) for line in fin if line.strip()]
        else:
            data = json.load(fin)
    if not isinstance(data, list) or any(not isinstance(obj, dict) for obj in data):
        raise ValueError("Input must contain a list of JSON objects or one object per JSONL line")
    return data


def save_json(path: str, data: List[Dict[str, Any]]):
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
        obj.pop("question_review", None)


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


def parse_review(raw: str):
    result = json.loads(raw)

    if not isinstance(result, dict) or set(result) != {"decision", "reason"}:
        raise ValueError('Expected exactly two JSON fields: "decision" and "reason"')
    
    decision = result["decision"]
    reason = result["reason"]

    if decision == "KEEP":
        if reason is not None:
            raise ValueError("reason must be null for KEEP")
        return {"decision": decision, "reason": None}
    
    if decision == "REWRITE":
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("REWRITE requires a non-empty reason")
        return {"decision": decision, "reason": " ".join(reason.split())}
    
    raise ValueError('decision must be "KEEP" or "REWRITE"')


def parse_rewrite(raw: str):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"question", "skip_reason"}:
        raise ValueError('Expected exactly two JSON fields: "question" and "skip_reason"')
    
    question = result["question"]
    skip_reason = result["skip_reason"]
    if question is None:
        if not isinstance(skip_reason, str) or not skip_reason.strip():
            raise ValueError("A skipped fragment requires a non-empty skip_reason")
        return {"question": None, "skip_reason": " ".join(skip_reason.split())}

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string or null")
    
    question = " ".join(question.split())
    if not question.endswith("?"):
        raise ValueError("The question must end with a question mark")
    if skip_reason is not None:
        raise ValueError("skip_reason must be null when a question is provided")
    return {"question": question, "skip_reason": None}


def call_with_retries(api_base, api_key, model, messages, temperature, max_tokens, parser, max_retries):
    base_messages = messages
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = request(api_base, api_key, model, messages, temperature, max_tokens)
            try:
                return parser(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                messages = base_messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Исправь формат ответа: {exc}. Верни только JSON согласно инструкции."},
                ]
                raise
        except Exception as exc:
            last_error = exc
        if attempt + 1 < max_retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}") from last_error


def review_and_rewrite(api_base, api_key, model, source_text, question, review_temperature,
                       rewrite_temperature, review_max_tokens, rewrite_max_tokens, max_retries,
                       max_rewrite_rounds):
    
    def review_question(candidate):
        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                source_text=source_text, question=candidate
            )},
        ]
        return call_with_retries(api_base, api_key, model, messages, review_temperature,
                                 review_max_tokens, parse_review, max_retries)

    review = review_question(question)
    
    if review["decision"] == "KEEP":
        return {"decision": "KEEP", "reason": None, "question": question, "rewrite_rounds": 0}

    initial_reason = review["reason"]
    post_review_reasons = []
    for rewrite_round in range(1, max_rewrite_rounds + 1):
        rewrite_messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": REWRITE_USER_PROMPT.format(answer=source_text)},
        ]
        rewritten = call_with_retries(api_base, api_key, model, rewrite_messages, rewrite_temperature,
                                      rewrite_max_tokens, parse_rewrite, max_retries)
        
        if rewritten["question"] is None:
            return {
                "decision": "SKIP",
                "reason": initial_reason,
                "skip_reason": rewritten["skip_reason"],
                "question": None,
                "original_question": question,
                "rewrite_rounds": rewrite_round,
                "post_review_reasons": post_review_reasons,
            }
        
        post_review = review_question(rewritten["question"])

        if post_review["decision"] == "KEEP":
            return {
                "decision": "REWRITE",
                "reason": initial_reason,
                "question": rewritten["question"],
                "original_question": question,
                "rewrite_rounds": rewrite_round,
                "post_review_reasons": post_review_reasons,
            }
        post_review_reasons.append(post_review["reason"])

    raise RuntimeError(
        f"Rewritten question failed post-review after {max_rewrite_rounds} round(s): "
        f"{post_review_reasons[-1]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--question-field", default="generated_question")
    parser.add_argument("--output-question-field", default="reviewed_question")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--review-temperature", type=float, default=0.0)
    parser.add_argument("--rewrite-temperature", type=float, default=0.2)
    parser.add_argument("--review-max-tokens", type=int, default=192)
    parser.add_argument("--rewrite-max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-rewrite-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_base = os.getenv("VLLM_BASE_URL")
    api_key = os.getenv("VLLM_API_KEY")
    if not api_base:
        raise RuntimeError("VLLM_BASE_URL is not set in .env")

    data = load_json(args.input)

    prompt_hash = hashlib.sha256((REVIEW_SYSTEM_PROMPT + "\n" + REVIEW_USER_PROMPT + "\n" + REWRITE_SYSTEM_PROMPT 
                                  + "\n" + REWRITE_USER_PROMPT).encode("utf-8")).hexdigest()
    metadata = {
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "review_temperature": args.review_temperature,
        "rewrite_temperature": args.rewrite_temperature,
        "review_max_tokens": args.review_max_tokens,
        "rewrite_max_tokens": args.rewrite_max_tokens,
        "max_rewrite_rounds": args.max_rewrite_rounds,
        "enable_thinking": False,
        "text_field": args.text_field,
        "question_field": args.question_field,
        "output_question_field": args.output_question_field,
    }
    run_signature = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    progress_path = args.output + ".review_progress.jsonl"

    completed = {}
    review_statuses = {}
    if args.resume and os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                if record.get("run_signature") == run_signature and record.get("status") in {"kept", "rewritten", "skipped"}:
                    completed[record["index"]] = record

    def hashes(obj):
        source = str(obj.get(args.text_field) or "").strip()
        question = str(obj.get(args.question_field) or "").strip()
        return (
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            hashlib.sha256(question.encode("utf-8")).hexdigest(),
        )

    def store_result(index, result, status):
        obj = data[index]
        obj[args.output_question_field] = result["question"]
        review_statuses[index] = status

    end = len(data) if args.limit is None else min(args.start + args.limit, len(data))
    selected = range(args.start, end)
    not_reviewed = sum(not str(data[index].get(args.question_field) or "").strip() for index in selected)
    tasks: List[Tuple[int, str, str]] = []
    for index in selected:
        obj = data[index]
        source_text = str(obj.get(args.text_field) or "").strip()
        question = str(obj.get(args.question_field) or "").strip()
        if not question:
            continue
        source_hash, question_hash = hashes(obj)
        previous = completed.get(index)
        if previous and previous.get("source_sha256") == source_hash and previous.get("question_sha256") == question_hash:
            store_result(index, previous["result"], previous["status"])
            continue
        tasks.append((index, source_text, question))

    async def run_parallel():
        if not tasks:
            return []
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        mode = "a" if args.resume else "w"
        errors = []
        loop = asyncio.get_running_loop()

        def sync_call(source_text, question):
            return review_and_rewrite(
                api_base, api_key, args.model, source_text, question,
                args.review_temperature, args.rewrite_temperature,
                args.review_max_tokens, args.rewrite_max_tokens, args.max_retries,
                args.max_rewrite_rounds,
            )

        async def one(index, source_text, question, executor):
            try:
                result = await loop.run_in_executor(executor, sync_call, source_text, question)
                return index, result, None
            except Exception as exc:
                return index, None, str(exc)

        with open(progress_path, mode, encoding="utf-8") as progress, ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            coroutines = [one(index, source_text, question, executor) for index, source_text, question in tasks]
            for future in tqdm(asyncio.as_completed(coroutines), total=len(coroutines), desc="Reviewing MID questions"):
                index, result, error = await future
                source_hash, question_hash = hashes(data[index])
                if error is not None:
                    errors.append((index, error))
                    fallback = {
                        "decision": None,
                        "reason": None,
                        "question": data[index].get(args.question_field),
                        "original_question": data[index].get(args.question_field),
                    }
                    store_result(index, fallback, "error")
                    record = {"index": index, "status": "error", "result": fallback, "error": error}
                    tqdm.write(f"[ERROR] row {index}: {error}")
                else:
                    status = {"KEEP": "kept", "REWRITE": "rewritten", "SKIP": "skipped"}[result["decision"]]
                    store_result(index, result, status)
                    record = {"index": index, "status": status, "result": result}
                    
                record.update({
                    "run_signature": run_signature,
                    "source_sha256": source_hash,
                    "question_sha256": question_hash,
                })
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.flush()
        return errors

    errors = asyncio.run(run_parallel())
    remove_row_metadata(data)
    save_json(args.output, data)
    with open(args.output + ".review.json", "w", encoding="utf-8") as fout:
        json.dump({
            **metadata,
            "input": args.input,
            "start": args.start,
            "limit": args.limit,
            "max_retries": args.max_retries,
            "run_signature": run_signature,
            "progress_path": progress_path,
            "review_system_prompt": REVIEW_SYSTEM_PROMPT,
            "review_user_prompt": REVIEW_USER_PROMPT,
            "rewrite_system_prompt": REWRITE_SYSTEM_PROMPT,
            "rewrite_user_prompt": REWRITE_USER_PROMPT,
        }, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    statuses = [review_statuses.get(index) for index in selected]
    kept = statuses.count("kept")
    rewritten = statuses.count("rewritten")
    skipped = statuses.count("skipped")
    print(f"Saved {len(data)} rows. Selected rows: kept={kept}, rewritten={rewritten}, skipped={skipped}, "
          f"not_reviewed={not_reviewed}, errors={len(errors)}")
    if errors:
        raise RuntimeError(f"Failed on {len(errors)} question(s); other results were saved to {args.output}")


if __name__ == "__main__":
    main()

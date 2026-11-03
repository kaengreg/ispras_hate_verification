import argparse
import asyncio
import hashlib
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from simpo_pair_selection_prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT, build_user_prompt


STANCE_LABELS = {"PRO", "NEU", "ANTI"}
SOURCE_CONSISTENCY = {"SUPPORTED", "PARTIAL", "CONTRADICTED", "NOT_APPLICABLE"}
COMPLETED_STATUSES = {"selected", "no_pair"}


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_num}: {exc}") from exc


def count_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")


def completion_url(api_base: str):
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def models_url(api_base: str):
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def row_id(row):
    return row.get("id", row.get("sentence_id", row.get("question_id")))


def question_text(row):
    value = row.get("question") or row.get("prompt") or row.get("generated_question")
    return str(value or "").strip()


def load_question_ids(path: Path):
    ids = set()
    for line_num, row in jsonl_rows(path):
        item_id = row_id(row)
        if item_id is None:
            raise ValueError(f"Missing id in {path} on line {line_num}")
        if item_id in ids:
            raise ValueError(f"Repeated id={item_id!r} in {path} on line {line_num}")
        ids.add(item_id)
    return ids


def answer_text(answer):
    value = answer.get("text") if isinstance(answer, dict) else answer
    return str(value or "").strip()


def paired_rows(positive_path: Path, negative_path: Path):
    positive_rows = jsonl_rows(positive_path)
    negative_rows = jsonl_rows(negative_path)
    for index, pair in enumerate(itertools.zip_longest(positive_rows, negative_rows)):
        positive_item, negative_item = pair
        if positive_item is None or negative_item is None:
            raise ValueError("Positive and negative input files contain different numbers of rows")

        positive_line, positive = positive_item
        negative_line, negative = negative_item
        if row_id(positive) is None or row_id(negative) is None:
            raise ValueError(f"Missing id in input rows {positive_line} and {negative_line}")
        if row_id(positive) != row_id(negative):
            raise ValueError(
                f"Input row mismatch: {positive_path}:{positive_line} has id={row_id(positive)!r}, "
                f"but {negative_path}:{negative_line} has id={row_id(negative)!r}"
            )
        if question_text(positive) != question_text(negative):
            raise ValueError(f"Question mismatch for id={row_id(positive)!r}")
        if not question_text(positive):
            raise ValueError(f"Missing question for id={row_id(positive)!r}")
        if str(positive.get("text") or "").strip() != str(negative.get("text") or "").strip():
            raise ValueError(f"Source text mismatch for id={row_id(positive)!r}")
        if not str(positive.get("text") or "").strip():
            raise ValueError(f"Missing source text for id={row_id(positive)!r}")
        yield index, positive, negative


def collect_candidates(positive, negative):
    candidates = []
    for origin, row in (("natural", positive), ("targeted_anti", negative)):
        answers = row.get("generated_answers") or []
        if not isinstance(answers, list):
            raise ValueError(f"generated_answers must be a list for id={row_id(row)!r}")
        for position, answer in enumerate(answers):
            text = answer_text(answer)
            if not text:
                continue
            candidate_id = answer.get("candidate_id", position) if isinstance(answer, dict) else position
            candidates.append(
                {
                    "origin": origin,
                    "candidate_id": candidate_id,
                    "text": text,
                    "previous_guard_label": answer.get("guard_label") if isinstance(answer, dict) else None,
                }
            )
    return candidates


def shuffle_candidates(candidates, item_id, seed):
    shuffle_seed = int(hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest(), 16)
    shuffled = list(candidates)
    random.Random(shuffle_seed).shuffle(shuffled)
    result = {}
    for index, candidate in enumerate(shuffled, start=1):
        display_id = f"A{index}"
        result[display_id] = candidate
    return result


def source_hash(positive, negative):
    payload = {
        "id": row_id(positive),
        "text": positive.get("text"),
        "question": question_text(positive),
        "positive_answers": [answer_text(answer) for answer in positive.get("generated_answers") or []],
        "negative_answers": [answer_text(answer) for answer in negative.get("generated_answers") or []],
    }
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def response_format(candidate_ids):
    assessment = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": candidate_ids},
            "stance": {"type": "string", "enum": sorted(STANCE_LABELS)},
            "answers_question": {"type": "boolean"},
            "source_consistency": {"type": "string", "enum": sorted(SOURCE_CONSISTENCY)},
            "language_ok": {"type": "boolean"},
            "meta_or_evasive": {"type": "boolean"},
            "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "acceptable": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "id", "stance", "answers_question", "source_consistency", "language_ok", "meta_or_evasive",
            "quality_score", "acceptable", "reason",
        ],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": assessment,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
            "pair_usable": {"type": "boolean"},
            "chosen_id": {"anyOf": [{"type": "string", "enum": candidate_ids}, {"type": "null"}]},
            "rejected_id": {"anyOf": [{"type": "string", "enum": candidate_ids}, {"type": "null"}]},
            "pair_reason": {"type": "string"},
        },
        "required": ["candidates", "pair_usable", "chosen_id", "rejected_id", "pair_reason"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "simpo_pair_selection", "schema": schema, "strict": True}}


def parse_reply(raw, candidate_map):
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(text)
    expected_fields = {"candidates", "pair_usable", "chosen_id", "rejected_id", "pair_reason"}
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError(f"Expected exactly these fields: {sorted(expected_fields)}")

    assessments = result["candidates"]
    if not isinstance(assessments, list) or len(assessments) != len(candidate_map):
        raise ValueError("The response must contain exactly one assessment for every candidate")

    assessment_by_id = {}
    required_assessment_fields = {
        "id", "stance", "answers_question", "source_consistency", "language_ok", "meta_or_evasive",
        "quality_score", "acceptable", "reason",
    }
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != required_assessment_fields:
            raise ValueError("Candidate assessment has an invalid set of fields")
        candidate_id = assessment["id"]
        if candidate_id not in candidate_map or candidate_id in assessment_by_id:
            raise ValueError(f"Unknown or repeated candidate id: {candidate_id!r}")
        if assessment["stance"] not in STANCE_LABELS:
            raise ValueError(f"Unknown stance: {assessment['stance']!r}")
        if assessment["source_consistency"] not in SOURCE_CONSISTENCY:
            raise ValueError(f"Unknown source_consistency: {assessment['source_consistency']!r}")
        if type(assessment["quality_score"]) is not int or not 1 <= assessment["quality_score"] <= 5:
            raise ValueError("quality_score must be an integer from 1 to 5")
        for field in ("answers_question", "language_ok", "meta_or_evasive", "acceptable"):
            if type(assessment[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        if not isinstance(assessment["reason"], str) or not assessment["reason"].strip():
            raise ValueError("Each candidate assessment must contain a non-empty reason")
        if assessment["acceptable"] and (
            not assessment["answers_question"]
            or assessment["source_consistency"] == "CONTRADICTED"
            or not assessment["language_ok"]
            or assessment["meta_or_evasive"]
            or assessment["quality_score"] < 3
        ):
            raise ValueError("acceptable=true conflicts with the candidate quality fields")
        assessment_by_id[candidate_id] = assessment

    if type(result["pair_usable"]) is not bool:
        raise ValueError("pair_usable must be boolean")
    if not isinstance(result["pair_reason"], str) or not result["pair_reason"].strip():
        raise ValueError("pair_reason must be a non-empty string")

    chosen_id = result["chosen_id"]
    rejected_id = result["rejected_id"]
    if not result["pair_usable"]:
        if chosen_id is not None or rejected_id is not None:
            raise ValueError("NO_PAIR requires null chosen_id and rejected_id")
        return result, None

    if chosen_id not in candidate_map or rejected_id not in candidate_map or chosen_id == rejected_id:
        raise ValueError("Selected candidate ids are invalid")
    if candidate_map[chosen_id]["text"].casefold() == candidate_map[rejected_id]["text"].casefold():
        raise ValueError("chosen and rejected must have different texts")
    chosen_assessment = assessment_by_id[chosen_id]
    rejected_assessment = assessment_by_id[rejected_id]
    if chosen_assessment["stance"] not in {"PRO", "NEU"} or not chosen_assessment["acceptable"]:
        raise ValueError("chosen must be an acceptable PRO or NEU answer")
    if rejected_assessment["stance"] != "ANTI" or not rejected_assessment["acceptable"]:
        raise ValueError("rejected must be an acceptable ANTI answer")

    def selected_candidate(display_id, assessment):
        candidate = candidate_map[display_id]
        return {
            "display_id": display_id,
            "origin": candidate["origin"],
            "candidate_id": candidate["candidate_id"],
            "text": candidate["text"],
            "previous_guard_label": candidate["previous_guard_label"],
            "judge_stance": assessment["stance"],
            "quality_score": assessment["quality_score"],
        }

    selection = {
        "chosen": selected_candidate(chosen_id, chosen_assessment),
        "rejected": selected_candidate(rejected_id, rejected_assessment),
    }
    return result, selection


async def validate_model(client, args):
    response = await client.get(models_url(args.api_base), headers=args.headers)
    response.raise_for_status()
    available = [item.get("id") for item in response.json().get("data", [])]
    if args.model not in available:
        raise ValueError(f"Model {args.model!r} is not served by vLLM. Available models: {available}")


async def judge_pair(client, semaphore, args, positive, negative):
    candidates = collect_candidates(positive, negative)
    if len(candidates) < 2:
        raise ValueError(f"At least two non-empty candidates are required for id={row_id(positive)!r}")
    candidate_map = shuffle_candidates(candidates, row_id(positive), args.seed)
    visible_candidates = [{"id": key, "text": value["text"]} for key, value in candidate_map.items()]
    user_prompt = build_user_prompt(
        source_text=str(positive.get("text") or "").strip(),
        question=question_text(positive),
        question_type=positive.get("question_type"),
        candidates=visible_candidates,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "response_format": response_format(list(candidate_map)),
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }

    last_error = None
    last_raw = None
    for attempt in range(args.max_retries + 1):
        try:
            request_body = dict(body)
            if last_raw:
                request_body["messages"] = messages + [
                    {"role": "assistant", "content": last_raw},
                    {
                        "role": "user",
                        "content": (
                            "Предыдущий ответ не прошел проверку формата или логических ограничений. "
                            "Повтори оценку и верни только корректный JSON. Если допустимой пары нет, "
                            "укажи pair_usable=false и null для chosen_id и rejected_id."
                        ),
                    },
                ]
            async with semaphore:
                response = await client.post(args.url, headers=args.headers, json=request_body)
            response.raise_for_status()
            last_raw = response.json()["choices"][0]["message"]["content"]
            result, selection = parse_reply(last_raw, candidate_map)
            return result, selection, last_raw
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                await asyncio.sleep(min(30.0, args.retry_delay * (2 ** attempt)))
    raise RuntimeError(f"Judge failed after retries: {last_error}; last response={str(last_raw)[:1000]!r}")


def load_completed(progress_path, run_signature):
    completed = {}
    if not progress_path.exists():
        return completed
    for _, entry in jsonl_rows(progress_path):
        if entry.get("run_signature") == run_signature:
            completed[entry["source_index"]] = entry
    return completed


def is_completed(entry, current_hash):
    return bool(entry and entry.get("source_sha256") == current_hash and entry.get("status") in COMPLETED_STATUSES)


async def run(args, completed, progress_path, run_signature):
    stop = args.total_rows if args.limit is None else min(args.total_rows, args.start + args.limit)
    pending_count = 0
    for index, positive, negative in paired_rows(args.positive_input, args.negative_input):
        if index < args.start or index >= stop:
            continue
        if row_id(positive) in args.excluded_ids:
            continue
        if not is_completed(completed.get(index), source_hash(positive, negative)):
            pending_count += 1

    print(f"Rows to judge: {pending_count}")
    if not pending_count:
        return 0

    semaphore = asyncio.Semaphore(args.workers)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    mode = "a" if args.resume else "w"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await validate_model(client, args)
        with progress_path.open(mode, encoding="utf-8") as checkpoint:
            progress = tqdm(total=pending_count, desc="Selecting pairs", unit="question")
            batch = []

            async def process_one(index, positive, negative, current_hash):
                try:
                    result, selection, raw = await judge_pair(client, semaphore, args, positive, negative)
                    status = "selected" if result["pair_usable"] else "no_pair"
                    entry = {
                        "source_index": index,
                        "id": row_id(positive),
                        "source_sha256": current_hash,
                        "run_signature": run_signature,
                        "status": status,
                        "result": result,
                        "selection": selection,
                    }
                    if args.save_raw:
                        entry["raw"] = raw
                    return entry
                except Exception as exc:
                    return {
                        "source_index": index,
                        "id": row_id(positive),
                        "source_sha256": current_hash,
                        "run_signature": run_signature,
                        "status": "error",
                        "error": str(exc),
                    }

            async def flush_batch(items):
                nonlocal errors
                results = await asyncio.gather(*items)
                for entry in results:
                    checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    completed[entry["source_index"]] = entry
                    errors += entry["status"] == "error"
                    progress.update(1)
                checkpoint.flush()

            try:
                for index, positive, negative in paired_rows(args.positive_input, args.negative_input):
                    if index < args.start or index >= stop:
                        continue
                    if row_id(positive) in args.excluded_ids:
                        continue
                    current_hash = source_hash(positive, negative)
                    if is_completed(completed.get(index), current_hash):
                        continue
                    batch.append(asyncio.create_task(process_one(index, positive, negative, current_hash)))
                    if len(batch) >= args.workers * 4:
                        await flush_batch(batch)
                        batch = []
                if batch:
                    await flush_batch(batch)
            finally:
                for task in batch:
                    if not task.done():
                        task.cancel()
                progress.close()
    return errors


def build_output(args, completed):
    stop = args.total_rows if args.limit is None else min(args.total_rows, args.start + args.limit)
    stats = Counter()
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as target:
        for index, positive, negative in paired_rows(args.positive_input, args.negative_input):
            if index < args.start or index >= stop:
                continue
            if row_id(positive) in args.excluded_ids:
                stats["excluded_holdout"] += 1
                continue
            entry = completed.get(index)
            status = entry.get("status") if entry else "missing"
            stats[status] += 1
            if args.only_selected and status != "selected":
                continue

            selection = entry.get("selection") if entry else None
            chosen = selection.get("chosen") if selection else None
            rejected = selection.get("rejected") if selection else None
            output = {
                "id": row_id(positive),
                "text": positive.get("text"),
                "prompt": question_text(positive),
                "source_label": positive.get("source_label"),
                "question_type": positive.get("question_type"),
                "chosen": chosen.get("text") if chosen else None,
                "rejected": rejected.get("text") if rejected else None,
                "pair_status": status,
                "selection_metadata": {
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "chosen_origin": chosen.get("origin") if chosen else None,
                    "chosen_candidate_id": chosen.get("candidate_id") if chosen else None,
                    "chosen_stance": chosen.get("judge_stance") if chosen else None,
                    "chosen_quality_score": chosen.get("quality_score") if chosen else None,
                    "rejected_origin": rejected.get("origin") if rejected else None,
                    "rejected_candidate_id": rejected.get("candidate_id") if rejected else None,
                    "rejected_stance": rejected.get("judge_stance") if rejected else None,
                    "rejected_quality_score": rejected.get("quality_score") if rejected else None,
                },
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
    tmp_path.replace(args.output)
    return stats


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Select one high-quality SimPO pair per question with an LLM judge")
    parser.add_argument("--positive-input", type=Path, required=True, help="Natural/positive candidate JSONL")
    parser.add_argument("--negative-input", type=Path, required=True, help="Targeted anti candidate JSONL")
    parser.add_argument("--exclude-questions", type=Path,
                        help="Fixed test-question JSONL; matching ids are not sent to the judge or written to output")
    parser.add_argument("--output", type=Path, required=True, help="Compact selection output JSONL")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "rcc"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible candidate order")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Store raw model responses in the progress file")
    parser.add_argument("--only-selected", action="store_true", help="Write only usable pairs to the compact output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.positive_input, args.negative_input):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.exclude_questions is not None and not args.exclude_questions.is_file():
        raise FileNotFoundError(args.exclude_questions)
    args.excluded_ids = load_question_ids(args.exclude_questions) if args.exclude_questions else set()
    if args.output.resolve() in {args.positive_input.resolve(), args.negative_input.resolve()}:
        raise ValueError("Output path must differ from both input paths")
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--start must be non-negative and --limit must be positive")

    positive_count = count_jsonl_rows(args.positive_input)
    negative_count = count_jsonl_rows(args.negative_input)
    if positive_count != negative_count:
        raise ValueError(f"Input row counts differ: positive={positive_count}, negative={negative_count}")
    args.total_rows = positive_count
    args.url = completion_url(args.api_base)
    args.headers = {"Content-Type": "application/json"}
    if args.api_key:
        args.headers["Authorization"] = f"Bearer {args.api_key}"

    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode("utf-8")).hexdigest()
    config = {
        "positive_input": str(args.positive_input),
        "negative_input": str(args.negative_input),
        "exclude_questions": str(args.exclude_questions) if args.exclude_questions else None,
        "exclude_questions_sha256": (
            hashlib.sha256(args.exclude_questions.read_bytes()).hexdigest() if args.exclude_questions else None
        ),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "start": args.start,
        "limit": args.limit,
    }
    run_signature = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    config["run_signature"] = run_signature

    progress_path = Path(str(args.output) + ".selection_progress.jsonl")
    config_path = Path(str(args.output) + ".selection.json")
    config["progress_path"] = str(progress_path)
    if (args.output.exists() or progress_path.exists()) and not args.resume and not args.overwrite:
        raise FileExistsError("Output or progress file already exists; use --resume or --overwrite")
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot resume without config file: {config_path}")
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config.get("run_signature") != run_signature:
            raise ValueError("Cannot resume: model, prompt, inputs or selection parameters have changed")

    save_json(config_path, config)
    completed = load_completed(progress_path, run_signature) if args.resume else {}
    errors = asyncio.run(run(args, completed, progress_path, run_signature))
    stats = build_output(args, completed)
    print(f"Saved output to {args.output}. Statuses={dict(stats)}")
    if errors:
        raise RuntimeError(f"Selection failed for {errors} row(s). Run again with --resume to retry them.")


if __name__ == "__main__":
    main()

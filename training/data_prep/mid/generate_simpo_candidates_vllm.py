import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm


QUESTION_FIELDS = ("generated_question", "question", "prompt")
ANSWER_FIELDS = ("chosen", "answer")
DEFAULT_SYSTEM_PROMPT = (
    "Ответь на вопрос по-русски, прямо и по существу. "
    "Сформулируй самостоятельный, связный и грамматически корректный ответ из одного или двух предложений. "
    "Если вопрос предполагает оценку, вырази позицию ясно и кратко обоснуй её. "
    "Не пытайся специально соглашаться с предполагаемой позицией пользователя или давать нейтральный ответ. "
    "Не добавляй лишние имена, должности, даты, числа, цитаты, события и другие подробности, без которых можно ответить. "
    "Не используй списки, заголовки, дисклеймеры и мета-комментарии."
)


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        first = ""
        while True:
            char = fin.read(1)
            if not char or not char.isspace():
                first = char
                break
        fin.seek(0)
        if first == "[":
            data = json.load(fin)
        else:
            data = []
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc

    return data


def save_jsonl(path: Path, records: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")


def nonempty_string(value: Any):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def is_degenerate_answer(text: str):
    words = re.findall(r"[\wёЁ-]+", text.casefold())
    repeated = 1
    for previous, current in zip(words, words[1:]):
        repeated = repeated + 1 if current == previous else 1
        if repeated >= 4:
            return True
    if len(words) >= 20 and max(Counter(words).values(), default=0) / len(words) >= 0.35:
        return True
    for size in (2, 3, 4):
        ngrams = Counter(tuple(words[index:index + size]) for index in range(len(words) - size + 1))
        if max(ngrams.values(), default=0) >= 5:
            return True
    return False


def value_from_messages(record: dict[str, Any], role: str):
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            value = nonempty_string(message.get("content"))
            if value:
                return value
    return None


def get_value(record: dict[str, Any], explicit_field: str | None, candidates: tuple[str, ...], role: str):
    if explicit_field:
        value = nonempty_string(record.get(explicit_field))
        if value:
            return value
        raise ValueError(f"field {explicit_field!r} is empty or absent")

    for field in candidates:
        value = nonempty_string(record.get(field))
        if value:
            return value

    value = value_from_messages(record, role)
    if value:
        return value
    raise ValueError(f"could not find a non-empty value in {candidates} or messages[{role!r}]")


def get_optional_value(record: dict[str, Any], explicit_field: str | None, candidates: tuple[str, ...], role: str):
    if explicit_field:
        return get_value(record, explicit_field, candidates, role)
    for field in candidates:
        value = nonempty_string(record.get(field))
        if value:
            return value
    return value_from_messages(record, role)


def completion_url(api_base: str):
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def answer_texts(record: dict[str, Any]):
    answers = record.get("generated_answers")
    if not isinstance(answers, list):
        return []
    result = []
    for answer in answers:
        value = answer.get("text") if isinstance(answer, dict) else answer
        value = nonempty_string(value)
        if value and value not in result:
            result.append(value)
    return result


def apply_checkpoint(path: Path, records: list[dict[str, Any]]):
    restored = 0
    if not path.exists():
        return restored
    for checkpoint_row in load_records(path):
        index = checkpoint_row.get("source_index")
        answers = checkpoint_row.get("generated_answers")
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError(f"Invalid source_index in checkpoint {path}: {index!r}")
        if not isinstance(answers, list):
            raise ValueError(f"Invalid generated_answers for row {index} in checkpoint {path}")
        records[index]["generated_answers"] = answers
        restored += 1
    return restored


async def generate_answers(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, args: argparse.Namespace, 
                           question: str, existing: list[str], source_index: int):
    answers = list(existing)
    last_error = None

    for attempt in range(args.max_retries + 1):
        missing = args.num_candidates - len(answers)
        if missing <= 0:
            return answers[: args.num_candidates]

        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": question})

        body: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "n": missing,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed + source_index * 1009 + attempt,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        if args.top_k >= 0:
            body["top_k"] = args.top_k

        try:
            async with semaphore:
                response = await client.post(args.url, headers=args.headers, json=body)
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                raise ValueError(f"vLLM returned no choices: {payload}")

            before = len(answers)
            for choice in choices:
                text = nonempty_string((choice.get("message") or {}).get("content"))
                if text and text not in answers and not is_degenerate_answer(text):
                    answers.append(text)
            if len(answers) == before:
                raise ValueError("vLLM returned only empty or duplicate answers")
        except Exception as exc:
            last_error = exc

        if len(answers) >= args.num_candidates:
            return answers[: args.num_candidates]
        if attempt < args.max_retries:
            await asyncio.sleep(min(30.0, args.retry_delay * (2**attempt)))

    raise RuntimeError(
        f"got {len(answers)}/{args.num_candidates} unique answers after retries; last error: {last_error}"
    )


async def run_generation(args: argparse.Namespace, records: list[dict[str, Any]]):
    pending = []
    stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    for index in range(args.start, stop):
        if len(answer_texts(records[index])) < args.num_candidates:
            pending.append(index)

    print(f"Rows to generate: {len(pending)}")
    if not pending:
        return []

    semaphore = asyncio.Semaphore(args.workers)
    timeout = httpx.Timeout(args.timeout)
    errors: list[tuple[int, str]] = []

    # Keep an append-only checkpoint. Rewriting a 100k+ row output after every
    # small batch would produce excessive I/O.
    checkpoint_mode = "a" if args.resume else "w"
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.open(checkpoint_mode, encoding="utf-8")
    progress = tqdm(total=len(pending), desc="Generating", unit="question")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def one(index: int) -> tuple[int, list[str] | None, str | None]:
                try:
                    answers = await generate_answers(
                        client=client,
                        semaphore=semaphore,
                        args=args,
                        question=records[index]["prompt"],
                        existing=answer_texts(records[index]),
                        source_index=index,
                    )
                    return index, answers, None
                except Exception as exc:
                    return index, None, str(exc)

            chunk_size = max(args.workers, args.workers * 4)
            completed_since_flush = 0
            for offset in range(0, len(pending), chunk_size):
                tasks = [asyncio.create_task(one(index)) for index in pending[offset : offset + chunk_size]]
                for task in asyncio.as_completed(tasks):
                    index, answers, error = await task
                    progress.update(1)
                    if error:
                        errors.append((index, error))
                        tqdm.write(f"[ERROR] row {index}: {error}", file=sys.stderr)
                        continue

                    records[index]["generated_answers"] = [
                        {"candidate_id": candidate_id, "text": text}
                        for candidate_id, text in enumerate(answers or [])
                    ]
                    checkpoint_record = {
                        "source_index": index,
                        "prompt": records[index]["prompt"],
                        "generated_answers": records[index]["generated_answers"],
                    }
                    if records[index].get("chosen") is not None:
                        checkpoint_record["chosen"] = records[index]["chosen"]
                    checkpoint.write(json.dumps(checkpoint_record, ensure_ascii=False) + "\n")
                    completed_since_flush += 1
                    if completed_since_flush >= args.save_every:
                        checkpoint.flush()
                        completed_since_flush = 0
            checkpoint.flush()
    finally:
        progress.close()
        checkpoint.close()

    return errors


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input JSON or JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "rcc"))
    parser.add_argument("--question-field", default=None, help="Default: auto-detect generated_question/question/prompt")
    parser.add_argument("--answer-field", default=None)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue an existing output file")
    mode.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()

def prepare_records(args: argparse.Namespace, source: list[dict[str, Any]]):
    prepared = []
    for index, source_record in enumerate(source):
        try:
            prompt = get_value(source_record, args.question_field, QUESTION_FIELDS, "user")
            chosen = get_optional_value(source_record, args.answer_field, ANSWER_FIELDS, "assistant")
        except ValueError as exc:
            raise ValueError(f"input row {index}: {exc}") from exc
        record = dict(source_record)
        record["prompt"] = prompt
        if chosen is not None:
            record["chosen"] = chosen
        prepared.append(record)
    return prepared


def main():
    args = parse_args()
    args.url = completion_url(args.api_base)
    args.checkpoint = Path(str(args.output) + ".progress.jsonl")
    args.config = Path(str(args.output) + ".generation.json")

    args.headers = {"Content-Type": "application/json"}
    if args.api_key:
        args.headers["Authorization"] = f"Bearer {args.api_key}"

    source = load_records(args.input)
    records = prepare_records(args, source)
    print(f"Loaded {len(records)} rows from {args.input}")
    print(f"Endpoint: {args.url}")

    if args.output.exists():
        if args.resume:
            previous = load_records(args.output)
            if len(previous) != len(records):
                raise ValueError(
                    f"Cannot resume: input has {len(records)} rows, output has {len(previous)} rows")
            for index, (current, old) in enumerate(zip(records, previous)):
                if current["prompt"] != old.get("prompt") or current.get("chosen") != old.get("chosen"):
                    raise ValueError(f"Cannot resume: prompt/chosen mismatch at row {index}")
                if answer_texts(old):
                    current["generated_answers"] = old["generated_answers"]
            print(f"Resuming from {args.output}")
        elif not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; use --resume or --overwrite")

    if args.checkpoint.exists():
        if args.resume:
            restored = apply_checkpoint(args.checkpoint, records)
            print(f"Restored {restored} checkpoint row(s) from {args.checkpoint}")
        elif not args.overwrite:
            raise FileExistsError(f"{args.checkpoint} already exists; use --resume or --overwrite")

    config = {
        "input": str(args.input),
        "model": args.model,
        "system_prompt": args.system_prompt,
        "prompt_sha256": hashlib.sha256(args.system_prompt.encode("utf-8")).hexdigest(),
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
    }
    run_signature = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    config["run_signature"] = run_signature

    if args.resume and args.config.exists():
        previous_config = json.loads(args.config.read_text(encoding="utf-8"))
        if previous_config.get("run_signature") != run_signature:
            raise ValueError("Cannot resume: generation parameters differ from the original run")
        
    save_json(args.config, config)

    errors = asyncio.run(run_generation(args, records))
    save_jsonl(args.output, records)
    args.checkpoint.unlink(missing_ok=True)
    generated = sum(len(answer_texts(record)) >= args.num_candidates for record in records)
    print(f"Saved {len(records)} rows to {args.output}; complete rows: {generated}")

    if errors:
        preview = "; ".join(f"row {index}: {error}" for index, error in errors[:5])
        raise RuntimeError(
            f"Generation failed for {len(errors)} row(s). Successful rows were saved. "
            f"Run again with --resume. {preview}"
        )


if __name__ == "__main__":
    main()

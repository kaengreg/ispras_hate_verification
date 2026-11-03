import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import CRITERION, build_messages, build_response_format, parse_model_reply


LABELS = {"PRO", "NEU", "ANTI"}
VERDICT_PATTERN = re.compile(r'''["']?verdict["']?\s*[:=]\s*["']?(PRO|NEU|ANTI)\b''', re.IGNORECASE)
DEFAULT_INPUT = PROJECT_ROOT/"data/mid/simpo_candidates/simpo_candidates_limited_filtered.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT/"data/mid/simpo_candidates/simpo_candidates_limited_filtered_guard.jsonl"
QUESTION_CONTEXT_INSTRUCTION = """Вопрос приведен только как контекст для интерпретации ответа.
Определи позицию, которую выражает или поддерживает сам ответ, а не позицию, упомянутую в вопросе.
Если ответ нейтрально передает чужое заявление, обвинение или оценку и не поддерживает их собственной
формулировкой, считай это нейтральным сообщением. Не оценивай фактологическую правильность ответа.

Вопрос:
{question}

Ответ:
{answer}"""


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield line_num, row


def count_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        return sum(1 for line in fin if line.strip())


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


def answer_text(answer):
    value = answer.get("text") if isinstance(answer, dict) else answer
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def row_identity(row):
    answers = row.get("generated_answers")
    texts = tuple(answer_text(answer) for answer in answers) if isinstance(answers, list) else ()
    row_id = row.get("question_id", row.get("sentence_id", row.get("id")))
    return row_id, row.get("prompt") or row.get("question") or row.get("generated_question"), texts


def question_text(row):
    value = row.get("prompt") or row.get("question") or row.get("generated_question")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing non-empty prompt/question/generated_question")
    return value.strip()


def validate_resume_prefix(input_path: Path, output_path: Path):
    input_rows = jsonl_rows(input_path)
    completed = 0
    for output_line, output_row in jsonl_rows(output_path):
        try:
            input_line, input_row = next(input_rows)
        except StopIteration:
            raise ValueError(f"Resume output {output_path} has more rows than input {input_path}")
        if row_identity(input_row) != row_identity(output_row):
            raise ValueError(
                f"Cannot resume: input line {input_line} does not match output line {output_line} "
                "by question_id/sentence_id/id, question and candidate texts"
            )
        completed += 1
    return completed, input_rows


async def validate_model(client, args):
    response = await client.get(models_url(args.api_base), headers=args.headers)
    response.raise_for_status()
    available = [item.get("id") for item in response.json().get("data", [])]
    if args.model not in available:
        raise ValueError(f"Model {args.model!r} is not served by vLLM. Available models: {available}")


def parse_guard_reply(raw):
    try:
        parsed = parse_model_reply(raw)
        label = str(parsed.get("verdict", "")).strip().upper()
        if label in LABELS:
            return label, str(parsed.get("reason", "")).strip()
    except ValueError:
        pass

    match = VERDICT_PATTERN.search(str(raw))
    if match:
        return match.group(1).upper(), ""
    raise ValueError(f"Unable to extract PRO/NEU/ANTI verdict from model reply: {str(raw)[:1000]!r}")


async def classify_text(client, semaphore, args, text, question=None):
    cfg = CRITERION[args.criterion]
    guard_input = QUESTION_CONTEXT_INSTRUCTION.format(question=question, answer=text) if args.with_question_context else text
    messages = build_messages(cfg, guard_input)
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "response_format": build_response_format(args.criterion),
    }
    if not args.enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

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
                            "Предыдущий ответ имеет некорректный формат. Верни только валидный JSON с ключами "
                            '"verdict" и "reason". verdict должен быть PRO, NEU или ANTI, reason — одно короткое предложение.'
                        ),
                    },
                ]
            async with semaphore:
                response = await client.post(args.url, headers=args.headers, json=request_body)
            response.raise_for_status()
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
            last_raw = raw
            label, reason = parse_guard_reply(raw)

            result = {"guard_label": label}
            if args.save_reason:
                result["guard_reason"] = reason
            if args.save_raw:
                result["guard_raw"] = raw
            return result
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                await asyncio.sleep(min(30.0, args.retry_delay * (2**attempt)))

    raw_preview = str(last_raw)[:1000] if last_raw is not None else None
    raise RuntimeError(
        f"Guard request failed after {args.max_retries + 1} attempt(s): {last_error}; "
        f"last raw response: {raw_preview!r}"
    )


async def label_row(client, semaphore, args, line_num, row):
    answers = row.get("generated_answers")
    if answers is None:
        row["guard_evaluation"] = {"model": args.model, "criterion": args.criterion, "status": "no_candidates"}
        return row
    if not isinstance(answers, list):
        raise ValueError(f"Input line {line_num}: generated_answers must be a list or null")
    if not answers:
        row["guard_evaluation"] = {"model": args.model, "criterion": args.criterion, "status": "no_candidates"}
        return row

    question = None
    if args.with_question_context:
        try:
            question = question_text(row)
        except ValueError as exc:
            raise ValueError(f"Input line {line_num}: {exc}") from exc

    tasks = []
    indices = []
    for index, answer in enumerate(answers):
        text = answer_text(answer)
        if text is None:
            raise ValueError(f"Input line {line_num}, candidate {index}: missing non-empty text")
        tasks.append(classify_text(client, semaphore, args, text, question))
        indices.append(index)

    evaluations = await asyncio.gather(*tasks, return_exceptions=True)

    labeled_answers = []
    for index, evaluation in zip(indices, evaluations):
        if isinstance(evaluation, BaseException):
            candidate_id = answers[index].get("candidate_id", index) if isinstance(answers[index], dict) else index
            raise RuntimeError(f"Input line {line_num}, candidate {candidate_id}: {evaluation}")

        answer = answers[index]
        labeled = dict(answer) if isinstance(answer, dict) else {"text": answer}
        labeled.update(evaluation)
        labeled_answers.append(labeled)

    row["generated_answers"] = labeled_answers
    row["guard_evaluation"] = {"model": args.model, "criterion": args.criterion, "status": "complete"}
    return row


async def process(args: argparse.Namespace) -> None:
    args.url = completion_url(args.api_base)
    args.headers = {"Content-Type": "application/json"}
    if args.api_key:
        args.headers["Authorization"] = f"Bearer {args.api_key}"

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    total = count_jsonl_rows(args.input)

    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; use --resume or --overwrite")

    if args.resume and args.output.exists():
        completed, remaining_rows = validate_resume_prefix(args.input, args.output)
        output_mode = "a"
        print(f"Validated {completed} completed row(s); resuming from the next input row")
    else:
        completed = 0
        remaining_rows = jsonl_rows(args.input)
        output_mode = "w"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    semaphore = asyncio.Semaphore(args.workers)
    processed_this_run = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await validate_model(client, args)
        with args.output.open(output_mode, encoding="utf-8") as fout:
            progress = tqdm(total=total, desc="Labeling", unit="question", initial=completed)
            pending = []
            try:
                for _ in range(args.workers):
                    try:
                        line_num, row = next(remaining_rows)
                    except StopIteration:
                        break
                    pending.append(asyncio.create_task(label_row(client, semaphore, args, line_num, row)))

                while pending:
                    task = pending.pop(0)
                    row = await task
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    processed_this_run += 1
                    progress.update(1)

                    try:
                        line_num, next_row = next(remaining_rows)
                    except StopIteration:
                        continue
                    pending.append(asyncio.create_task(label_row(client, semaphore, args, line_num, next_row)))
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                progress.close()

    print(f"Wrote {processed_this_run} new row(s) to {args.output}")


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--criterion", default="anti_russia", choices=("anti_russia",))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--save-reason", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--with-question-context", action="store_true", help="Classify each generated answer together with its question")
    parser.add_argument("--enable-thinking", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    asyncio.run(process(parse_args()))

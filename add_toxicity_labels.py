import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()


TOX_LABELS = ("TOX", "NEU")

SYSTEM_PROMPT = (
    "Ты - модель для классификации токсичности высказывания. "
    "Есть два класса: TOX и NEU. "
    "TOX означает токсичное, оскорбительное, агрессивное или уничижительное высказывание. "
    "NEU означает нетоксичное высказывание. "
    "Необходимо вернуть ТОЛЬКО одну метку: TOX или NEU."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {exc}") from exc
    return data


def save_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def build_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    text = example.get("text", "").strip()
    target = example.get("target", "").strip()

    user_content = (
        "Определи, является ли высказывание токсичным.\n\n"
        f"**Высказывание:**\n\n{text}\n\n"
        f"**Цель/объект упоминания:**\n\n{target}\n\n"
        "Ответ должен быть ОДНИМ словом: 'TOX' или 'NEU'."
    )

    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}]


def normalize_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None

    text = str(raw).strip().upper()
    if text in TOX_LABELS:
        return text

    match = re.search(r"\b(TOX|NEU)\b", text)
    if match:
        return match.group(1)

    return None


async def request_label(client: httpx.AsyncClient, api_base: str, api_key: str, model: str, example: Dict[str, Any], 
                         temperature: float, max_tokens: int, max_retries: int):
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": build_messages(example),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"].strip()
            label = normalize_label(raw)
            if label is None:
                raise ValueError(f"Could not parse toxicity label from model reply: {raw!r}")
            return label, raw
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"Failed to classify id={example.get('sentence_id')}: {last_error}") from last_error


async def label_data(args, data):
    output_data = [dict(item) for item in data]
    pending = [idx for idx, item in enumerate(output_data) if not (args.skip_existing and normalize_label(item.get("toxicity_label")) in TOX_LABELS)]

    if not pending:
        print("No rows to classify.")
        return output_data

    print(f"Rows to classify: {len(pending)}")

    completed = 0
    started_at = time.time()
    semaphore = asyncio.Semaphore(args.workers)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def classify_one(idx: int) -> Tuple[int, str, str]:
            async with semaphore:
                label, raw = await request_label(
                    client=client,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    example=output_data[idx],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                return idx, label, raw

        tasks = [asyncio.create_task(classify_one(idx)) for idx in pending]
        progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Classifying toxicity", unit="row")
        for task in progress:
            idx, label, raw = await task
            output_data[idx]["toxicity_label"] = label
            if args.keep_raw_reply:
                output_data[idx]["toxicity_raw_reply"] = raw

            completed += 1
            elapsed = time.time() - started_at
            rate = completed / elapsed if elapsed > 0 else 0.0
            progress.set_postfix(rate=f"{rate:.2f} rows/sec")

            if completed % args.save_every == 0 or completed == len(pending):
                save_jsonl(args.output, output_data)

    return output_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/countries_classification/train_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/countries_classification/train_raw_with_toxicity.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:6266"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw-reply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Parsed from env: \n VLLM BASE: {args.api_base} \n VLLM KEY: {args.api_key}")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    input_data = load_jsonl(args.input)
    print(f"Loaded {len(input_data)} rows from {args.input}")

    if args.resume and args.output.exists():
        output_data = load_jsonl(args.output)
        if len(output_data) != len(input_data):
            raise ValueError(
                f"Cannot resume: input has {len(input_data)} rows, "
                f"but output has {len(output_data)} rows."
            )
        input_data = output_data
        args.skip_existing = True
        print(f"Resuming from {args.output}")

    output_data = asyncio.run(label_data(args, input_data))
    save_jsonl(args.output, output_data)
    print(f"Done. Saved {len(output_data)} rows to {args.output}")


if __name__ == "__main__":
    main()

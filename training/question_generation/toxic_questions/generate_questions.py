import argparse
import asyncio
import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
from questions_prompts import SYSTEM_PROMPT, USER_PROMPT
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def load_json(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fin:
        first_char = fin.read(1)
        fin.seek(0)
        if first_char == "[":
            loaded = json.load(fin)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON array in {path}")
            return loaded

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            data.append(obj)
    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as fout:
            for obj in data:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return

    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def request(api_base, api_key, model, messages, temp, max_tokens):
    url = f"{api_base.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }

    req = requests.post(url, headers=headers, json=payload, timeout=60)
    if req.status_code >= 400:
        raise RuntimeError(f"{req.status_code}: {req.text}")
    
    return req.json()['choices'][0]['message']['content'].strip()


def normalize_question(question: str) -> str:
    q = question.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("«") and q.endswith("»")):
        q = q[1:-1].strip()
    else:
        q = q.strip('"').strip("«").strip("»").strip()

    q = " ".join(q.split())
    if q and not q.endswith("?"):
        q = q.rstrip(".!…") + "?"
    return q


def generate_question(api_base, api_key,model, answer_text, target, stance, toxicity_label, temp, max_tokens, max_retries):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                answer=answer_text,
                target=target,
                stance=stance,
                toxicity_label=toxicity_label,
            )
        }
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            raw = request(api_base, api_key, model, messages, temp, max_tokens)
            question = normalize_question(raw)
            if question:
                return question
        except Exception as exc:
            last_error = exc
        time.sleep(2 * (attempt + 1))

    if last_error:
        raise RuntimeError("Failed to generate question after retries") from last_error
    raise RuntimeError("Failed to generate a non-empty question after retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--stance-field", default="stance")
    parser.add_argument("--toxicity-field", default="toxicity_label")
    parser.add_argument("--question-field", default="generated_question")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    api_base = os.getenv("VLLM_BASE_URL")
    api_key = os.getenv("VLLM_API_KEY")

    if not api_base:
        raise RuntimeError("VLLM_BASE_URL is not set in .env")
    if not api_key:
        raise RuntimeError("VLLM_API_KEY is not set in .env")

    data = load_json(args.input)

    tasks = []
    end = len(data) if args.limit is None else args.start + args.limit
    selected = range(args.start, min(end, len(data)))

    for oi in selected:
        obj = data[oi]
        if args.skip_existing and obj.get(args.question_field):
            continue

        text = str(obj.get(args.text_field) or "").strip()
        if not text:
            continue

        target = str(obj.get(args.target_field) or "").strip()
        stance = str(obj.get(args.stance_field) or "").strip()
        toxicity_label = str(obj.get(args.toxicity_field) or "").strip()
        tasks.append((oi, text, target, stance, toxicity_label))

    async def run_parallel():
        if not tasks:
            return []

        loop = asyncio.get_running_loop()
        results = []

        def sync_call(args_for_call):
            text, target, stance, toxicity_label = args_for_call
            return generate_question(
                api_base=api_base,
                api_key=api_key,
                model=args.model,
                answer_text=text,
                target=target,
                stance=stance,
                toxicity_label=toxicity_label,
                temp=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )

        async def one(oi, text, target, stance, toxicity_label, executor):
            question = await loop.run_in_executor(executor, sync_call, (text, target, stance, toxicity_label))
            return oi, question

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            coros = [one(oi, text, target, stance, toxicity_label, executor) for oi, text, target, stance, toxicity_label in tasks]
            for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Generating toxic questions"):
                results.append(await fut)
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)

        return results

    results = asyncio.run(run_parallel())

    for oi, question in results:
        data[oi][args.question_field] = question

    save_json(args.output, data)
    print(f"Done. Generated questions: {len(results)}")


if __name__ == "__main__":
    main()

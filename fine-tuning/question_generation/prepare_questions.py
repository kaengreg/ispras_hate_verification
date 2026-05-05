import argparse
import asyncio
import json
import os
import time
from urllib.parse import unquote, urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from questions_prompts import SYSTEM_PROMPT, USER_PROMPT

load_dotenv()

def load_json(path):
    data = []
    with open(path, "r", encoding='utf-8') as fin:
        first_char = fin.read(1)
        fin.seek(0)
        if first_char == '[':
            data = json.load(fin)
        else:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    return data 

def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as fout:
            for obj in data:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
    

def request(api_base, api_key, model, messages, temp=0.3, max_tokens=128):
    url = f"{api_base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    request = {"model": model,
               "messages": messages,
               "temperature": temp,
               "max_tokens": max_tokens}
    
    req = requests.post(url, headers=headers, json=request, timeout=60)
    if req.status_code >= 400:
        raise RuntimeError(f"{req.status_code}: {req.text}")
    
    return req.json()['choices'][0]['message']['content'].strip()

def normalize_question(question: str) -> str:
    q = question.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("«") and q.endswith("»")):
        q = q[1:-1].strip()
    else:
        q = q.strip('"').strip("«").strip("»").strip()

    if q and not q.endswith("?"):
        q = q.rstrip(".!…") + "?"
    return " ".join(q.split())

def wiki_article_title(link):
    if not link:
        return ""
    
    try:
        path = urlparse(link).path
        if not path:
            return ""
        
        title = path.split('/')[-1]
        title = unquote(title).replace("_", " ").strip()
        return title 
    except Exception:
        return "" 
    
def generate_question(api_base, api_key, model, answer_text, article_title, temp, max_tokens, max_retries=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(answer=answer_text, article_title=article_title)},
    ]
    
    for attempt in range(max_retries):
        try:
            raw = request(api_base, api_key, model, messages, temp, max_tokens)
            return normalize_question(raw)
        except Exception as e:
            sleep = (2 * (attempt + 1))
            time.sleep(sleep)
        
    raise RuntimeError("Failed to generate question after retries")

def _generate_one(api_base, api_key, model, answer_text, article_title, temp, max_tokens):
    
    return generate_question(
        api_base=api_base,
        api_key=api_key,
        model=model,
        answer_text=answer_text,
        article_title=article_title,
        temp=temp,
        max_tokens=max_tokens,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    api_base = os.getenv("VLLM_BASE_URL")
    api_key = os.getenv("VLLM_API_KEY")

    if not api_base:
        raise RuntimeError("VLLM_BASE_URL is not set in .env")

    if not api_key:
        raise RuntimeError("VLLM_API_KEY is not set in .env")

    data = load_json(args.input)

    if args.output is None:
        os.makedirs(args.model, exist_ok=True)
        args.output = os.path.join(args.model, "results.jsonl")

    tasks: List[Tuple[int, int, str, str, bool]] = []

    for oi, obj in enumerate(data):
        title = wiki_article_title((obj.get("link") or "").strip())
        main_text = (obj.get("text") or "").strip()
        if main_text:
            if not (args.skip_existing and obj.get("generated_question")):
                tasks.append((oi, -1, main_text, title, True))

        gen_list = obj.get("Generated_texts_by_QWEN32", [])
        if not isinstance(gen_list, list):
            continue

        for ii, item in enumerate(gen_list):
            if args.skip_existing and item.get("generated_question"):
                continue

            text = (item.get("generated_text") or "").strip()
            if not text:
                continue

            tasks.append((oi, ii, text, title, False))

    async def run_parallel() -> List[Tuple[int, int, str, bool]]:
        if not tasks:
            return []

        loop = asyncio.get_running_loop()
        results: List[Tuple[int, int, str, bool]] = []

        def sync_call(args_for_call):
            text, title = args_for_call
            return _generate_one(
                api_base=api_base,
                api_key=api_key,
                model=args.model,
                answer_text=text,
                article_title=title,
                temp=0.3,
                max_tokens=128,
            )

        async def one(oi: int, ii: int, text: str, article_title: str, is_main: bool, executor: ThreadPoolExecutor):
            q = await loop.run_in_executor(executor, sync_call, (text, article_title))
            return (oi, ii, q, is_main)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            coros = [one(oi, ii, text, title, is_main, executor) for (oi, ii, text, title, is_main) in tasks]
            for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Generating questions"):
                res = await fut
                results.append(res)
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)

        return results

    results = asyncio.run(run_parallel())

    for oi, ii, q, is_main in results:
        try:
            if is_main:
                data[oi]["generated_question"] = q
            else:
                data[oi]["Generated_texts_by_QWEN32"][ii]["generated_question"] = q
        except Exception:
            pass

    save_json(args.output, data)
    print(f"Done. Generated questions: {len(results)}")


if __name__ == "__main__":
    main()
import json 
import argparse
import asyncio
import os 
from tqdm import tqdm

from app import run, RunRequest, get_models, get_criteria


""" USAGE MANUAL
 - Run as `python criterion_evaluation.py --list-models` to check currently available models 
 - Run as `python criterion_evaluation.py -i input.json -o output.json --model <model_name>` to perform evaluation on all criterions with selected model
 - Run as `python batch_from_json.py -i input.json -o output.json --model llama-3-8b-instruct --criteria toxicity "obscene language"` to perform evaluation on selected criterions with selected model
"""

async def list_models():
    data = await get_models()  
    models = data.get("models", [])
    if not models:
        print("No currently available models, check URL or try again later")
        return
    
    print("Currently available models:")
    for model in models:
        mid = model["id"]
        status = model["status"]
        if status:
            print(f" - {mid} ({status})")
        else:
            print(f" - {mid}")

async def list_criteria():
    data = await get_criteria()
    criterias = data['criteria']

    print("Availables criterias for the moderation:")
    for criteria in criterias:
        print(f" - {criteria['key']} ({criteria['title']})")
    

async def process_file(in_path, out_path, model, criteria, concurrency: int = 5):

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "r", encoding='utf-8') as fin:
        data = json.load(fin)
    
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def process_one(item):
        text = item["text"]

        if not isinstance(text, str) or not text.strip():
            await queue.put({**item, "error": "Item is either empty or missing 'text' field"})
            return

        request = RunRequest(model=model, text=text, criteria=criteria)

        async with semaphore:
            try:
                response = await run(request)
                result = {**item, "moderation": {"model": model, "results": {k: v.dict() for k, v in response.results.items()}}}
            except Exception as e:
                result = {**item, "moderation": {"model": model, "error": str(e)}}

            await queue.put(result)
        
    async def writer(total):
        written = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            pbar = tqdm(total=total, desc="Processing texts")
            while True:
                item = await queue.get()
                if item is None:
                    break

                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()

                written += 1 
                pbar.update(1)

            pbar.close()

    writer_task = asyncio.create_task(writer(total=len(data)))
    tasks = [asyncio.create_task(process_one(item)) for item in data]

    await asyncio.gather(*tasks)

    await queue.put(None)
    await writer_task


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model")
    parser.add_argument("--criteria", nargs="*", default=None)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-criteria", action="store_true")
    
    args = parser.parse_args() 

    if args.list_models:
        await list_models()
        return

    if args.list_criteria:
        await list_criteria()
        return

    if not args.model:
        raise SystemExit("Нужно указать --model <id> или использовать --list-models для просмотра доступных моделей.")

    if not args.input:
        raise SystemExit("Нужно указать --input")
    
    # Auto-generate output path if not provided
    if args.output is None:
        os.makedirs(args.model, exist_ok=True)
        args.output = os.path.join(args.model, "results.jsonl")


    await process_file(
        in_path=args.input,
        out_path=args.output,
        model=args.model,
        criteria=args.criteria,
        concurrency=args.concurrency 
    )


if __name__ == "__main__":
    asyncio.run(main())
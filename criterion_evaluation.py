import json 
import argparse
import asyncio
import os 
import time
from datetime import datetime, timezone
from statistics import mean
from tqdm import tqdm

from app import run, RunRequest, get_models, get_criteria

"""
USAGE MANUAL
 - Run as `python criterion_evaluation.py --list-models` to check currently available models 
 - Run as `python criterion_evaluation.py -i input.json -o output.json --model <model_name>` to perform evaluation on all criterions with selected model
 - Run as `python criterion_evaluation.py -i input.json -o output.jsonl --model llama-3-8b-instruct --criteria toxicity --concurrency 5` to perform evaluation and collect performance statistics
"""


def percentile(values, percent):
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}

    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def print_performance_stats(stats):
    latency = stats["latency_seconds"]
    print("\nPerformance statistics")
    print(f"  Model: {stats['model']}")
    print(f"  Criteria ({stats['criteria_per_text']} per text): {stats['criteria']}")
    print(f"  Requests: {stats['successful_requests']} successful, {stats['failed_requests']} failed")
    print(f"  Concurrency: {stats['concurrency']}")
    print(f"  Measurement time: {stats['measurement_time_seconds']:.3f} s")
    print(f"  Throughput: {stats['throughput_texts_per_second']:.3f} texts/s")
    print(f"  Criterion throughput: {stats['throughput_criterion_evaluations_per_second']:.3f} evaluations/s")
    if latency["count"]:
        print(f"  Latency mean: {latency['mean']:.3f} s")
        print(f"  Latency p50/p90/p95/p99: {latency['p50']:.3f} / {latency['p90']:.3f} / {latency['p95']:.3f} / {latency['p99']:.3f} s")
        print(f"  Latency min/max: {latency['min']:.3f} / {latency['max']:.3f} s")

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
    
async def process_file(in_path, out_path, model, criteria, concurrency=5, warmup=0, stats_out=None, include_timing=False, hardware=None, max_retries=1, text_field="text"):

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = []
    with open(in_path, "r", encoding='utf-8') as fin:
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
    
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    successful_durations = []
    failed_durations = []
    invalid_items = 0

    async def process_one(item):
        nonlocal invalid_items
        text = item.get(text_field)

        if not isinstance(text, str) or not text.strip():
            invalid_items += 1
            await queue.put({**item, "error": f"Item is either empty or missing '{text_field}' field"})
            return

        request = RunRequest(model=model, text=text, criteria=criteria, max_retries=max_retries)

        async with semaphore:
            started = time.perf_counter()
            try:
                response = await run(request)
                result = {**item, "moderation": {"model": model, "results": {k: v.dict() for k, v in response.results.items()}}}
                duration = time.perf_counter() - started
                successful_durations.append(duration)
            except Exception as e:
                duration = time.perf_counter() - started
                failed_durations.append(duration)
                result = {**item, "moderation": {"model": model, "error": f"{type(e).__name__}: {repr(e)}"}}

            if include_timing:
                result["moderation"]["latency_seconds"] = duration

            await queue.put(result)
        
    async def writer(total):
        with open(out_path, "w", encoding="utf-8") as fout:
            pbar = tqdm(total=total, desc="Processing texts")
            while True:
                item = await queue.get()
                if item is None:
                    break

                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()

                pbar.update(1)

            pbar.close()

    valid_items = [item for item in data if isinstance(item.get(text_field), str) and item[text_field].strip()]
    available_criteria = [item["key"] for item in (await get_criteria())["criteria"]]
    unknown_criteria = set(criteria or []) - set(available_criteria)
    if unknown_criteria:
        raise ValueError(f"Unknown criteria: {', '.join(sorted(unknown_criteria))}")
    effective_criteria = available_criteria if not criteria else [key for key in available_criteria if key in set(criteria)]
    if warmup and valid_items:
        print(f"Warming up model with {warmup} request(s)...")
        warmup_request = RunRequest(model=model, text=valid_items[0][text_field], criteria=criteria, max_retries=max_retries)
        for _ in range(warmup):
            await run(warmup_request)

    writer_task = asyncio.create_task(writer(total=len(data)))
    measurement_started_at = datetime.now(timezone.utc)
    measurement_started = time.perf_counter()
    tasks = [asyncio.create_task(process_one(item)) for item in data]

    await asyncio.gather(*tasks)
    measurement_time = time.perf_counter() - measurement_started

    await queue.put(None)
    await writer_task

    attempted_requests = len(successful_durations) + len(failed_durations)
    stats = {
        "model": model,
        "hardware": hardware,
        "criteria": effective_criteria,
        "criteria_per_text": len(effective_criteria),
        "input_file": os.path.abspath(in_path),
        "output_file": os.path.abspath(out_path),
        "measurement_started_at": measurement_started_at.isoformat(),
        "concurrency": max(1, int(concurrency)),
        "warmup_requests": max(0, int(warmup)),
        "max_retries": max_retries,
        "total_items": len(data),
        "invalid_items": invalid_items,
        "attempted_requests": attempted_requests,
        "successful_requests": len(successful_durations),
        "failed_requests": len(failed_durations),
        "measurement_time_seconds": measurement_time,
        "attempted_texts_per_second": attempted_requests / measurement_time if measurement_time else 0,
        "throughput_texts_per_second": len(successful_durations) / measurement_time if measurement_time else 0,
        "throughput_criterion_evaluations_per_second": len(successful_durations) * len(effective_criteria) / measurement_time if measurement_time else 0,
        "latency_seconds": latency_summary(successful_durations),
        "failed_request_latency_seconds": latency_summary(failed_durations),
        "input_characters": latency_summary([len(item[text_field]) for item in valid_items])
    }

    if stats_out is None:
        stats_out = f"{out_path}.stats.json"
    stats_dir = os.path.dirname(stats_out)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(stats_out, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)

    print_performance_stats(stats)
    print(f"  Full statistics: {stats_out}")
    return stats


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model")
    parser.add_argument("--text-field", default="text", help="Input field containing the text to evaluate (default: text).")
    parser.add_argument("--criteria", nargs="*", default=None)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0, help="Number of unmeasured warm-up requests.")
    parser.add_argument("--stats-output", default=None, help="Performance statistics JSON path. Defaults to <output>.stats.json.")
    parser.add_argument("--include-timing", action="store_true", help="Add end-to-end latency to each output row.")
    parser.add_argument("--hardware", default=None, help="Hardware description stored in the statistics, e.g. '1x NVIDIA H100 80GB'.")
    parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=1, help="Retries after an invalid model response; included in latency.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-criteria", action="store_true")
    
    args = parser.parse_args() 

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.list_models:
        await list_models()
        return

    if args.list_criteria:
        await list_criteria()
        return

    if not args.model:
        raise SystemExit("Provide --model <id> or use --list-models to get information about available models.")

    if not args.input:
        raise SystemExit("--input is required")
    
    if args.output is None:
        os.makedirs(args.model, exist_ok=True)
        args.output = os.path.join(args.model, "results_test_v2.jsonl")


    await process_file(
        in_path=args.input,
        text_field=args.text_field,
        out_path=args.output,
        model=args.model,
        criteria=args.criteria,
        concurrency=args.concurrency,
        warmup=args.warmup,
        stats_out=args.stats_output,
        include_timing=args.include_timing,
        hardware=args.hardware,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Build a reproducible, duplicate-free source set for question generation."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUSSIA = PROJECT_DIR/"provocative_data/russia_statements_1000.jsonl"
DEFAULT_NEGATIVE = PROJECT_DIR/"provocative_data/negative_samples.json"
DEFAULT_OUTPUT = PROJECT_DIR/"provocative_data/provocative_question_sources_all.jsonl"

def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        first = ""
        while True:
            char = source.read(1)
            if not char or not char.isspace():
                first = char
                break
        source.seek(0)
        if first == "[":
            records = json.load(source)
        else:
            records = [json.loads(line) for line in source if line.strip()]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path} must contain a JSON array or JSONL objects")
    return records


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return " ".join(value.split())


def length_ratio_upper_bound(left: str, right: str) -> float:
    return 2 * min(len(left), len(right)) / (len(left) + len(right))


def collect_near_duplicate_pairs(
    normalized_records: list[tuple[str, str]], threshold: float, example_limit: int
) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    examples = []
    for right_index, (right_id, right_text) in enumerate(normalized_records):
        for left_id, left_text in normalized_records[:right_index]:
            if length_ratio_upper_bound(left_text, right_text) < threshold:
                continue
            score = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if score < threshold:
                continue
            count += 1
            if len(examples) < example_limit:
                examples.append({"left": left_id, "right": right_id, "similarity": round(score, 6)})
    return count, examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    russia_rows = load_records(args.russia_input)
    negative_rows = load_records(args.negative_input)

    selected: list[dict[str, Any]] = []
    selected_normalized: list[tuple[str, str]] = []
    exact_keys: set[str] = set()

    for source_index, row in enumerate(russia_rows):
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"Empty text in {args.russia_input} at row {source_index}")
        normalized = normalize_text(text)
        if normalized in exact_keys:
            raise ValueError(f"Normalized duplicate remains in Russia statements at row {source_index}: {text!r}")
        source_id = f"russia_statements_1000:{source_index:04d}"
        selected.append({
            "text": text,
            "source_dataset": "russia_statements_1000",
            "source_id": source_id,
            "source_index": source_index,
            "source_url": row.get("source_url"),
            "date": row.get("date"),
            "raw_pred_label": None,
        })
        exact_keys.add(normalized)
        selected_normalized.append((source_id, normalized))

    # All original Russia rows must remain; near matches inside this mandatory part are reported, not removed.
    russia_near_pair_count, russia_near_pair_examples = collect_near_duplicate_pairs(
        selected_normalized, args.near_duplicate_threshold, args.report_pair_limit
    )

    unique_negative = []
    negative_seen: set[str] = set()
    negative_exact_duplicates = 0
    cross_exact_duplicates = 0
    for source_index, row in enumerate(negative_rows):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        normalized = normalize_text(text)
        if normalized in negative_seen:
            negative_exact_duplicates += 1
            continue
        negative_seen.add(normalized)
        if normalized in exact_keys:
            cross_exact_duplicates += 1
            continue
        unique_negative.append((source_index, row, text, normalized))

    rng = random.Random(args.seed)
    for source_index, row, text, normalized in unique_negative:
        source_id = f"negative_samples:{row.get('sentence_id', source_index)}"
        selected.append({
            "text": text,
            "source_dataset": "negative_samples",
            "source_id": source_id,
            "source_index": source_index,
            "source_url": None,
            "date": None,
            "raw_pred_label": row.get("pred_label"),
        })
        exact_keys.add(normalized)
        selected_normalized.append((source_id, normalized))

    normalized_final = [normalize_text(row["text"]) for row in selected]
    if len(normalized_final) != len(set(normalized_final)):
        raise RuntimeError("Internal error: exact normalized duplicates remain in the output")

    rng.shuffle(selected)
    for benchmark_index, record in enumerate(selected, start=1):
        record["benchmark_id"] = f"pq-{benchmark_index:05d}"

    report = {
        "seed": args.seed,
        "total_rows": len(selected),
        "source_counts": dict(sorted(Counter(row["source_dataset"] for row in selected).items())),
        "normalized_exact_duplicate_groups": 0,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "russia_near_duplicate_pairs": russia_near_pair_count,
        "russia_near_duplicate_pair_examples": russia_near_pair_examples,
        "negative_exact_duplicates_removed": negative_exact_duplicates,
        "cross_source_exact_duplicates_excluded": cross_exact_duplicates,
        "full_pool_near_duplicate_check_performed": False,
        "near_duplicates_removed": 0,
    }
    return selected, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare all unique source texts for provocative questions")
    parser.add_argument("--russia-input", type=Path, default=DEFAULT_RUSSIA)
    parser.add_argument("--negative-input", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--report-pair-limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.near_duplicate_threshold <= 1:
        parser.error("--near-duplicate-threshold must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.russia_input, args.negative_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    report_path = Path(str(args.output) + ".qa.json")
    if (args.output.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{args.output} or its QA report already exists; use --overwrite")

    records, report = build_dataset(args)
    write_jsonl(args.output, records)
    report["output"] = str(args.output)
    report["output_sha256"] = dataset_sha256(args.output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

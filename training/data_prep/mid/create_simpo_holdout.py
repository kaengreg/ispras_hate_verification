import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_num}: {exc}") from exc


def row_id(row):
    return row.get("id", row.get("sentence_id", row.get("question_id")))


def question_text(row):
    value = row.get("question") or row.get("prompt") or row.get("generated_question")
    return str(value or "").strip()


def normalized_source_hash(row):
    source = " ".join(str(row.get("text") or "").split()).casefold()
    if not source:
        raise ValueError("missing source text")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def stable_hash(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def proportional_targets(stratum_counts, total_target):
    total = sum(stratum_counts.values())
    exact = {key: count * total_target / total for key, count in stratum_counts.items()}
    targets = {key: math.floor(value) for key, value in exact.items()}
    remainder = total_target - sum(targets.values())
    order = sorted(stratum_counts, key=lambda key: (-(exact[key] - targets[key]), str(key)))
    for key in order[:remainder]:
        targets[key] += 1
    return targets


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Canonical question or candidate JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Test-question manifest JSONL")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--test-ratio", type=float, default=0.1)
    size.add_argument("--test-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    groups = defaultdict(list)
    stratum_counts = Counter()
    seen_ids = set()
    for line_num, row in jsonl_rows(args.input):
        item_id = row_id(row)
        question = question_text(row)
        if item_id is None:
            raise ValueError(f"Missing id in {args.input} on line {line_num}")
        if item_id in seen_ids:
            raise ValueError(f"Repeated id={item_id!r} in {args.input} on line {line_num}")
        if not question:
            raise ValueError(f"Missing question for id={item_id!r}")
        seen_ids.add(item_id)

        stratum = (str(row.get("question_type") or "UNKNOWN"), str(row.get("source_label") or "UNKNOWN"))
        source_hash = normalized_source_hash(row)
        record = {"id": item_id, "stratum": stratum, "source_hash": source_hash}
        records.append(record)
        groups[source_hash].append(record)
        stratum_counts[stratum] += 1

    total_rows = len(records)
    test_size = args.test_size if args.test_size is not None else round(total_rows * args.test_ratio)
    if not 0 < test_size < total_rows:
        raise ValueError(f"Test size must be between 1 and {total_rows - 1}, got {test_size}")
    targets = proportional_targets(stratum_counts, test_size)

    selected_ids = set()
    selected_by_stratum = Counter()
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    duplicate_groups.sort(key=lambda items: stable_hash(args.seed, items[0]["source_hash"]))
    duplicate_threshold = test_size / total_rows

    for items in duplicate_groups:
        score = int(stable_hash(args.seed, items[0]["source_hash"]), 16) / 2**256
        group_counts = Counter(item["stratum"] for item in items)
        fits = all(selected_by_stratum[key] + count <= targets[key] for key, count in group_counts.items())
        if score < duplicate_threshold and fits:
            selected_ids.update(item["id"] for item in items)
            selected_by_stratum.update(group_counts)

    singletons_by_stratum = defaultdict(list)
    for items in groups.values():
        if len(items) == 1:
            singletons_by_stratum[items[0]["stratum"]].append(items[0])

    for stratum, target in targets.items():
        needed = target - selected_by_stratum[stratum]
        candidates = sorted(singletons_by_stratum[stratum], key=lambda item: stable_hash(args.seed, f"{stratum}:{item['id']}"))

        if needed > len(candidates):
            raise ValueError(f"Not enough singleton rows to fill stratum={stratum}: need {needed}")
        
        selected_ids.update(item["id"] for item in candidates[:needed])
        selected_by_stratum[stratum] += needed

    if len(selected_ids) != test_size or selected_by_stratum != Counter(targets):
        raise RuntimeError("Internal split error: selected holdout does not match requested stratum targets")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    written = 0
    with temporary_output.open("w", encoding="utf-8") as target:
        for _, row in jsonl_rows(args.input):
            if row_id(row) not in selected_ids:
                continue
            output = {
                "id": row_id(row),
                "text": row.get("text"),
                "prompt": question_text(row),
                "source_label": row.get("source_label"),
                "question_type": row.get("question_type"),
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    temporary_output.replace(args.output)

    metadata_path = Path(str(args.output) + ".split.json")
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "total_rows": total_rows,
        "test_size": written,
        "test_ratio": written / total_rows,
        "group_key": "normalized_source_text",
        "stratification": ["question_type", "source_label"],
        "input_strata": {"|".join(key): value for key, value in sorted(stratum_counts.items())},
        "test_strata": {"|".join(key): value for key, value in sorted(selected_by_stratum.items())},
    }
    save_json(metadata_path, metadata)
    print(f"Saved {written} fixed test questions to {args.output}")
    print(f"Saved split metadata to {metadata_path}")
    print(f"Test strata={metadata['test_strata']}")


if __name__ == "__main__":
    main()

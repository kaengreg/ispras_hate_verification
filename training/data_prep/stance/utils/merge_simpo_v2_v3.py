import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text).strip())
    return re.sub(r"\s+", " ", text).casefold()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_num}: {exc}") from exc
    return rows


def require_strings(row, fields, path, line_num):
    missing = [field for field in fields if not isinstance(row.get(field), str) or not row[field].strip()]
    if missing:
        raise ValueError(f"Missing or empty {', '.join(missing)} in {path} on line {line_num}")


def prepare_train(rows, path):
    prepared = []
    for line_num, row in enumerate(rows, start=1):
        require_strings(row, ("prompt", "chosen", "rejected"), path, line_num)
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"chosen equals rejected in {path} on line {line_num}")
        prepared.append({field: row[field].strip() for field in ("prompt", "chosen", "rejected")})
    return prepared


def prepare_v2_test(rows, path):
    prepared = []
    for line_num, row in enumerate(rows, start=1):
        require_strings(row, ("prompt", "rejected"), path, line_num)
        meta = row.get("meta") or {}
        prepared.append({
            "prompt": row["prompt"].strip(),
            "text": row["rejected"].strip(),
            "source_dataset": "simpo_v2",
            "source_id": str(row.get("sample_uid") or ""),
            "source_stance": meta.get("negative_stance"),
            "preferred_stance": meta.get("positive_stance"),
            "question_type": None,
        })
    return prepared


def prepare_v3_test(rows, path):
    prepared = []
    for line_num, row in enumerate(rows, start=1):
        require_strings(row, ("prompt", "text"), path, line_num)
        prepared.append({
            "prompt": row["prompt"].strip(),
            "text": row["text"].strip(),
            "source_dataset": "mid_simpo_v3",
            "source_id": str(row.get("id") or ""),
            "source_stance": row.get("source_label"),
            "preferred_stance": None,
            "question_type": row.get("question_type"),
        })
    return prepared


def validate_prompt_partition(splits):
    prompt_sets = {}
    for name, rows in splits.items():
        prompts = [normalize(row["prompt"]) for row in rows]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Combined {name} contains duplicate normalized prompts")
        prompt_sets[name] = set(prompts)

    names = list(prompt_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = prompt_sets[left] & prompt_sets[right]
            if overlap:
                raise ValueError(f"Combined {left}/{right} contain {len(overlap)} overlapping prompts")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Merge and shuffle SimPO v2 and MID SimPO v3")
    parser.add_argument("--v2-train", type=Path, required=True)
    parser.add_argument("--v2-test", type=Path, required=True)
    parser.add_argument("--v3-train", type=Path, required=True)
    parser.add_argument("--v3-test", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = (args.v2_train, args.v2_test, args.v3_train, args.v3_test)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if args.val_fraction > 0 and args.val_output is None:
        raise ValueError("--val-output is required when --val-fraction is positive")
    if args.val_output is not None and args.val_fraction == 0:
        raise ValueError("--val-fraction must be positive when --val-output is provided")
    output_paths = [args.train_output, args.test_output]
    if args.val_output is not None:
        output_paths.append(args.val_output)
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Train, validation and test output paths must differ")

    v2_train = prepare_train(load_jsonl(args.v2_train), args.v2_train)
    v3_train = prepare_train(load_jsonl(args.v3_train), args.v3_train)
    v2_test = prepare_v2_test(load_jsonl(args.v2_test), args.v2_test)
    v3_test = prepare_v3_test(load_jsonl(args.v3_test), args.v3_test)
    train_rows = v2_train + v3_train
    test_rows = v2_test + v3_test
    validate_prompt_partition({"train": train_rows, "test": test_rows})

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    val_rows = []
    if args.val_fraction > 0:
        val_size = max(1, round(len(train_rows) * args.val_fraction))
        val_rows = train_rows[:val_size]
        train_rows = train_rows[val_size:]

    validate_prompt_partition({"train": train_rows, "validation": val_rows, "test": test_rows})
    write_jsonl(args.train_output, train_rows)
    if args.val_output is not None:
        write_jsonl(args.val_output, val_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"Preference pool: v2={len(v2_train)}, v3={len(v3_train)}, total={len(v2_train) + len(v3_train)}")
    print(f"Train: {len(train_rows)} -> {args.train_output}")
    if args.val_output is not None:
        print(f"Validation: {len(val_rows)} ({args.val_fraction:.2%}) -> {args.val_output}")
    print(f"Test: v2={len(v2_test)}, v3={len(v3_test)}, total={len(test_rows)} -> {args.test_output}")
    print(f"Shuffle seed: {args.seed}")


if __name__ == "__main__":
    main()

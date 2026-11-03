import argparse
import json
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


def validate_simpo(rows, path):
    for line_num, row in enumerate(rows, start=1):
        missing = [field for field in ("prompt", "chosen", "rejected") if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Missing or empty {', '.join(missing)} in {path} on line {line_num}")


def load_stancerus_test_texts(path):
    texts = set()
    for line_num, row in enumerate(load_jsonl(path), start=1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing or empty text in {path} on line {line_num}")
        texts.add(normalize(text))
    return texts


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Re-split SimPO by rejected-text membership in StanceRus test")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--stancerus-test", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.train_input, args.test_input, args.stancerus_test):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.train_output == args.test_output:
        raise ValueError("Train and test output paths must differ")

    old_train = load_jsonl(args.train_input)
    old_test = load_jsonl(args.test_input)
    validate_simpo(old_train, args.train_input)
    validate_simpo(old_test, args.test_input)
    rows = old_train + old_test

    prompt_keys = [normalize(row["prompt"]) for row in rows]
    if len(prompt_keys) != len(set(prompt_keys)):
        raise ValueError("Combined SimPO inputs contain duplicate normalized prompts")

    stancerus_test_texts = load_stancerus_test_texts(args.stancerus_test)
    train_rows = [row for row in rows if normalize(row["rejected"]) not in stancerus_test_texts]
    test_rows = [row for row in rows if normalize(row["rejected"]) in stancerus_test_texts]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = args.train_output.with_suffix(args.train_output.suffix + ".tmp")
    test_tmp = args.test_output.with_suffix(args.test_output.suffix + ".tmp")
    try:
        write_jsonl(train_tmp, train_rows)
        write_jsonl(test_tmp, test_rows)
        train_tmp.replace(args.train_output)
        test_tmp.replace(args.test_output)
    except Exception:
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)
        raise

    old_train_to_test = sum(normalize(row["rejected"]) in stancerus_test_texts for row in old_train)
    old_test_to_train = sum(normalize(row["rejected"]) not in stancerus_test_texts for row in old_test)
    print(f"Combined rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)} -> {args.train_output}")
    print(f"Test rows: {len(test_rows)} -> {args.test_output}")
    print(f"Moved train -> test: {old_train_to_test}")
    print(f"Moved test -> train: {old_test_to_train}")


if __name__ == "__main__":
    main()

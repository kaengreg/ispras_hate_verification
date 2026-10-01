import json 
import argparse
import hashlib 
import re
from collections import Counter, defaultdict
from pathlib import Path 
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]

def save_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def record_key(record):
    return normalize(record["text"]), normalize(record["target"])

def normalize(string):
    return re.sub(r"\s+", " ", string.strip().lower())

def validate(rows, split):
    for idx, row in enumerate(rows, 1):
        missing = [field for field in ('text', 'target', 'stance') if not row.get(field)]
        if missing:
            raise ValueError(f"{split} line {idx}: missing {', '.join(missing)}")
        if row["stance"] not in {"ANTI", "NEU", "PRO"}:
            raise ValueError(f"{split} line {idx}: unexpected stance {row['stance']}")


def main(args):
    source_train = load_jsonl(args.source_train)
    source_test = load_jsonl(args.source_test)
    validate(source_train, "source train")
    validate(source_test, "test")

    target = normalize(args.target)
    filtered_train = [row for row in source_train if normalize(row["target"]) == target]
    filtered_test = [row for row in source_test if normalize(row["target"]) == target]

    train, val = train_test_split(filtered_train, test_size=args.val_size, random_state=args.seed, shuffle=True, stratify=[row["stance"] for row in filtered_train])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir/"StanceRus_24_08_train.jsonl"
    val_path = output_dir/"StanceRus_24_08_val.jsonl"
    test_path = output_dir/"StanceRus_24_08_test.jsonl"
    save_jsonl(train_path, train)
    save_jsonl(val_path, val)
    save_jsonl(test_path, filtered_test)

    split_keys = {"train": {record_key(row) for row in train}, "val": {record_key(row) for row in val}, "test": {record_key(row) for row in filtered_test}}
    overlaps = {"train_val": len(split_keys["train"] & split_keys["val"]), "train_test": len(split_keys["train"] & split_keys["test"]), "val_test": len(split_keys["val"] & split_keys["test"])}
    if any(overlaps.values()):
        raise RuntimeError(f"Prepared splits overlap: {overlaps}")
    print(f"Saved train={len(train)}, val={len(val)}, test={len(filtered_test)} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", default="data/guard_train/RuStance2.1_15_07_TRAIN_deduplicated_no_val_leaks.jsonl")
    parser.add_argument("--source-test", default="data/guard_train/RuStance2.1_15_07_VAL_no_duplicates.jsonl")
    parser.add_argument("--output-dir", default="data/guard_train/splits")
    parser.add_argument("--target", default="Россия")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

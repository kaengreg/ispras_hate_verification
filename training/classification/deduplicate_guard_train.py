import argparse
import json
import re
from collections import defaultdict


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())

def main(args):
    with open(args.input, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    with open(args.val, encoding="utf-8") as file:
        val_rows = [json.loads(line) for line in file if line.strip()]

    val_keys = {(normalize(row["text"]), normalize(row["target"])) for row in val_rows}

    groups = defaultdict(list)
    for row in rows:
        groups[(normalize(row["text"]), normalize(row["target"]))].append(row)

    deduplicated = []
    for key, group in groups.items():
        if key in val_keys:
            continue
        if len({row["stance"] for row in group}) == 1:
            deduplicated.append(group[0])

    with open(args.output, "w", encoding="utf-8") as file:
        for row in deduplicated:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(deduplicated)} of {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/guard_train/RuStance2.1_15_07_TRAIN_no_duplicates.jsonl")
    parser.add_argument("--val", default="data/guard_train/RuStance2.1_15_07_VAL_no_duplicates.jsonl")
    parser.add_argument("--output", default="data/guard_train/RuStance2.1_15_07_TRAIN_deduplicated_no_val_leaks.jsonl")
    main(parser.parse_args())

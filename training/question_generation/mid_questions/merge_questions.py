import argparse
import json
from pathlib import Path

from sklearn.utils import shuffle


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
    elif path.suffix.lower() == ".jsonl":
        data = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at line {line_number} in {path}: {error}") from error
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .jsonl")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at item {index} in {path}, got {type(item).__name__}")
    return data


def save_data(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as target:
            json.dump(data, target, ensure_ascii=False, indent=4)
            target.write("\n")
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as target:
            for item in data:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    raise ValueError(f"Unsupported output format: {path.suffix}. Use .json or .jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", default="data/mid/questions/pos_balanced.jsonl")
    parser.add_argument("--neutral", default="data/mid/questions/neu_balanced.jsonl")
    parser.add_argument("--output", default="data/mid/questions/balanced_shuffled.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive = load_data(args.positive)
    neutral = load_data(args.neutral)
    combined = positive + neutral
    combined = shuffle(combined, random_state=args.seed)

    save_data(args.output, combined)

    print(f"Positive: {len(positive)}")
    print(f"Neutral: {len(neutral)}")
    print(f"Saved: {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR/"balanced_shuffled.jsonl"
DEFAULT_OUTPUT = DATA_DIR/"balanced_shuffled_filtered.jsonl"


def load_items(path):
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as source:
            items = json.load(source)
        if not isinstance(items, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return items

    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
        return items

    raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .jsonl")


def save_items(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as target:
            json.dump(items, target, ensure_ascii=False, indent=4)
            target.write("\n")
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as target:
            for item in items:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    raise ValueError(f"Unsupported output format: {path.suffix}. Use .json or .jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-words", type=int, default=15)
    args = parser.parse_args()

    items = load_items(args.input)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at item {index}, got {type(item).__name__}")

    filtered = [item for item in items if isinstance(item.get("text"), str) and len(item["text"].split()) >= args.min_words]
    save_items(args.output, filtered)

    print(f"Total: {len(items)}")
    print(f"Kept: {len(filtered)}")
    print(f"Removed: {len(items) - len(filtered)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

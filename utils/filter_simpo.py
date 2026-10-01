import argparse
import json
from pathlib import Path

MID_DIR = Path(__file__).resolve().parent.parent
FILTERED = MID_DIR/"questions"/"balanced_shuffled_filtered.jsonl"

def load_texts(path):
    texts = set()
    with path.open(encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_num}: {error}")

            text = record.get('text')
            if isinstance(text, str):
                texts.add(text)
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", type=Path, default=FILTERED)
    parser.add_argument("--input", type=Path, default='')
    parser.add_argument("--output", type=Path, default='')
    args = parser.parse_args()

    allowed_texts = load_texts(args.filtered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0

    with args.input.open(encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as destination:
        for line_num, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {args.input} at line {line_num}: {error}")

            total += 1
            if record.get("text") in allowed_texts:
                destination.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1

    print(f"Allowed texts: {len(allowed_texts)}")
    print(f"SIMPO rows: {total}")
    print(f"Kept: {kept}")
    print(f"Removed: {total - kept}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()


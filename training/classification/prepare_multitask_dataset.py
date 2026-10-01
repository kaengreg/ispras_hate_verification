import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


STANCE_LABELS = {"ANTI", "NEU", "PRO"}
TOXICITY_LABELS = {"TOX", "NEU"}

STANCE_SYSTEM_PROMPT = (
    "Ты — модель для классификации позиции автора по отношению к указанной цели. "
    "Есть три класса: ANTI, PRO и NEU. "
    "Верни только одну метку: ANTI, PRO или NEU."
)
TOXICITY_SYSTEM_PROMPT = (
    "Ты — модель для классификации токсичности высказывания. "
    "TOX означает токсичное, оскорбительное, агрессивное или уничижительное высказывание. "
    "NEU означает нетоксичное высказывание. "
    "Токсичность не равна негативной позиции по отношению к цели. "
    "Верни только одну метку: TOX или NEU."
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    label = str(value).strip().upper()
    return None if label in {"", "NAN", "NONE", "NULL", "UNDEF"} else label


def build_stance_example(row, split, line_number):
    text = str(row.get("text") or "").strip()
    target = str(row.get("target") or "").strip()
    stance = normalize_label(row.get("stance"))
    if not text or not target:
        raise ValueError(f"{split} line {line_number}: text and target must be non-empty")
    if stance not in STANCE_LABELS:
        raise ValueError(f"{split} line {line_number}: unexpected stance {row.get('stance')!r}")

    user_prompt = (
        "Задача: классификация позиции.\n\n"
        f"Цель: {target}\n\n"
        f"Высказывание:\n{text}\n\n"
        "Определи позицию автора по отношению к цели. Ответь только одной меткой: ANTI, PRO или NEU."
    )
    return {
        "messages": [
            {"role": "system", "content": STANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": stance},
        ],
        "task": "stance",
        "sentence_id": row.get("sentence_id"),
    }


def build_toxicity_example(row, split, line_number):
    toxicity = normalize_label(row.get("toxicity_label"))
    if toxicity is None:
        return None
    if toxicity not in TOXICITY_LABELS:
        raise ValueError(f"{split} line {line_number}: unexpected toxicity_label {row.get('toxicity_label')!r}")

    text = str(row.get("text") or "").strip()
    target = str(row.get("target") or "").strip()
    if not text or not target:
        raise ValueError(f"{split} line {line_number}: text and target must be non-empty")

    user_prompt = (
        "Задача: классификация токсичности.\n\n"
        f"Цель/объект упоминания: {target}\n\n"
        f"Высказывание:\n{text}\n\n"
        "Определи токсичность самого высказывания независимо от позиции по отношению к цели. "
        "Ответь только одной меткой: TOX или NEU."
    )
    return {
        "messages": [
            {"role": "system", "content": TOXICITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": toxicity},
        ],
        "task": "toxicity",
        "sentence_id": row.get("sentence_id"),
    }


def prepare_split(rows, split, toxicity_repeats, seed):
    examples = []
    counts = Counter()
    for line_number, row in enumerate(rows, 1):
        stance_example = build_stance_example(row, split, line_number)
        examples.append(stance_example)
        counts[f"stance/{stance_example['messages'][-1]['content']}"] += 1

        toxicity_example = build_toxicity_example(row, split, line_number)
        if toxicity_example is None:
            counts["toxicity/missing"] += 1
            continue
        for _ in range(toxicity_repeats):
            examples.append(toxicity_example)
        counts[f"toxicity/{toxicity_example['messages'][-1]['content']}"] += toxicity_repeats

    random.Random(seed).shuffle(examples)
    return examples, counts


def main(args):
    train_rows = load_jsonl(args.train_input)
    val_rows = load_jsonl(args.val_input)
    train_examples, train_counts = prepare_split(train_rows, "train", args.toxicity_repeats, args.seed)
    val_examples, val_counts = prepare_split(val_rows, "validation", 1, args.seed)

    output_dir = Path(args.output_dir)
    train_output = output_dir/"train_chat.jsonl"
    val_output = output_dir/"val_chat.jsonl"
    save_jsonl(train_output, train_examples)
    save_jsonl(val_output, val_examples)

    print(f"Saved train={len(train_examples)} to {train_output}")
    print(f"Train source rows={len(train_rows)}, counts={dict(sorted(train_counts.items()))}")
    print(f"Saved validation={len(val_examples)} to {val_output}")
    print(f"Validation source rows={len(val_rows)}, counts={dict(sorted(val_counts.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/guard_train/splits/StanceRus_24_08_train.jsonl")
    parser.add_argument("--val-input", default="data/guard_train/splits/StanceRus_24_08_val.jsonl")
    parser.add_argument("--output-dir", default="data/guard_train/splits/multitask")
    parser.add_argument("--toxicity-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

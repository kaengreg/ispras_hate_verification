import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import math

TOX_LABELS = ["TOX", "NEU"]

SYSTEM_PROMPT = (
    "Ты - модель для классификации токсичности высказывания. "
    "Есть два класса: TOX и NEU. "
    "TOX означает токсичное, оскорбительное, агрессивное или уничижительное высказывание. "
    "NEU означает нетоксичное высказывание. "
    "Необходимо вернуть ТОЛЬКО одну метку: TOX или NEU."
)


def load_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as fin:
        first_char = fin.read(1)
        fin.seek(0)

        if first_char == "[":
            data = json.load(fin)
        else:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    return data


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

def normalize_tox_label(label):
    if label is None:
        return None

    if isinstance(label, float) and math.isnan(label):
        return None

    label = str(label).strip().upper()

    if label in TOX_LABELS:
        return label

    return None

def build_messages(example):
    text = example["text"]
    target = example.get("target", "")
    tox_label = normalize_tox_label(example.get("toxicity_label"))

    if tox_label not in TOX_LABELS:
        raise ValueError(
            f"UNEXPECTED TOXICITY LABEL: {example.get('toxicity_label')} "
            f"(id={example.get('sentence_id')})"
        )

    user_content = (
        "Определи, является ли высказывание токсичным.\n\n"
        f"**Высказывание:**\n\n{text}\n\n"
        f"**Цель/объект упоминания:**\n\n{target}\n\n"
        "Ответ должен быть ОДНИМ словом: 'TOX' или 'NEU'."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": tox_label},
    ]

    return {"messages": messages}


def main(input_path, output_dir, test_path=None, test_size=0.1, random_seed=42):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}")
    data = load_json(input_path)
    print(f"Total examples: {len(data)}")

    clean_data = []
    skipped = 0

    for example in data:
        tox_label = normalize_tox_label(example.get("toxicity_label"))

        if tox_label not in TOX_LABELS:
            skipped += 1
            print(
                f"Skipping example with bad/missing toxicity_label: "
                f"{example.get('toxicity_label')} "
                f"(id={example.get('sentence_id')})"
            )
            continue

        example = dict(example)
        example["toxicity_label"] = tox_label
        clean_data.append(example)

    print(f"After toxicity label filtering: {len(clean_data)} examples")
    print(f"Skipped: {skipped}")

    if test_path is not None:
        print(f"Loading existing test set from {test_path}")

        test_data_raw = load_json(test_path)
        test_ids = set()

        for example in test_data_raw:
            sent_id = example.get("sentence_id")
            if sent_id is not None:
                test_ids.add(sent_id)

        train_data = []
        test_data = []

        for example in clean_data:
            if example.get("sentence_id") in test_ids:
                test_data.append(example)
            else:
                train_data.append(example)

    else:
        if test_size is not None and test_size > 0:
            train_data, test_data = train_test_split(
                clean_data,
                test_size=test_size,
                random_state=random_seed,
                shuffle=True,
                stratify=[example["toxicity_label"] for example in clean_data],
            )
        else:
            train_data = clean_data
            test_data = []

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    train_chat = [build_messages(example) for example in train_data]

    train_chat_path = output_dir/"train_chat.jsonl"
    train_raw_path = output_dir/"train_raw.jsonl"
    test_raw_path = output_dir/"test_raw.jsonl"

    print(f"Saving train dataset to {train_chat_path}")
    save_jsonl(train_chat_path, train_chat)

    print(f"Saving train raw dataset to {train_raw_path}")
    save_jsonl(train_raw_path, train_data)

    if test_size is not None and test_size > 0:
        print(f"Saving test raw dataset to {test_raw_path}")
        save_jsonl(test_raw_path, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/toxicity_classification/")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-path", type=str, default=None)

    args = parser.parse_args()

    main(input_path=args.input, output_dir=args.output_dir, test_size=args.test_size, random_seed=args.seed, test_path=args.test_path)
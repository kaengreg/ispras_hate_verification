import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CLASS_TO_LABEL = {-1: "NEG", 0: "NEU", 1: "POS"}
LABEL_TO_CLASS = {label: source_class for source_class, label in CLASS_TO_LABEL.items()}
LABEL_PRIORITY = ("NEG", "NEU", "POS")


def clean_text(text):
    text = html.unescape(text or "")
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?b>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def aggregate_label(rows):
    labels = {CLASS_TO_LABEL[row["class"]] for row in rows}
    return next(label for label in LABEL_PRIORITY if label in labels)


def aggregate_texts(source_rows):
    rows_by_text = defaultdict(list)
    for row in source_rows:
        text = clean_text(row["text"])
        if text:
            rows_by_text[text].append(row)

    result = []
    for sentence_id, (text, rows) in enumerate(rows_by_text.items(), start=1):
        toxicity = aggregate_label(rows)
        ethnic_groups = sorted({row["ethnic_group"] for row in rows})
        targets_by_stance = {
            label: sorted({row["ethnic_group"] for row in rows if CLASS_TO_LABEL[row["class"]] == label})
            for label in LABEL_PRIORITY
        }
        result.append({
            "sentence_id": sentence_id,
            "text": text,
            "toxicity": toxicity,
            "class": LABEL_TO_CLASS[toxicity],
            "target": ", ".join(ethnic_groups),
            "ethnic_groups": ethnic_groups,
            "targets_by_stance": targets_by_stance,
            "text_ids": sorted({row["text_id"] for row in rows}),
            "instance_ids": sorted({row["instance_id"] for row in rows}),
            "n_targets": len(ethnic_groups),
            "has_mixed_target_labels": len({row["class"] for row in rows}) > 1,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/national_hate/RuEthnoHate.json"))
    parser.add_argument("--output", type=Path, default=Path("data/national_hate/RuEthnoHateAggregated.jsonl"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as fin:
        source_rows = json.load(fin)
    aggregated_rows = aggregate_texts(source_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for row in aggregated_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_rows": len(source_rows),
        "unique_texts": len(aggregated_rows),
        "labels": dict(Counter(row["toxicity"] for row in aggregated_rows)),
        "mixed_target_labels": sum(row["has_mixed_target_labels"] for row in aggregated_rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

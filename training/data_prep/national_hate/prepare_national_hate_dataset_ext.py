import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_TO_LABEL = {-1: "NEG", 0: "NEU", 1: "POS",
                   "-1": "NEG", "0": "NEU", "1": "POS"}

LABELS = ("NEG", "NEU", "POS", "UNK")
TEXT_LABEL_PRIORITY = ("NEG", "POS", "NEU", "UNK")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?b>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_source_label(value):
    return SOURCE_TO_LABEL.get(value)


def majority(values, *, ignore_empty: bool = False):
    filtered = [value for value in values if not (ignore_empty and value == "")]
    if not filtered:
        return "missing"

    counts = Counter(filtered)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"
    return top[0][0]


def aggregate_instance(rows, *, min_agree: int, skip_unk: bool):
    sense = majority([row.get("does_text_make_sense", "") for row in rows])
    if sense != "yes":
        return None, f"drop_sense_{sense}"

    has_ethnonyms = majority([row.get("text_has_ethnonyms", "") for row in rows], ignore_empty=True)
    
    if has_ethnonyms != "yes":
        return None, f"drop_ethnonyms_{has_ethnonyms}"

    labels = [normalize_source_label(row.get("class")) for row in rows]
    labels = [label for label in labels if label is not None]

    first = rows[0]
    text = clean_text(first["text"])
    if not text:
        return None, "drop_empty_text"

    if not labels:
        if skip_unk:
            return None, "drop_no_valid_label"

        return {
            "text_id": first["text_id"],
            "instance_id": first["instance_id"],
            "text": text,
            "target": first["ethnic_group"],
            "toxicity": "UNK",
            "source_class": None,
            "votes": {},
            "n_valid_votes": 0,
            "n_assessors": len(rows),
            "agreement_votes": 0,
            "agreement_ratio": 0.0,
            "assessors": sorted(row.get("assessor") for row in rows),
        }, "ok_unlabeled"

    votes = Counter(labels)
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, "drop_class_tie"

    label, top_votes = top[0]
    if top_votes < min_agree:
        return None, "drop_not_enough_agreement"

    return {
        "text_id": first["text_id"],
        "instance_id": first["instance_id"],
        "text": text,
        "target": first["ethnic_group"],
        "toxicity": label,
        "source_class": {"NEG": -1, "NEU": 0, "POS": 1}[label],
        "votes": dict(votes),
        "n_valid_votes": len(labels),
        "n_assessors": len(rows),
        "agreement_votes": top_votes,
        "agreement_ratio": round(top_votes / len(labels), 6),
        "assessors": sorted(row.get("assessor") for row in rows),
    }, "ok"


def aggregate_instances(source_rows, *, min_agree: int, skip_unk: bool):
    by_instance = defaultdict(list)
    for row in source_rows:
        by_instance[row["instance_id"]].append(row)

    aggregated = []
    drop_reasons = Counter()
    for rows in by_instance.values():
        item, status = aggregate_instance(rows, min_agree=min_agree, skip_unk=skip_unk)
        if item is None:
            drop_reasons[status] += 1
            continue
        aggregated.append(item)

    return aggregated, drop_reasons


def aggregate_text_label(labels):
    labels = set(labels)
    for label in TEXT_LABEL_PRIORITY:
        if label in labels:
            return label
    raise ValueError(f"Unexpected empty label set: {labels}")


def aggregate_texts(instance_rows):
    by_text = defaultdict(list)
    for row in instance_rows:
        by_text[row["text"]].append(row)

    result = []
    for sentence_id, rows in enumerate(by_text.values(), start=1):
        toxicity = aggregate_text_label(row["toxicity"] for row in rows)
        targets = sorted({row["target"] for row in rows})
        targets_by_stance = {
            label: sorted({row["target"] for row in rows if row["toxicity"] == label})
            for label in LABELS
        }
        target_labels = [
            {
                "target": row["target"],
                "toxicity": row["toxicity"],
                "source_class": row["source_class"],
                "votes": row["votes"],
                "agreement_ratio": row["agreement_ratio"],
                "n_valid_votes": row["n_valid_votes"],
                "n_assessors": row["n_assessors"],
                "instance_id": row["instance_id"],
            }
            for row in sorted(rows, key=lambda item: item["target"])
        ]

        result.append(
            {
                "sentence_id": sentence_id,
                "text": rows[0]["text"],
                "target": ", ".join(targets),
                "ethnic_groups": targets,
                "toxicity": toxicity,
                "targets_by_stance": targets_by_stance,
                "target_labels": target_labels,
                "text_ids": sorted({row["text_id"] for row in rows}),
                "n_targets": len(targets),
                "has_mixed_target_labels": len({row["toxicity"] for row in rows}) > 1,
            }
        )

    return result


def label_counts(rows):
    return dict(Counter(row["toxicity"] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/national_hate/RuEthnoHateExtended.json"))
    parser.add_argument("--output", type=Path, default=Path("data/national_hate/national_hate_aggregated.jsonl"))
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--skip-unk", action="store_true")
    args = parser.parse_args()

    source_rows = load_json(args.input)
    instance_rows, drop_reasons = aggregate_instances(
        source_rows,
        min_agree=args.min_agree,
        skip_unk=args.skip_unk,
    )
    text_rows = aggregate_texts(instance_rows)
    save_jsonl(args.output, text_rows)

    summary = {
        "source_rows": len(source_rows),
        "aggregated_instances": len(instance_rows),
        "aggregated_texts": len(text_rows),
        "min_agree": args.min_agree,
        "skip_unk": args.skip_unk,
        "drop_reasons": dict(drop_reasons),
        "labels": label_counts(text_rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

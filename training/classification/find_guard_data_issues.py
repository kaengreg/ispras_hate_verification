import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                rows.append((line_number, json.loads(line)))
    return rows


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def record_key(record):
    return normalize(record.get("text", "")), normalize(record.get("target", ""))


def group_id(key):
    return hashlib.sha1("\0".join(key).encode("utf-8")).hexdigest()[:12]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def mark_record(record, issue, split, line_number, key, group_size, labels, label_conflict):
    marked = dict(record)
    marked["_data_issue"] = {
        "type": issue,
        "split": split,
        "source_line": line_number,
        "group_id": group_id(key),
        "group_size": group_size,
        "labels": sorted(labels),
        "label_conflict": label_conflict,
    }
    return marked


def main(args):
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)

    train_groups = defaultdict(list)
    val_groups = defaultdict(list)
    for line_number, record in train_rows:
        train_groups[record_key(record)].append((line_number, record))
    for line_number, record in val_rows:
        val_groups[record_key(record)].append((line_number, record))

    duplicate_rows = []
    duplicate_keys = {key for key, group in train_groups.items() if len(group) > 1}
    for key in sorted(duplicate_keys):
        group = train_groups[key]
        labels = {record.get("stance") for _, record in group}
        for line_number, record in group:
            duplicate_rows.append(mark_record(record, "train_duplicate", "train", line_number, key, len(group), labels, len(labels) > 1))

    overlap_rows = []
    overlap_keys = set(train_groups) & set(val_groups)
    for key in sorted(overlap_keys):
        train_group = train_groups[key]
        val_group = val_groups[key]
        train_labels = {record.get("stance") for _, record in train_group}
        val_labels = {record.get("stance") for _, record in val_group}
        labels = train_labels | val_labels
        label_conflict = not bool(train_labels & val_labels)
        group_size = len(train_group) + len(val_group)
        for split, group in (("train", train_group), ("val", val_group)):
            for line_number, record in group:
                marked = mark_record(record, "train_val_overlap", split, line_number, key, group_size, labels, label_conflict)
                marked["_data_issue"]["train_labels"] = sorted(train_labels)
                marked["_data_issue"]["val_labels"] = sorted(val_labels)
                overlap_rows.append(marked)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicates_path = output_dir / "TRAIN_duplicates_highlighted.jsonl"
    overlaps_path = output_dir / "TRAIN_VAL_overlaps_highlighted.jsonl"
    summary_path = output_dir / "data_issues_summary.json"

    write_jsonl(duplicates_path, duplicate_rows)
    write_jsonl(overlaps_path, overlap_rows)

    summary = {
        "normalization": "lowercase, strip, collapse whitespace; grouped by (text, target)",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_duplicate_groups": len(duplicate_keys),
        "train_rows_in_duplicate_groups": len(duplicate_rows),
        "train_duplicate_groups_with_label_conflicts": sum(len({record.get('stance') for _, record in train_groups[key]}) > 1 for key in duplicate_keys),
        "train_val_overlap_groups": len(overlap_keys),
        "train_val_rows_in_overlap_groups": len(overlap_rows),
        "train_val_overlap_groups_with_label_conflicts": sum(not ({record.get('stance') for _, record in train_groups[key]} & {record.get('stance') for _, record in val_groups[key]}) for key in overlap_keys),
        "outputs": {"duplicates": str(duplicates_path), "overlaps": str(overlaps_path)}
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/guard_train/RuStance2.1_15_07_TRAIN_no_duplicates.jsonl")
    parser.add_argument("--val", default="data/guard_train/RuStance2.1_15_07_VAL_no_duplicates.jsonl")
    parser.add_argument("--output-dir", default="data/guard_train/diagnostics")
    main(parser.parse_args())

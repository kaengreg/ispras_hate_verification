import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def prepare_test(input_path, output_path):
    rows = []
    groups = {}
    counts = Counter()
    with input_path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{input_path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{number}: expected an object")
            if "moderation" in row:
                raise ValueError(f"{input_path}:{number}: use test data before inference, without moderation results")
            task = row.get("task")
            if task not in ("general_toxicity", "targeted_toxicity"):
                raise ValueError(f"{input_path}:{number}: unsupported task {task!r}")
            if row.get("label") not in ("NEU", "TOX", "UNK"):
                raise ValueError(f"{input_path}:{number}: expected label NEU, TOX or UNK")
            for key in ("source", "text"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{input_path}:{number}: expected nonempty {key}")
            counts[task] += 1
            counts["input_unknown_gold"] += row["label"] == "UNK"
            if task == "general_toxicity":
                rows.append(row)
                continue
            target = row.get("target")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"{input_path}:{number}: expected a nonempty target")
            key = (row["source"], row["text"])
            if key not in groups:
                output = {**row, "target": None, "label": "NEU", "target_annotations": []}
                identity = json.dumps([row["source"], task, row["text"], "text_level/v1"], ensure_ascii=False)
                output["example_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                output["prompt_id"] = f"{row['source']}/{task}/text_level/v1"
                groups[key] = {"row": output, "labels": set(), "targets": {}}
                rows.append(output)
            group = groups[key]
            previous = group["targets"].get(target)
            if previous is not None and previous != row["label"]:
                raise ValueError(f"{input_path}:{number}: conflicting labels for the same (source, text, target)")
            group["targets"][target] = row["label"]
            group["labels"].add(row["label"])
            group["row"]["target_annotations"].append(
                {key: row[key] for key in ("target", "label", "example_id", "prompt_id", "source_file",
                                          "source_line", "sentence_id", "text_ids") if key in row})

    if not rows:
        raise ValueError(f"{input_path}: input file is empty")
    for group in groups.values():
        labels = group["labels"]
        group["row"]["label"] = "TOX" if "TOX" in labels else ("UNK" if "UNK" in labels else "NEU")
    with output_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"general_toxicity: {counts['general_toxicity']} -> {counts['general_toxicity']}")
    print(f"targeted_toxicity: {counts['targeted_toxicity']} -> {len(groups)}")
    print(f"Total: {sum(counts[task] for task in ('general_toxicity', 'targeted_toxicity'))} -> {len(rows)}")
    print(f"Unknown gold: {counts['input_unknown_gold']} input rows; {sum(r['label'] == 'UNK' for r in rows)} output examples")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate target annotations into text-level test examples before inference")
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    prepare_test(args.input, args.output)


if __name__ == "__main__":
    main()

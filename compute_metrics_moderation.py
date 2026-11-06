import argparse
import csv
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


LABELS = ("NEU", "TOX")
CRITERIA = ("hate_speech", "toxicity")
TASKS = ("general_toxicity", "targeted_toxicity")


def extract_json(text: str) -> Dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
    except ValueError:
        left, right = text.find("{"), text.rfind("}")
        if left == -1 or right <= left:
            return None
        try:
            obj = json.loads(text[left:right + 1])
        except ValueError:
            return None
    return obj if isinstance(obj, dict) else None


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "UNK"
    label = str(value if value is not None else "").strip().upper()
    # Numeric labels retain the previous NEG/NEU/POS convention, not binary 0/1.
    return {"TOX": "TOX", "NEU": "NEU", "NEG": "TOX", "POS": "NEU",
            "-1": "TOX", "0": "NEU", "1": "NEU",
            "VIOLATION": "TOX", "FAIL": "TOX", "PASS": "NEU"}.get(label, "UNK")


def extract_pred_verdict(result):
    for key in ("verdict", "raw_verdict"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = result.get("raw")
    obj = extract_json(raw)
    if obj is not None:
        value = obj.get("verdict")
        if isinstance(value, str) and value.strip():
            return value.strip()
    # SFT models can return a single label instead of JSON.
    if isinstance(raw, str) and normalize_label(raw) != "UNK":
        return raw.strip()
    return "UNK"


def extract_prediction(item, criterion):
    moderation = item.get("moderation")
    if not isinstance(moderation, dict):
        return "UNK", "UNK", "no_moderation"
    results = moderation.get("results")
    if not isinstance(results, dict):
        return "UNK", "UNK", "no_results"
    result = results.get(criterion)
    if not isinstance(result, dict):
        return "UNK", "UNK", "no_criterion"
    raw = extract_pred_verdict(result)
    label = normalize_label(raw)
    return label, raw, "unk_verdict" if label == "UNK" else ""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def combine_predictions(hate_speech, toxicity):
    if "TOX" in (hate_speech, toxicity):
        return "TOX"
    return "NEU" if hate_speech == toxicity == "NEU" else "UNK"


def prepare_examples(data):
    general = []
    targeted = []
    seen_targeted = set()
    counts = Counter()
    diagnostics = {criterion: Counter() for criterion in CRITERIA}
    for number, row in enumerate(data, 1):
        task = row.get("task")
        if task not in TASKS:
            counts["excluded_other_or_missing_task"] += 1
            continue
        counts[task] += 1
        gold = normalize_label(row.get("label"))
        if gold == "UNK":
            counts[f"{task}_unknown_gold_rows"] += 1
        predictions = {}
        for criterion in (("toxicity",) if task == "general_toxicity" else CRITERIA):
            pred, raw, reason = extract_prediction(row, criterion)
            predictions[criterion] = pred
            diagnostics[criterion][f"{task}/raw/{raw}"] += 1
            if reason:
                diagnostics[criterion][f"{task}/missing/{reason}"] += 1
        if task == "general_toxicity":
            general.append({"gold": gold, **predictions})
            continue
        for key in ("source", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Record {number}: targeted_toxicity requires nonempty {key}")
        key = (row["source"], row["text"])
        if key in seen_targeted:
            raise ValueError(f"Record {number}: repeated targeted text; prepare the test with "
                             "training/data_prep/toxicity/prepare_evaluation_test.py before inference")
        seen_targeted.add(key)
        predictions["hate_speech OR toxicity"] = combine_predictions(predictions["hate_speech"], predictions["toxicity"])
        targeted.append({"gold": gold, **predictions})
    return general, targeted, counts, diagnostics


def calculate_metrics(examples, detector, skip_unk):
    valid_gold = [e for e in examples if e["gold"] != "UNK"]
    unknown_pred = sum(e[detector] == "UNK" for e in valid_gold)
    evaluated = [e for e in valid_gold if not skip_unk or e[detector] != "UNK"]
    result = {"total": len(examples), "unknown_gold": len(examples) - len(valid_gold),
              "unknown_pred": unknown_pred, "evaluated": len(evaluated),
              "gold_counts": dict(Counter(e["gold"] for e in examples)),
              "pred_counts": dict(Counter(e[detector] for e in examples))}
    if not evaluated:
        return result
    y_true = [e["gold"] for e in evaluated]
    y_pred = [e[detector] for e in evaluated]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["TOX"], zero_division=0)
    result.update(accuracy=accuracy_score(y_true, y_pred), precision=precision[0], recall=recall[0], f1=f1[0],
                  macro_f1=precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS),
                                                         average="macro", zero_division=0)[2],
                  classification_report=classification_report(y_true, y_pred, labels=list(LABELS),
                                                             digits=4, zero_division=0),
                  matrix=confusion_matrix(y_true, y_pred, labels=["NEU", "TOX", "UNK"])[:2])
    return result


def print_details(task, detector, metrics, skip_unk, counts, diagnostics):
    print(f"\n--- {task} / {detector} ---")
    print("\n--- DATASET & PREDICTION STATISTICS ---")
    print(f"Loaded subset rows: {counts[task]}")
    print(f"Total samples: {metrics['total']}")
    for title, key in (("Gold label distribution", "gold_counts"),
                       ("Predicted label distribution (mapped where possible)", "pred_counts")):
        print(f"\n{title}:")
        for label in (*LABELS, "UNK"):
            print(f"  {label}: {metrics[key].get(label, 0)}")
    for criterion in (CRITERIA if detector == "hate_speech OR toxicity" else (detector,)):
        raw_prefix = f"{task}/raw/"
        raw_counts = {key[len(raw_prefix):]: value for key, value in diagnostics[criterion].items()
                      if key.startswith(raw_prefix)}
        print(f"\nPredicted raw verdict distribution ({criterion}):")
        for raw, count in sorted(raw_counts.items(), key=lambda pair: (-pair[1], pair[0])):
            print(f"  {raw}: {count}")
        print(f"\nMissing / invalid prediction reasons ({criterion}):")
        for reason in ("no_moderation", "no_results", "no_criterion", "unk_verdict"):
            print(f"  {reason}: {diagnostics[criterion][f'{task}/missing/{reason}']}")
    print(f"\nExcluded unknown gold: {metrics['unknown_gold']}")
    print(f"Unknown predictions with valid gold: {metrics['unknown_pred']}")
    print(f"Excluded unknown predictions: {metrics['unknown_pred'] if skip_unk else 0}")
    print(f"Evaluated: {metrics['evaluated']}")
    if not metrics["evaluated"]:
        print("N/A: no examples left for evaluation")
        return
    print("\n--- BINARY (positive=TOX, negative=NEU) ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred), rows: ['NEU', 'TOX'], cols: ['NEU', 'TOX', 'UNK']")
    print(metrics["matrix"])
    print("\nClassification report (macro average over NEU and TOX):")
    print(metrics["classification_report"])


def print_tsv(summary, model, input_path, skip_unk):
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(("model", "input", "task", "detector", "n", "accuracy", "precision_tox", "recall_tox",
                     "f1_tox", "macro_f1", "unknown_gold", "unknown_pred", "unknown_policy"))
    for task, detector, metrics in summary:
        values = [f"{metrics[key]:.4f}" if metrics["evaluated"] else "NA"
                  for key in ("accuracy", "precision", "recall", "f1", "macro_f1")]
        writer.writerow((model, str(input_path), task, detector, metrics["evaluated"], *values,
                         metrics["unknown_gold"], metrics["unknown_pred"], "skip" if skip_unk else "error"))


def report(data, input_path, skip_unk, output_format="text"):
    general, targeted, counts, diagnostics = prepare_examples(data)
    if not general and not targeted:
        raise SystemExit("No supported tasks found. Expected task=general_toxicity or targeted_toxicity.")
    models = set()
    for row in data:
        moderation = row.get("moderation")
        if isinstance(moderation, dict) and moderation.get("model"):
            models.add(str(moderation["model"]))
    model = "; ".join(sorted(models)) or "UNK"
    summary = []
    for task, examples, detectors in (
        ("general_toxicity", general, ("toxicity",)),
        ("targeted_toxicity", targeted, (*CRITERIA, "hate_speech OR toxicity")),
    ):
        for detector in detectors:
            summary.append((task, detector, calculate_metrics(examples, detector, skip_unk)))
    if output_format == "tsv":
        print_tsv(summary, model, input_path, skip_unk)
        return
    print(f"Evaluation of results produced by: {model}")
    print(f"File: {input_path}")
    print(f"Loaded: {len(data)}")
    print(f"general_toxicity: {counts['general_toxicity']} rows")
    print(f"targeted_toxicity: {counts['targeted_toxicity']} rows")
    print(f"Excluded other/missing task: {counts['excluded_other_or_missing_task']}")
    for task, detector, metrics in summary:
        print_details(task, detector, metrics, skip_unk, counts, diagnostics)


def main():
    parser = argparse.ArgumentParser(description="Evaluate toxicity on general labels and three detectors on targeted labels.")
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("--skip-unk", action="store_true", help="Exclude unknown predictions instead of counting them as errors")
    parser.add_argument("--tofile", action="store_true", help="Write the report next to the input file")
    parser.add_argument("--format", choices=("text", "tsv"), default="text", help="Full report or only a spreadsheet table")
    parser.add_argument("--results", default=None, help="Report filename (used with --tofile)")
    args = parser.parse_args()
    data = load_jsonl(args.input)
    if not data:
        parser.error("Input file is empty")
    if args.tofile:
        extension = "tsv" if args.format == "tsv" else "txt"
        output = args.input.resolve().parent / (args.results or f"metrics_moderation.{extension}")
        if output.resolve() == args.input.resolve():
            parser.error("Results file must differ from input file")
        with output.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            report(data, args.input, args.skip_unk, args.format)
        print(f"Report saved: {output}")
    else:
        report(data, args.input, args.skip_unk, args.format)


if __name__ == "__main__":
    main()



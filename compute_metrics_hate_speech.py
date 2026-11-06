import argparse
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


LABELS = ("NEU", "TOX")
TASKS = {"hate_speech": "targeted_toxicity", "toxicity": "general_toxicity"}


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


def extract_true_label(item):
    for key in ("label", "toxicity_label", "toxicity", "stance", "class"):
        if key in item:
            return normalize_label(item[key])
    return "UNK"


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


def prepare_examples(data, criterion):
    task = TASKS[criterion]
    selected = [row for row in data if row.get("task") in (None, task)]
    examples = []
    groups = {}
    for row in selected:
        gold = extract_true_label(row)
        pred, raw, reason = extract_prediction(row, criterion)
        example = {"gold": gold, "pred": pred, "raw": raw, "reason": reason}
        if criterion != "hate_speech" or row.get("task") != "targeted_toxicity":
            examples.append(example)
            continue
        text = row.get("text")
        target = row.get("target")
        if not isinstance(text, str) or not text.strip() or not isinstance(target, str) or not target.strip():
            raise ValueError(f"targeted_toxicity requires nonempty text and target: {row.get('example_id')}")
        # Keep independent source annotations separate, including cross-source disagreements.
        key = (row.get("source"), text)
        if key not in groups:
            groups[key] = {"labels": [], "predictions": set(), "example": example}
        group = groups[key]
        group["labels"].append(gold)
        group["predictions"].add(pred)
    inconsistent = 0
    for group in groups.values():
        example = group["example"]
        labels = group["labels"]
        example["gold"] = "UNK" if "UNK" in labels else ("TOX" if "TOX" in labels else "NEU")
        # Repeated requests for different targets may disagree despite having identical input text.
        if len(group["predictions"]) > 1:
            example.update(pred="UNK", raw="INCONSISTENT", reason="inconsistent_predictions")
            inconsistent += 1
        examples.append(example)
    return examples, len(selected), inconsistent


def report(data, criterion, skip_unk):
    examples, selected, inconsistent = prepare_examples(data, criterion)
    print(f"Criterion: {criterion}; task: {TASKS[criterion]}")
    print(f"Loaded rows: {len(data)}; selected rows: {selected}; other tasks excluded: {len(data) - selected}")
    print(f"Evaluation examples: {len(examples)}")
    if criterion == "hate_speech":
        print("Target annotations aggregated by (source, text): TOX if any annotated target is TOX, otherwise NEU.")
        print("Assumption: annotated targets cover all relevant groups. Sources remain separate.")
        print(f"Rows collapsed: {selected - len(examples)}; groups with inconsistent predictions: {inconsistent}")
    print("Gold labels:", dict(Counter(e["gold"] for e in examples)))
    print("Predicted labels:", dict(Counter(e["pred"] for e in examples)))
    print("Raw verdicts:", dict(Counter(e["raw"] for e in examples)))
    print("Missing / invalid prediction reasons:", dict(Counter(e["reason"] for e in examples if e["reason"])))
    invalid_gold = sum(e["gold"] == "UNK" for e in examples)
    invalid_pred = sum(e["gold"] != "UNK" and e["pred"] == "UNK" for e in examples)
    print(f"Excluded unknown gold: {invalid_gold}")
    print(f"Unknown predictions with valid gold: {invalid_pred}")
    evaluated = [e for e in examples if e["gold"] != "UNK" and (not skip_unk or e["pred"] != "UNK")]
    print(f"Evaluated: {len(evaluated)}; excluded unknown predictions: {invalid_pred if skip_unk else 0}")
    print("Unknown predictions are counted as errors." if not skip_unk else "Metrics are conditional on valid predictions (--skip-unk).")
    if not evaluated:
        raise SystemExit("No examples left for evaluation. Check task, gold labels and predictions.")
    y_true = [e["gold"] for e in evaluated]
    y_pred = [e["pred"] for e in evaluated]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["TOX"], zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)[2]
    print("\n--- BINARY (positive=TOX/violation, negative=NEU/pass) ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision TOX: {precision[0]:.4f}")
    print(f"Recall TOX:    {recall[0]:.4f}")
    print(f"F1 TOX:        {f1[0]:.4f}")
    print(f"Macro-F1 (NEU, TOX): {macro:.4f}")
    labels = list(LABELS) + (["UNK"] if "UNK" in y_pred else [])
    print("\nConfusion matrix (rows=true, cols=pred), labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nClassification report (macro average over NEU and TOX):")
    print(classification_report(y_true, y_pred, labels=list(LABELS), digits=4, zero_division=0))


def main(default_criterion="hate_speech"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--criterion", choices=tuple(TASKS), default=default_criterion)
    parser.add_argument("--skip-unk", action="store_true")
    parser.add_argument("--tofile", action="store_true")
    parser.add_argument("--results", default=None)
    args = parser.parse_args()
    data = load_jsonl(args.input)
    if not data:
        raise SystemExit("Input file is empty.")
    if args.tofile:
        output = Path(args.input).resolve().parent / (args.results or f"metrics_{args.criterion}.txt")
        if output.resolve() == Path(args.input).resolve():
            parser.error("Results file must differ from input file")
        with output.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            report(data, args.criterion, args.skip_unk)
    else:
        report(data, args.criterion, args.skip_unk)


if __name__ == "__main__":
    main()

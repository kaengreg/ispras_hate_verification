import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


LABELS = ("NEG", "NEU", "POS")
CLASS_TO_LABEL = {-1: "NEG", 0: "NEU", 1: "POS", "-1": "NEG", "0": "NEU", "1": "POS"}
POSITIVE_LABELS = {"NEU", "POS"}
NEGATIVE_LABELS = {"NEG"}


def extract_json(text: str) -> Dict[str, Any] | None:
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return None

    try:
        obj = json.loads(text[left:right + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def normalize_label(value: Any) -> str:
    if isinstance(value, (int, str)) and not isinstance(value, bool) and value in CLASS_TO_LABEL:
        return CLASS_TO_LABEL[value]

    label = str(value if value is not None else "").strip().upper()
    if label in LABELS:
        return label
    return "UNK"


def extract_true_label(item: Dict[str, Any]) -> str:
    for key in ("toxicity", "stance", "class"):
        if key in item:
            return normalize_label(item[key])
    return "UNK"


def extract_pred_verdict(criterion_result: Dict[str, Any]) -> str:
    verdict = criterion_result.get("verdict")
    if isinstance(verdict, str) and verdict.strip():
        return verdict.strip()

    raw = criterion_result.get("raw")
    if isinstance(raw, str) and raw.strip():
        obj = extract_json(raw)
        if isinstance(obj, dict):
            verdict = obj.get("verdict")
            if isinstance(verdict, str) and verdict.strip():
                return verdict.strip()

    return "UNK"


def extract_pred_true(item: Dict[str, Any], criterion_key: str) -> Tuple[str, str, str]:
    y_true = extract_true_label(item)

    moderation = item.get("moderation")
    if not isinstance(moderation, dict):
        return y_true, "UNK", "UNK"

    results = moderation.get("results")
    if not isinstance(results, dict):
        return y_true, "UNK", "UNK"

    criterion_result = results.get(criterion_key)
    if not isinstance(criterion_result, dict):
        return y_true, "UNK", "UNK"

    raw_verdict = extract_pred_verdict(criterion_result)
    return y_true, normalize_label(raw_verdict), raw_verdict


def label_to_binary(label: str) -> int:
    label = normalize_label(label)
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    return -1


def verdict_to_binary(verdict: str) -> int:
    label = normalize_label(verdict)
    if label != "UNK":
        return label_to_binary(label)

    verdict = str(verdict or "").strip().lower()
    if verdict == "pass":
        return 1
    if verdict in {"violation", "fail"}:
        return 0
    return -1


def collect_stats(data: List[Dict[str, Any]], criterion_key: str) -> Dict[str, Any]:
    stats = {
        "total": len(data),
        "gold": {label: 0 for label in (*LABELS, "UNK")},
        "pred_verdict": {},
        "pred_label": {label: 0 for label in (*LABELS, "UNK")},
        "missing": {
            "no_moderation": 0,
            "no_results": 0,
            "no_criterion": 0,
            "unk_verdict": 0,
            "unk_verdicts": set(),
        },
    }

    for item in data:
        gold = extract_true_label(item)
        stats["gold"][gold] += 1

        moderation = item.get("moderation")
        if not isinstance(moderation, dict):
            stats["missing"]["no_moderation"] += 1
            stats["pred_label"]["UNK"] += 1
            continue

        results = moderation.get("results")
        if not isinstance(results, dict):
            stats["missing"]["no_results"] += 1
            stats["pred_label"]["UNK"] += 1
            continue

        criterion_result = results.get(criterion_key)
        if not isinstance(criterion_result, dict):
            stats["missing"]["no_criterion"] += 1
            stats["pred_label"]["UNK"] += 1
            continue

        raw_verdict = extract_pred_verdict(criterion_result)
        stats["pred_verdict"][raw_verdict] = stats["pred_verdict"].get(raw_verdict, 0) + 1
        pred_label = normalize_label(raw_verdict)
        if pred_label == "UNK":
            stats["missing"]["unk_verdict"] += 1
            stats["missing"]["unk_verdicts"].add(raw_verdict)
        stats["pred_label"][pred_label] += 1

    stats["missing"]["unk_verdicts"] = sorted(stats["missing"]["unk_verdicts"])
    return stats


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--criterion", default="hate_speech")
    parser.add_argument("--skip-unk", action="store_true")
    parser.add_argument("--tofile", action="store_true")
    parser.add_argument("--results", default="metrics_hate_speech.txt")
    args = parser.parse_args()

    output_file = None
    if args.tofile:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), args.results)
        output_file = open(output_path, "w", encoding="utf-8")
        sys.stdout = output_file

    data = load_jsonl(args.input)
    if not data:
        if output_file is not None:
            sys.stdout = sys.__stdout__
            output_file.close()
        raise SystemExit("Input file is empty.")

    stats = collect_stats(data, args.criterion)
    model = next((item.get("moderation", {}).get("model") for item in data if isinstance(item.get("moderation"), dict)), "UNK")

    print(f"Evaluation of results produced by: {model}")
    print(f"Loaded: {len(data)}")

    print("\n--- DATASET & PREDICTION STATISTICS ---")
    print(f"Total samples: {stats['total']}")
    print("\nGold label distribution:")
    for label, count in stats["gold"].items():
        print(f"  {label}: {count}")

    print("\nPredicted label distribution:")
    for label, count in stats["pred_label"].items():
        print(f"  {label}: {count}")

    print("\nPredicted raw verdict distribution:")
    for verdict, count in sorted(stats["pred_verdict"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {verdict}: {count}")

    print("\nMissing / skipped prediction reasons:")
    for reason, count in stats["missing"].items():
        print(f"  {reason}: {count}")

    y_true_mc, y_pred_mc = [], []
    y_true_bin, y_pred_bin = [], []
    skipped_mc = 0
    skipped_bin = 0

    for item in data:
        true_label, pred_label, raw_verdict = extract_pred_true(item, args.criterion)

        true_binary = label_to_binary(true_label)
        pred_binary = verdict_to_binary(raw_verdict)
        if true_binary == -1 or pred_binary == -1:
            skipped_bin += 1
        else:
            y_true_bin.append(true_binary)
            y_pred_bin.append(pred_binary)

        if args.skip_unk and (true_label == "UNK" or pred_label == "UNK"):
            skipped_mc += 1
            continue

        y_true_mc.append(true_label)
        y_pred_mc.append(pred_label)

    print(f"\nCriterion: {args.criterion}")
    print(f"Evaluated multiclass: {len(y_true_mc)}")
    print(f"Skipped multiclass: {skipped_mc}")
    print(f"Evaluated binary: {len(y_true_bin)}")
    print(f"Skipped binary: {skipped_bin}")

    if not y_true_mc:
        if output_file is not None:
            sys.stdout = sys.__stdout__
            output_file.close()
        raise SystemExit("No examples left for evaluation. Check that the input contains moderation predictions or remove --skip-unk.")

    labels_for_report = list(LABELS)
    if "UNK" in y_true_mc or "UNK" in y_pred_mc:
        labels_for_report.append("UNK")

    print("\n--- MULTICLASS (NEG/NEU/POS) ---")
    print("Accuracy:", accuracy_score(y_true_mc, y_pred_mc))
    print("\nConfusion matrix (rows=true, cols=pred), labels:", labels_for_report)
    print(confusion_matrix(y_true_mc, y_pred_mc, labels=labels_for_report))
    print("\nClassification report:")
    print(classification_report(y_true_mc, y_pred_mc, labels=labels_for_report, zero_division=0))

    if y_true_bin:
        print("\n--- BINARY (positive=NEU+POS, negative=NEG) ---")
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary", pos_label=1, zero_division=0)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred), labels: [0=NEG, 1=NEU+POS]")
        print(confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]))
    else:
        print("\n--- BINARY ---")
        print("No examples with valid binary labels.")

    if output_file is not None:
        sys.stdout = sys.__stdout__
        output_file.close()


if __name__ == "__main__":
    main()

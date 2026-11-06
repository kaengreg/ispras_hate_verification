import json 
import os
import argparse 
from typing import Any, Dict, Tuple, List 
from contextlib import redirect_stdout

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix


STANCE_LABELS = {"ANTI", "NEU", "PRO"}
POSITIVE_STANCES = {"PRO", "NEU"}  
NEGATIVE_STANCES = {"ANTI"}         

PASS_VIOLATION = {"pass", "violation", "fail"}  # "fail" in older saved results


def extract_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return None
    
    s = text.strip()
    if not s:
        return None
    
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    left = s.find('{')
    right = s.rfind('}')
    if left == -1 or right == -1 or right <= left:
        return None
    try:
        data = json.loads(s[left:right+1])
        if isinstance(data, dict):
            return data 
    except Exception:
        return None 
    return None 


def extract_pred_verdict(criterion_res: Dict[str, Any]) -> str:
    verdict = criterion_res.get('raw_verdict')
    if isinstance(verdict, str) and verdict.strip():
        return verdict.strip()
    raw = criterion_res.get('raw')
    if isinstance(raw, str) and raw.strip():
        obj = extract_json(raw)
        if isinstance(obj, dict):
            verdict = obj.get('verdict')
            if isinstance(verdict, str) and verdict.strip():
                return verdict.strip()
        if raw.strip().upper() in STANCE_LABELS or raw.strip().lower() in PASS_VIOLATION:
            return raw.strip()

    verdict = criterion_res.get('verdict')
    if isinstance(verdict, str) and verdict.strip():
        return verdict.strip()

    return "UNK"


def verdict_to_stance(verdict: str) -> str:
    if not isinstance(verdict, str):
        return "UNK"
    
    verdict = verdict.strip()
    if not verdict:
        return "UNK"

    verdict = verdict.upper()
    if verdict in STANCE_LABELS:
        return verdict

    verdict = verdict.lower()
    if verdict in {"violation", "fail"}:
        return "ANTI"
    if verdict == "pass":
        return "UNK"

    return "UNK"


def verdict_to_binary(verdict: str) -> int:
    if not isinstance(verdict, str):
        return -1
    verdict = verdict.strip()
    if not verdict:
        return -1

    verdict = verdict.upper()
    if verdict == "ANTI":
        return 0
    if verdict in {"PRO", "NEU"}:
        return 1

    verdict = verdict.lower()
    if verdict in {"violation", "fail"}:
        return 0
    if verdict == "pass":
        return 1
    return -1


def extract_pred_true(item: Dict[str, Any], criterion_key: str) -> Tuple[str, str, str]:
    y_true = str(item.get('stance') or 'UNK').strip().upper() or 'UNK'

    moderation = item.get('moderation')
    if not isinstance(moderation, dict):
        return y_true, "UNK", "UNK"

    results = moderation.get('results')
    if not isinstance(results, dict):
        return y_true, "UNK", "UNK"

    criterion_res = results.get(criterion_key)
    if not isinstance(criterion_res, dict):
        return y_true, "UNK", "UNK"

    raw_verdict = extract_pred_verdict(criterion_res)
    y_pred = verdict_to_stance(raw_verdict)

    return y_true, y_pred, raw_verdict


def stance_to_binary_label(stance: str) -> int:
    stance = str(stance).strip().upper()
    if stance in POSITIVE_STANCES:
        return 1
    if stance in NEGATIVE_STANCES:
        return 0
    return -1


def collect_stats(data: List[Dict[str, Any]], criterion_key: str) -> Dict[str, Any]:
    stats = {
        "total": 0,
        "gold": {"ANTI": 0, "NEU": 0, "PRO": 0, "UNK": 0},
        "pred_verdict": {}, 
        "pred_stance": {"ANTI": 0, "NEU": 0, "PRO": 0, "UNK": 0},
        "missing": {
            "no_moderation": 0,
            "no_results": 0,
            "no_criterion": 0,
            "unk_verdict": 0,
            "unk_verdicts": set(),
        },
    }

    for item in data:
        stats["total"] += 1

        gold = str(item.get("stance") or "UNK").strip().upper() or "UNK"
        if gold not in stats["gold"]:
            gold = "UNK"
        stats["gold"][gold] += 1

        moderation = item.get("moderation")
        if not isinstance(moderation, dict):
            stats["missing"]["no_moderation"] += 1
            stats["pred_stance"]["UNK"] += 1
            continue

        results = moderation.get("results")
        if not isinstance(results, dict):
            stats["missing"]["no_results"] += 1
            stats["pred_stance"]["UNK"] += 1
            continue

        criterion = results.get(criterion_key)
        if not isinstance(criterion, dict):
            stats["missing"]["no_criterion"] += 1
            stats["pred_stance"]["UNK"] += 1
            continue

        raw_verdict = extract_pred_verdict(criterion)
        stats["pred_verdict"][raw_verdict] = stats["pred_verdict"].get(raw_verdict, 0) + 1

        pred_stance = verdict_to_stance(raw_verdict)
        if pred_stance == "UNK":
            stats["missing"]["unk_verdict"] += 1
            stats["missing"]["unk_verdicts"].add(raw_verdict)
        stats["pred_stance"][pred_stance] += 1

    stats["missing"]["unk_verdicts"] = sorted(list(stats["missing"]["unk_verdicts"]))
    return stats

def load_jsonl(path): 
    data = []
    with open(path, 'r', encoding='utf-8') as fin:
        for number, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            data.append(obj)

    return data

def print_generation_metrics(data, criterion_key):
    counts = {"ANTI": 0, "NEU": 0, "PRO": 0, "PASS_ONLY": 0, "UNK": 0}
    binary_counts = {0: 0, 1: 0, -1: 0}
    for item in data:
        _, stance, verdict = extract_pred_true(item, criterion_key)
        binary = verdict_to_binary(verdict)
        binary_counts[binary] += 1
        counts["PASS_ONLY" if stance == "UNK" and binary == 1 else stance] += 1
    total = len(data)
    valid = binary_counts[0] + binary_counts[1]
    print("\n--- GENERATED ANSWER EVALUATION (no reference labels) ---")
    print("Accuracy / precision / recall / F1: N/A without reference answer stance labels")
    print("Aggregation: one vote per input row; labels assigned by the moderation model")
    print(f"Total answers: {total}")
    for label, count in counts.items():
        print(f"  {label}: {count} ({count / total:.2%} of all answers)")
    print(f"Valid binary verdicts: {valid}/{total}")
    print(f"Unknown / missing binary verdicts: {binary_counts[-1]}")
    print("PASS_ONLY means pass without a recoverable NEU/PRO stance")
    print(f"ANTI rate (all answers): {binary_counts[0] / total:.4%}")
    print(f"PRO+NEU / pass rate (all answers): {binary_counts[1] / total:.4%}")
    if valid:
        print(f"ANTI rate (valid verdicts): {binary_counts[0] / valid:.4%}")
        print(f"PRO+NEU / pass rate (valid verdicts): {binary_counts[1] / valid:.4%}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--criterion", type=str, default="anti_russia")
    parser.add_argument("--skip-unk", action="store_true")
    parser.add_argument("--tofile", action="store_true")
    parser.add_argument("--results", default="metrics.txt")
    parser.add_argument("--mode", choices=("auto", "classification", "generation"), default="auto",
                        help="auto selects generation for model_answer records without reference stance labels")


    args = parser.parse_args()

    data = load_jsonl(args.input)
    if not data:
        parser.error("Input contains no records")
    if args.tofile:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        output_path = os.path.join(input_dir, args.results)
        if os.path.realpath(output_path) == os.path.realpath(args.input):
            parser.error("Output path must differ from input path")
        with open(output_path, 'w', encoding='utf-8') as output_file, redirect_stdout(output_file):
            report(data, args)
    else:
        report(data, args)


def report(data, args):

    stats = collect_stats(data, args.criterion)

    models = sorted({str(item['moderation']['model']) for item in data
                     if isinstance(item.get('moderation'), dict) and item['moderation'].get('model')})
    generators = sorted({str(item['answer_generation']['model']) for item in data
                         if isinstance(item.get('answer_generation'), dict) and item['answer_generation'].get('model')})
    print(f"Evaluator model: {'; '.join(models) or 'UNK'}")
    if generators:
        print(f"Answer generation model: {'; '.join(generators)}")
    print(f"Loaded: {len(data)}")

    print("\n--- DATASET & PREDICTION STATISTICS ---")
    print(f"Total samples: {stats['total']}")
    print("\nGold stance distribution:")
    for k, v in stats['gold'].items():
        print(f"  {k}: {v}")

    print("\nPredicted stance distribution (mapped where possible):")
    for k, v in stats['pred_stance'].items():
        print(f"  {k}: {v}")

    print("\nPredicted raw verdict distribution:")
    for k, v in sorted(stats['pred_verdict'].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    print("\nMissing / skipped prediction reasons:")
    for k, v in stats['missing'].items():
        print(f"  {k}: {v}")

    mode = args.mode
    if mode == "auto":
        mode = "generation" if stats['gold']['UNK'] == len(data) and all('model_answer' in x for x in data) else "classification"
    if mode == "generation":
        print_generation_metrics(data, args.criterion)
        return

    y_true_mc, y_pred_mc = [], []
    y_true_bin, y_pred_bin = [], []

    unknown_gold = unknown_mc = unknown_bin = 0

    for item in data:
        true_stance, pred_stance, raw_verdict = extract_pred_true(item, args.criterion)

        if true_stance not in ("PRO", "NEU", "ANTI"):
            unknown_gold += 1
            continue
        if pred_stance == "UNK":
            unknown_mc += 1
        if not args.skip_unk or pred_stance != "UNK":
            y_true_mc.append(true_stance)
            y_pred_mc.append(pred_stance)

        true_bin = stance_to_binary_label(true_stance)
        pred_bin = verdict_to_binary(raw_verdict)
        if pred_bin == -1:
            unknown_bin += 1
        if not args.skip_unk or pred_bin != -1:
            y_true_bin.append(true_bin)
            y_pred_bin.append(pred_bin)

    print(f"Excluded unknown reference stance: {unknown_gold}")
    print(f"Unknown multiclass predictions with valid reference: {unknown_mc}")
    print(f"Unknown binary predictions with valid reference: {unknown_bin}")
    print(f"Unknown prediction policy: {'skip' if args.skip_unk else 'count as errors'}")
    print(f"Evaluated multiclass: {len(y_true_mc)}; binary: {len(y_true_bin)}")
    if not y_true_mc:
        print("Multiclass metrics: N/A, no evaluable reference/prediction pairs")
    else:
        print_classification_metrics(y_true_mc, y_pred_mc)
    if not y_true_bin:
        print("Binary metrics: N/A, no evaluable reference/prediction pairs")
        return
    print_binary_metrics(y_true_bin, y_pred_bin)


def print_classification_metrics(y_true_mc, y_pred_mc):

    labels_mc = ["ANTI", "NEU", "PRO", "UNK"]
    print("\n--- MULTICLASS (ANTI/NEU/PRO) ---")
    print("Accuracy:", accuracy_score(y_true_mc, y_pred_mc))
    print("\nConfusion matrix (rows=true, cols=pred), labels:", labels_mc)
    print(confusion_matrix(y_true_mc, y_pred_mc, labels=labels_mc))
    print("\nClassification report:")
    print(classification_report(y_true_mc, y_pred_mc, labels=labels_mc[:3], zero_division=0))


def print_binary_metrics(y_true_bin, y_pred_bin):

    # --- Binary metrics ---
    # Positive class = 1 (PRO+NEU), Negative = 0 (ANTI)
    print("\n--- BINARY (positive=PRO+NEU, negative=ANTI) ---")
    acc = accuracy_score(y_true_bin, y_pred_bin)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[1], average=None, zero_division=0
    )
    p, r, f1 = p[0], r[0], f1[0]
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred), labels: [0=ANTI, 1=PRO+NEU, -1=UNK]")
    print(confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1, -1]))


if __name__ == "__main__":
    main()

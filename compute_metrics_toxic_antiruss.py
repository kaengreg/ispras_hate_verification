import json 
import sys
import os
import argparse 
from typing import Any, Dict, Tuple, List 

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix


STANCE_LABELS = {"ANTI", "NEU", "PRO"}
POSITIVE_STANCES = {"PRO", "NEU"}  
NEGATIVE_STANCES = {"ANTI"}         

PASS_FAIL = {"pass", "fail"}

TOXICITY_LABELS = {"TOX", "NEU", "UNK"}


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
    right = s.find('}')
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
    raw = criterion_res.get('raw')
    if isinstance(raw, str) and raw.strip():
        obj = extract_json(raw)
        if isinstance(obj, dict):
            verdict = obj.get('verdict')
            if isinstance(verdict, str) and verdict.strip():
                return verdict.strip()

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
    if verdict == "fail":
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
    if verdict == "fail":
        return 0
    if verdict == "pass":
        return 1
    return -1


def extract_pred_true(item: Dict[str, Any], criterion_key: str) -> Tuple[str, str, str]:
    y_true = item.get('stance', 'UNK').strip().upper() or 'UNK'

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

        gold = item.get("stance", "UNK").strip().upper() or "UNK"
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

def collect_toxicity_by_stance(data: List[Dict[str, Any]], criterion_key: str) -> Dict[str, Dict[str, int]]:
    toxicity_by_stance = {
        "ANTI": {"TOX": 0, "NEU": 0, "UNK": 0},
        "NEU": {"TOX": 0, "NEU": 0, "UNK": 0},
        "PRO": {"TOX": 0, "NEU": 0, "UNK": 0},
        "UNK": {"TOX": 0, "NEU": 0, "UNK": 0},
    }

    for item in data:
        moderation = item.get("moderation")
        if not isinstance(moderation, dict):
            pred_stance = "UNK"
        else:
            results = moderation.get("results")
            if not isinstance(results, dict):
                pred_stance = "UNK"
            else:
                criterion_res = results.get(criterion_key)
                if not isinstance(criterion_res, dict):
                    pred_stance = "UNK"
                else:
                    raw_verdict = extract_pred_verdict(criterion_res)
                    pred_stance = verdict_to_stance(raw_verdict)

        toxicity = str(item.get("toxicity_label", "UNK")).strip().upper() or "UNK"

        if pred_stance not in toxicity_by_stance:
            pred_stance = "UNK"
        if toxicity not in TOXICITY_LABELS:
            toxicity = "UNK"

        toxicity_by_stance[pred_stance][toxicity] += 1

    return toxicity_by_stance

def load_jsonl(path): 
    data = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    return data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--criterion", type=str, default="anti_russia")
    parser.add_argument("--skip-unk", action="store_true")
    parser.add_argument("--tofile", action="store_true")
    parser.add_argument("--results", default="metrics.txt")


    args = parser.parse_args()

    output_file = None
    if args.tofile: 
        input_dir = os.path.dirname(os.path.abspath(args.input))
        output_path = os.path.join(input_dir, args.results)
        output_file = open(output_path, 'w', encoding='utf-8')
        sys.stdout = output_file


    data = load_jsonl(args.input)

    stats = collect_stats(data, args.criterion)
    toxicity_by_stance = collect_toxicity_by_stance(data, args.criterion)

    print(f"Evaluation of results produced by: {data[0]['moderation']['model']}")
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

    print("\n--- TOXICITY LABEL DISTRIBUTION BY PREDICTED STANCE ---")
    for stance in ["ANTI", "NEU", "PRO", "UNK"]:
        print(f"\nPredicted stance = {stance}:")
        total_in_stance = sum(toxicity_by_stance[stance].values())
        print(f"  Total: {total_in_stance}")
        for toxicity in ["TOX", "NEU", "UNK"]:
            print(f"  {toxicity}: {toxicity_by_stance[stance][toxicity]}")

    y_true_mc, y_pred_mc = [], []
    y_true_bin, y_pred_bin = [], []

    skipped = 0

    for item in data:
        true_stance, pred_stance, raw_verdict = extract_pred_true(item, args.criterion)

        if true_stance not in ("PRO", "NEU", "ANTI"):
            if args.skip_unk:
                skipped += 1
                continue
            true_stance = "UNK"
        if pred_stance not in ("PRO", "NEU", "ANTI"):
            if args.skip_unk:
                skipped += 1
                continue
        y_true_mc.append(true_stance)
        y_pred_mc.append(pred_stance)

        true_bin = stance_to_binary_label(true_stance)
        pred_bin = verdict_to_binary(raw_verdict)
        if true_bin == -1 or pred_bin == -1:
            skipped += 1
            continue
        y_true_bin.append(true_bin)
        y_pred_bin.append(pred_bin)

    if args.skip_unk:
        print(f"Skipped (unknown/missing labels): {skipped}")
    print(f"Evaluated: {len(y_true_mc)}")

    labels_mc = ["ANTI", "NEU", "PRO"]
    print("\n--- MULTICLASS (ANTI/NEU/PRO) ---")
    print("Accuracy:", accuracy_score(y_true_mc, y_pred_mc))
    print("\nConfusion matrix (rows=true, cols=pred), labels:", labels_mc)
    print(confusion_matrix(y_true_mc, y_pred_mc, labels=labels_mc))
    print("\nClassification report:")
    print(classification_report(y_true_mc, y_pred_mc, labels=labels_mc, zero_division=0))

    # --- Binary metrics ---
    # Positive class = 1 (PRO+NEU), Negative = 0 (ANTI)
    print("\n--- BINARY (positive=PRO+NEU, negative=ANTI) ---")
    acc = accuracy_score(y_true_bin, y_pred_bin)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", pos_label=1, zero_division=0
    )
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred), labels: [0=ANTI, 1=PRO+NEU]")
    print(confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]))

    if output_file is not None:
        sys.stdout = sys.__stdout__
        output_file.close()


if __name__ == "__main__":
    main()

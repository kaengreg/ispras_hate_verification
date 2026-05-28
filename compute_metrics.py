import json 
import sys
import os
import argparse 
from typing import Any, Dict, Tuple, List 

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

TONALITY_TO_STANCE = {"POS": "PRO", 
                      "NEU": "NEU",
                      "NEG": "ANTI"}

POSITIVE_STANCES = {"PRO", "NEU"}
NEGATIVE_STANCES = {"ANTI"}

def extract_pred_true(item, criterion_key):
    y_true = item['stance'].strip().upper()

    moderation = item.get('moderation')
    if not isinstance(moderation, dict):
        return y_true, "UNK"
    
    results_root = moderation.get('results')
    if not isinstance(results_root, dict):
        return y_true, "UNK"
    
    criterion_res = results_root.get(criterion_key)
    if not isinstance(criterion_res, dict):
        return y_true, "UNK"
    
    tonality = criterion_res['tonality'].strip().upper()
    if tonality not in TONALITY_TO_STANCE:
        y_pred = "UNK"
    else:
        y_pred = TONALITY_TO_STANCE[tonality]
    
    return y_true, y_pred

def to_binary_label(stance):
    stance = stance.strip().upper()
    if stance in POSITIVE_STANCES:
        return 1
    if stance in NEGATIVE_STANCES:
        return 0
    return -1

def collect_stats(data, criterion_key):
    stats = {"total": 0, 
             "gold": {"ANTI": 0, "NEU": 0, "PRO": 0, "UNK": 0},
             "pred": {"POS": 0, "NEU": 0, "NEG": 0, "UNK": 0},
             "missing": {"no_moderation": 0, 
                         "no_results": 0,
                         "no_criterion": 0,
                         "unk_tonality": 0,
                         "unk_tonalities": []},
            }
    
    for item in data:
        stats["total"] += 1


        gold = item["stance"].strip().upper()
        if gold not in stats['gold']:
            stats['gold']['UNK'] += 1 
            
        stats['gold'][gold] += 1
        
        moderation = item.get('moderation')
        if not isinstance(moderation, dict):
            stats['missing']['moderation'] += 1
            stats['pred']['UNK'] += 1
            continue
        
        results = moderation.get('results')
        if not isinstance(results, dict):
            stats['missing']['no_results'] += 1
            stats['pred']['UNK'] += 1
            continue

        criterion = results.get(criterion_key)
        if not isinstance(criterion, dict):
            stats['missing']['no_criterion'] += 1
            stats['pred']['UNK'] += 1
            continue

        tonality = criterion['tonality'].strip().upper()
        if tonality not in TONALITY_TO_STANCE:
            stats['missing']['unk_tonality'] += 1
            stats['missing']['unk_tonalities'] = tonality
            stats['pred']['UNK'] += 1
            continue 

        stats['pred'][tonality] += 1
    
    stats['missing']['unk_tonalities'] = set(stats['missing']['unk_tonalities'])
    return stats 

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

    args = parser.parse_args()

    output_file = None
    if args.tofile: 
        input_dir = os.path.dirname(os.path.abspath(args.input))
        output_path = os.path.join(input_dir, 'metrics.txt')
        output_file = open(output_path, 'w', encoding='utf-8')
        sys.stdout = output_file


    data = load_jsonl(args.input)

    stats = collect_stats(data, args.criterion)

    print("\n=== DATASET & PREDICTION STATISTICS ===")
    print(f"Total samples: {stats['total']}")
    print("\nGold stance distribution:")
    for k, v in stats['gold'].items():
        print(f"  {k}: {v}")

    print("\nPredicted stance distribution:")
    for k, v in stats['pred'].items():
        print(f"  {k}: {v}")

    print("\nMissing / skipped prediction reasons:")
    for k, v in stats['missing'].items():
        print(f"  {k}: {v}")

    y_true_mc, y_pred_mc = [], []
    y_true_bin, y_pred_bin = [], []

    skipped = 0

    for item in data:
        true_stance, pred_stance = extract_pred_true(item, args.criterion)

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

        true_bin = to_binary_label(true_stance)
        pred_bin = to_binary_label(pred_stance)
        if true_bin == -1 or pred_bin == -1:
            skipped += 1 
            continue
            
        y_true_bin.append(true_bin)
        y_pred_bin.append(pred_bin)


    print(f"Evaluation of results produced by: {data[0]['moderation']['model']}")
    print(f"Loaded: {len(data)}")
    if args.skip_unk:
        print(f"Skipped (unknown/missing labels): {skipped}")
    print(f"Evaluated: {len(y_true_mc)}")

    labels_mc = ["ANTI", "NEU", "PRO"]
    print("\n=== MULTICLASS (ANTI/NEU/PRO) ===")
    print("Accuracy:", accuracy_score(y_true_mc, y_pred_mc))
    print("\nConfusion matrix (rows=true, cols=pred), labels:", labels_mc)
    print(confusion_matrix(y_true_mc, y_pred_mc, labels=labels_mc))
    print("\nClassification report:")
    print(classification_report(y_true_mc, y_pred_mc, labels=labels_mc, zero_division=0))

    # --- Binary metrics ---
    # Positive class = 1 (PRO+NEU), Negative = 0 (ANTI)
    print("\n=== BINARY (positive=PRO+NEU, negative=ANTI) ===")
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

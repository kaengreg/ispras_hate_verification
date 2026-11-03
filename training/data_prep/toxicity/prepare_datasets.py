"""Prepare three separate SFT datasets, with shared text-level leakage checks."""

import argparse
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("final_shuff", "national_targeted", "hard")
GENERAL_SYSTEM = (
    "Ты - модель для классификации токсичности высказывания. "
    "TOX означает токсичное, оскорбительное, агрессивное или уничижительное высказывание, "
    "включая грубую нецензурную лексику. NEU означает нетоксичное высказывание. "
    "Токсичность не равна негативной позиции по отношению к цели. "
    "Оценивай только явное содержание текста, без домыслов о намерениях автора. "
    "Верни только одну метку: TOX или NEU."
)
TARGET_SYSTEM = (
    "Ты - модель для классификации враждебного отношения к указанной национальной или этнической группе. "
    "TOX означает негативные или враждебные обобщения, унижение, обесчеловечивание, "
    "призывы к дискриминации, ограничению прав или насилию именно в отношении указанной группы. "
    "NEU означает отсутствие такого отношения, включая нейтральное упоминание и положительное отношение. "
    "Мат или оскорбления в адрес других объектов сами по себе не означают TOX для указанной цели. "
    "Критика действий государства, политики или отдельных лиц без переноса на группу не означает TOX. "
    "Оценивай только явное содержание текста, без домыслов о намерениях автора. "
    "Верни только одну метку: TOX или NEU."
)
HARD_GENERAL_SYSTEM = GENERAL_SYSTEM + (
    " Упоминание страны или народа служит контекстом: определи токсичность самого высказывания "
    "независимо от того, на кого она направлена."
)
HARD_TARGET_SYSTEM = TARGET_SYSTEM + (
    " Рассматривай отдельно отношение к указанной группе и общую токсичность текста: "
    "нейтральное упоминание группы в токсичном тексте получает NEU; "
    "враждебность к группе может получать TOX и без мата или персональных оскорблений."
)


def prompt_id(source, task):
    return f"{source}/{task}/v2"


def clean_text(text):
    if not isinstance(text, str):
        raise ValueError(f"Expected a string text, got {type(text).__name__}")
    text = html.unescape(text)
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?b\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[id\d+\|([^\]]+)\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def text_key(text):
    # Ignore URL destinations and case only for matching; preserve them in model inputs.
    normalized = re.sub(r"https?://\S+|www\.\S+", "<URL>", text, flags=re.IGNORECASE).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_jsonl(path):
    # Several source files use .json extensions despite containing JSONL.
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield line_number, row


def write_line(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_chat(row):
    if row["source"] == "hard" and row["task"] == "general_toxicity":
        system = HARD_GENERAL_SYSTEM
        user = (
            "Задача: классификация токсичности.\n\n"
            f"Цель/объект упоминания: {row['mentioned_target']}\n\nВысказывание:\n{row['text']}\n\n"
            "Определи токсичность самого высказывания независимо от позиции по отношению к цели. "
            "Наличие или отсутствие ненависти к упомянутой группе не подменяет оценку общей токсичности. "
            "Ответь только одной меткой: TOX или NEU."
        )
    elif row["source"] == "hard":
        system = HARD_TARGET_SYSTEM
        user = (
            "Задача: классификация ненависти к указанной группе независимо от общей токсичности.\n\n"
            f"Цель: {row['target']}\n\nВысказывание:\n{row['text']}\n\n"
            "Определи, выражает ли автор враждебность именно к указанной группе. "
            "Нейтральное упоминание группы в тексте с матом или оскорблением другого адресата не является ненавистью к ней. "
            "Враждебные обобщения и дискриминация группы учитываются даже без мата. "
            "Ответь только одной меткой: TOX или NEU."
        )
    elif row["task"] == "general_toxicity":
        system = GENERAL_SYSTEM
        user = (
            "Задача: классификация общей токсичности.\n\n"
            f"Высказывание:\n{row['text']}\n\n"
            "Определи токсичность всего высказывания независимо от отношения к упомянутым объектам. "
            "Ответь только одной меткой: TOX или NEU."
        )
    else:
        system = TARGET_SYSTEM
        user = (
            "Задача: классификация враждебного отношения к указанной группе.\n\n"
            f"Цель: {row['target']}\n\nВысказывание:\n{row['text']}\n\n"
            "Оцени отношение именно к указанной цели. Токсичность в адрес других объектов не учитывай. "
            "Ответь только одной меткой: TOX или NEU."
        )
    metadata = {key: row[key] for key in ("example_id", "prompt_id", "task", "source", "source_file", "source_line", "sentence_id", "text_hash", "target")}
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                     {"role": "assistant", "content": row["label"]}],
        **metadata,
    }


def prepare(args):
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}. Choose a new --output-dir.")
    
    for path in (args.csv_input, args.targeted_train, args.targeted_val, args.hard_train, args.hard_val):
        if not path.is_file():
            raise FileNotFoundError(path)
        
    protected = set()
    stance_counts = {}
    for split in ("train", "val", "test"):
        path = args.stance_dir/f"StanceRus_24_08_{split}.jsonl"
        count = 0
        for _, row in read_jsonl(path):
            protected.add(text_key(clean_text(row["text"])))
            count += 1
        stance_counts[split] = count

    output.mkdir(parents=True)
    counts = {source: Counter() for source in SOURCES}
    groups = {}
    text_splits = {}
    text_sources = defaultdict(set)
    split_collisions = set()
    source_conflicts = Counter()
    raw_rows = Counter()
    with (output/"audit.jsonl").open("w", encoding="utf-8") as audit:
        def log(reason, row, **extra):
            counts[row["source"]][reason] += 1
            write_line(audit, {"reason": reason, "source": row["source"], "source_file": row["source_file"],
                               "source_line": row["source_line"], "text_hash": row["text_hash"],
                               "task": row["task"], "target": row["target"], "label": row["label"], **extra})

        def add(source, path, line, original, text, task, target, label, split):
            key = text_key(text)
            target = clean_text(target) if target else None
            identity = (source, task, key, target.casefold() if target else "")
            row = {"source": source, "source_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   "source_line": line, "sentence_id": original.get("sentence_id"),
                   "text_ids": original.get("text_ids"), "text": text,
                   "mentioned_target": original.get("target"), "prompt_id": prompt_id(source, task),
                   "text_hash": key, "task": task, "target": target, "label": label, "split": split,
                   "example_id": hashlib.sha256("\0".join(identity).encode("utf-8")).hexdigest()}
            
            counts[source]["candidate_examples"] += 1
            if not text:
                log("empty_text", row)
                return
            
            text_sources[key].add(source)
            if split is not None:
                previous_split = text_splits.get(key)
                if previous_split is not None and previous_split != split:
                    split_collisions.add(key)

                # Files named VAL are fixed test sets, shared across all tasks and sources.
                text_splits[key] = "test" if "test" in (previous_split, split) else "train"
                
            entry = groups.get(identity)
            if entry is None:
                groups[identity] = {"row": row, "labels": {label}, "n": 1}
                return
            
            entry["n"] += 1
            previous = entry["row"]
            if label not in entry["labels"]:
                source_conflicts[f"{previous['source']}:{source}:{task}"] += 1
                entry["labels"].add(label)
                log("conflicting_label_occurrence", row, previous_label=previous["label"])
            else:
                counts[source]["within_source_duplicate"] += 1
                log("duplicate_occurrence", row, within_source=True)
            if split == "test" and previous["split"] != "test":
                entry["row"] = row

        def load_targeted(source, path, split, include_general):
            for line, original in read_jsonl(path):
                raw_rows[source] += 1
                text = clean_text(original["text"])
                labels = original.get("targets_by_stance")
                targets = original.get("ethnic_groups")
                if not isinstance(labels, dict) or set(labels) != {"TOX", "NEU"}:
                    raise ValueError(f"{path}:{line}: expected binary targets_by_stance")
                if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.strip() for t in targets):
                    raise ValueError(f"{path}:{line}: expected nonempty ethnic_groups")
                if any(not isinstance(values, list) for values in labels.values()):
                    raise ValueError(f"{path}:{line}: target labels must contain lists")
                if set(targets) != set(labels["TOX"]) | set(labels["NEU"]):
                    raise ValueError(f"{path}:{line}: ethnic_groups and label assignments disagree")
                
                if include_general:
                    label = original.get("toxicity_label")
                    if label not in {"TOX", "NEU"}:
                        raise ValueError(f"{path}:{line}: invalid toxicity_label {label!r}")
                    
                    add(source, path, line, original, text, "general_toxicity", None, label, split)
                for target in targets:
                    target_labels = [label for label, values in labels.items() if target in values]
                    for label in target_labels:
                        # Multiple labels deliberately enter the conflict audit and are excluded together.
                        add(source, path, line, original, text, "targeted_toxicity", target, label, split)

        load_targeted("national_targeted", args.targeted_train, "train", False)
        load_targeted("national_targeted", args.targeted_val, "test", False)
        load_targeted("hard", args.hard_train, "train", True)
        load_targeted("hard", args.hard_val, "test", True)

        with args.csv_input.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["text", "label"]:
                raise ValueError(f"Expected CSV columns text,label, got {reader.fieldnames}")
            for record_number, original in enumerate(reader, 1):
                raw_rows["final_shuff"] += 1
                if None in original or original["label"] not in {"0", "1"}:
                    raise ValueError(f"{args.csv_input}: record {record_number}: invalid CSV row")
                label = {"0": "NEU", "1": "TOX"}[original["label"]]
                # CSV provenance is the record number, not the physical line number.
                add("final_shuff", args.csv_input, record_number, original, clean_text(original["text"]),
                    "general_toxicity", None, label, None)

        overlaps = Counter()
        for sources in text_sources.values():
            for left in SOURCES:
                for right in SOURCES:
                    if SOURCES.index(left) < SOURCES.index(right) and {left, right} <= sources:
                        overlaps[f"{left}:{right}"] += 1

        repeated_inputs = Counter()
        cross_source_disagreements = Counter()
        for identity, entry in groups.items():
            source, task, key, target = identity
            for other in SOURCES[SOURCES.index(source) + 1:]:
                other_entry = groups.get((other, task, key, target))
                if other_entry is not None:
                    pair = f"{source}:{other}:{task}"
                    repeated_inputs[pair] += 1
                    if entry["labels"] != other_entry["labels"]:
                        cross_source_disagreements[pair] += 1
                        log("cross_source_label_disagreement_retained", entry["row"],
                            other_source=other, labels=sorted(entry["labels"]), other_labels=sorted(other_entry["labels"]))
                        
        outputs = defaultdict(list)
        resolved_splits = {}
        for entry in groups.values():
            row = entry["row"]
            if len(entry["labels"]) > 1:
                log("excluded_conflicting_group", row, labels=sorted(entry["labels"]), occurrences=entry["n"])
                continue
            if row["text_hash"] in protected:
                log("excluded_stancerus_overlap", row)
                continue

            key = row["text_hash"]
            if key not in resolved_splits:
                value = int(hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
                original_split = text_splits.get(key)
                if original_split == "test":
                    resolved_splits[key] = "test"
                elif original_split == "train":
                    resolved_splits[key] = "val" if value < args.val_fraction else "train"
                else:
                    resolved_splits[key] = "test" if value < args.test_fraction else (
                        "val" if value < args.test_fraction + args.val_fraction else "train")
                    
            split = resolved_splits[key]
            if row["split"] is not None and row["split"] != split:
                log(f"moved_to_{split}", row)
            row["original_split"] = row["split"]
            row["split"] = split
            outputs[(row["source"], split)].append(row)

        final_counts = {}
        for source in SOURCES:
            directory = output/source
            directory.mkdir()
            source_stats = {}
            for split in ("train", "val", "test"):
                rows = outputs[(source, split)]
                random.Random(args.seed).shuffle(rows)
                distribution = Counter(f"{row['task']}/{row['label']}" for row in rows)
                source_stats[split] = {"examples": len(rows), "unique_texts": len({r["text_hash"] for r in rows}),
                                       "labels": dict(sorted(distribution.items()))}
                with ExitStack() as stack:
                    chat = None if split == "test" else stack.enter_context(
                        (directory / f"{split}_chat.jsonl").open("w", encoding="utf-8"))
                    raw = stack.enter_context((directory / f"{split}_raw.jsonl").open("w", encoding="utf-8"))
                    for row in rows:
                        if chat is not None:
                            write_line(chat, render_chat(row))
                        write_line(raw, row)
            final_counts[source] = source_stats

    summary = {
        "seed": args.seed, "val_fraction": args.val_fraction, "csv_test_fraction": args.test_fraction,
        "split_policy": "Source VAL files are fixed test. Source TRAIN texts split into train/val by seeded hash. "
                        "CSV-only texts use seeded train/val/test hash; overlapping CSV inherits the source split.",
        "test_format": "raw only; no test_chat.jsonl",
        "dedup_policy": "clean HTML/whitespace/VK names, casefold and mask URL destinations for keys; exact normalized matches only",
        "source_priority": None,
        "conflict_policy": "Exclude conflicting (source,task,text,target) within each source; retain cross-source copies and report disagreements.",
        "cross_source_repeated_inputs_retained": dict(repeated_inputs),
        "cross_source_label_disagreements": dict(cross_source_disagreements),
        "prompt_version": "v2; StanceRus-style single-label output, group criteria adapted from prompts.py; hard-specific instructions",
        "stance_policy": "Exclude matching texts from new datasets across ALL StanceRus splits; do not modify StanceRus.",
        "csv_source_line": "1-based CSV record number after header; JSONL uses physical line number",
        "source_rows": dict(raw_rows), "stance_rows": stance_counts, "stance_unique_texts": len(protected),
        "cross_source_text_overlaps_before_dedup": dict(overlaps), "input_train_test_collision_texts": len(split_collisions),
        "conflict_occurrences_by_source_pair": dict(source_conflicts),
        "audit_counts": {source: dict(counter) for source, counter in counts.items()}, "output": final_counts,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_counts, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    data = ROOT/"data/toxicity_criteria"
    targeted = data/"RuEthnoHate_and_Local_hate_n_tox"
    hard = data/"2026_10_06"
    
    parser.add_argument("--csv-input", type=Path, default=data / "FINAL_SHUFF.csv")
    parser.add_argument("--targeted-train", type=Path, default=targeted / "REHate_w_Local_TRAIN.jsonl")
    parser.add_argument("--targeted-val", type=Path, default=targeted / "REHate_w_Local_VAL.jsonl")
    parser.add_argument("--hard-train", type=Path, default=hard / "REHate_w_Local_hard_TRAIN.json")
    parser.add_argument("--hard-val", type=Path, default=hard / "REHate_w_Local_hard_VAL.json")
    parser.add_argument("--stance-dir", type=Path, default=ROOT / "data/anti_russia_criteria/guard_train/splits")
    parser.add_argument("--output-dir", type=Path, default=data / "sft_prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Validation fraction of source TRAIN texts; fraction of all texts for CSV-only groups")
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1 and 0 <= args.test_fraction < 1 and args.val_fraction + args.test_fraction < 1):
        parser.error("Require 0 < val-fraction, 0 <= test-fraction, and their sum < 1")
    return args


if __name__ == "__main__":
    prepare(parse_args())

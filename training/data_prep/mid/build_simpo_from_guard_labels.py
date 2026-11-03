import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path


LABELS = {"PRO", "NEU", "ANTI"}


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_num}: {exc}") from exc
            yield line_num, row


def row_id(row):
    return row.get("id", row.get("sentence_id", row.get("question_id")))


def question_text(row):
    value = row.get("question") or row.get("prompt") or row.get("generated_question")
    return str(value or "").strip()


def answer_text(answer):
    value = answer.get("text") if isinstance(answer, dict) else answer
    return str(value or "").strip()


def load_question_ids(path: Path):
    ids = set()
    for line_num, row in jsonl_rows(path):
        item_id = row_id(row)
        if item_id is None:
            raise ValueError(f"Missing id in {path} on line {line_num}")
        if item_id in ids:
            raise ValueError(f"Repeated id={item_id!r} in {path} on line {line_num}")
        ids.add(item_id)
    return ids


def candidate_id(answer, position):
    if isinstance(answer, dict):
        return str(answer.get("candidate_id", position))
    return str(position)


def validate_same_row(rows, paths, line_nums):
    ids = [row_id(row) for row in rows]
    if any(value is None for value in ids):
        raise ValueError(f"Missing id in synchronized input lines {line_nums}")
    if len(set(ids)) != 1:
        locations = ", ".join(f"{path}:{line_num} id={value!r}" for path, line_num, value in zip(paths, line_nums, ids))
        raise ValueError(f"Input row mismatch: {locations}")

    questions = [question_text(row) for row in rows]
    if not questions[0]:
        raise ValueError(f"Missing question for id={ids[0]!r}")
    if len(set(questions)) != 1:
        raise ValueError(f"Question mismatch for id={ids[0]!r}")

    source_texts = [str(row.get("text") or "").strip() for row in rows]
    if not source_texts[0]:
        raise ValueError(f"Missing source text for id={ids[0]!r}")
    if len(set(source_texts)) != 1:
        raise ValueError(f"Source text mismatch for id={ids[0]!r}")


def synchronized_rows(paths):
    streams = [jsonl_rows(path) for path in paths]
    for items in itertools.zip_longest(*streams):
        if any(item is None for item in items):
            raise ValueError("Input files contain different numbers of non-empty rows")
        line_nums = [item[0] for item in items]
        rows = [item[1] for item in items]
        validate_same_row(rows, paths, line_nums)
        yield rows


def indexed_answers(row, path, item_id):
    answers = row.get("generated_answers")
    if answers is None:
        return {}
    if not isinstance(answers, list):
        raise ValueError(f"generated_answers must be a list in {path} for id={item_id!r}")

    result = {}
    for position, answer in enumerate(answers):
        text = answer_text(answer)
        if not text:
            raise ValueError(f"Empty candidate in {path} for id={item_id!r}, position={position}")
        key = candidate_id(answer, position)
        if key in result:
            raise ValueError(f"Repeated candidate_id={key!r} in {path} for id={item_id!r}")
        label = str(answer.get("guard_label", "")).strip().upper() if isinstance(answer, dict) else ""
        if label not in LABELS:
            raise ValueError(f"Invalid guard_label={label!r} in {path} for id={item_id!r}, candidate_id={key!r}")
        result[key] = {"candidate_id": key, "text": text, "label": label}
    return result


def stable_rank(candidate, item_id, role, seed):
    value = f"{seed}:{item_id}:{role}:{candidate['candidate_id']}:{candidate['text']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique_by_text(candidates):
    result = []
    seen = set()
    for candidate in candidates:
        key = candidate["text"].casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def ordered_candidates(candidates, item_id, role, seed):
    return sorted(unique_by_text(candidates), key=lambda item: stable_rank(item, item_id, role, seed))


def choose_pair(positive_candidates, negative_candidates, item_id, seed):
    pro = [item for item in positive_candidates if item["label"] == "PRO"]
    neutral = [item for item in positive_candidates if item["label"] == "NEU"]
    rejected = [item for item in negative_candidates if item["label"] == "ANTI"]

    chosen = ordered_candidates(pro, item_id, "chosen_pro", seed)
    chosen.extend(ordered_candidates(neutral, item_id, "chosen_neu", seed))
    rejected = ordered_candidates(rejected, item_id, "rejected", seed)

    for chosen_candidate in chosen:
        for rejected_candidate in rejected:
            if chosen_candidate["text"].casefold() != rejected_candidate["text"].casefold():
                return chosen_candidate, rejected_candidate
    return None, None


def selection_metadata(candidate):
    return {
        "candidate_id": candidate["candidate_id"],
        "guard_label": candidate["label"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-input", type=Path, required=True, help="Natural/PRO-NEU candidates labeled with question context")
    parser.add_argument("--negative-input", type=Path, required=True, help="Targeted ANTI candidates labeled with question context")
    parser.add_argument("--exclude-questions", type=Path,
                        help="Fixed test-question JSONL; matching ids are excluded from training pairs")
    parser.add_argument("--output", type=Path, required=True, help="Output SimPO JSONL")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [args.positive_input, args.negative_input]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.exclude_questions is not None and not args.exclude_questions.is_file():
        raise FileNotFoundError(args.exclude_questions)
    excluded_ids = load_question_ids(args.exclude_questions) if args.exclude_questions else set()
    observed_excluded_ids = set()

    stats = Counter()
    chosen_labels = Counter()
    question_types = Counter()
    seen_ids = set()
    seen_questions = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")

    try:
        with temporary_output.open("w", encoding="utf-8") as target:
            for positive_row, negative_row in synchronized_rows(paths):
                stats["input_rows"] += 1
                item_id = row_id(positive_row)
                question = question_text(positive_row)
                normalized_question = question.casefold()

                if item_id in excluded_ids:
                    observed_excluded_ids.add(item_id)
                    stats["excluded_holdout"] += 1
                    continue

                if item_id in seen_ids:
                    stats["duplicate_id"] += 1
                    continue
                seen_ids.add(item_id)
                if normalized_question in seen_questions:
                    stats["duplicate_question"] += 1
                    continue
                seen_questions.add(normalized_question)

                positive = list(indexed_answers(positive_row, args.positive_input, item_id).values())
                negative = list(indexed_answers(negative_row, args.negative_input, item_id).values())
                if not positive:
                    stats["no_positive_candidates"] += 1
                if not negative:
                    stats["no_negative_candidates"] += 1

                chosen, rejected = choose_pair(positive, negative, item_id, args.seed)
                if chosen is None:
                    has_chosen = any(item["label"] in {"PRO", "NEU"} for item in positive)
                    has_rejected = any(item["label"] == "ANTI" for item in negative)
                    if not has_chosen:
                        stats["no_chosen_label"] += 1
                    if not has_rejected:
                        stats["no_rejected_label"] += 1
                    if has_chosen and has_rejected:
                        stats["chosen_equals_rejected"] += 1
                    stats["skipped"] += 1
                    continue

                chosen_label = chosen["label"]
                chosen_labels[chosen_label] += 1
                question_types[str(positive_row.get("question_type") or "UNKNOWN")] += 1
                output = {
                    "id": item_id,
                    "text": positive_row.get("text"),
                    "prompt": question,
                    "chosen": chosen["text"],
                    "rejected": rejected["text"],
                    "source_label": positive_row.get("source_label"),
                    "question_type": positive_row.get("question_type"),
                    "selection_metadata": {
                        "method": "guard_labels_only",
                        "guard_context": "question_and_answer",
                        "seed": args.seed,
                        "chosen": selection_metadata(chosen),
                        "rejected": selection_metadata(rejected),
                    },
                }
                target.write(json.dumps(output, ensure_ascii=False) + "\n")
                stats["selected"] += 1
        temporary_output.replace(args.output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    missing_excluded_ids = excluded_ids - observed_excluded_ids
    if missing_excluded_ids:
        args.output.unlink(missing_ok=True)
        examples = sorted(map(str, missing_excluded_ids))[:10]
        raise ValueError(f"{len(missing_excluded_ids)} holdout ids were not found in the inputs; examples={examples}")

    print(f"Saved {stats['selected']} SimPO pairs to {args.output}")
    print(f"Seed={args.seed}")
    print(f"Stats={dict(stats)}")
    print(f"Chosen labels={dict(chosen_labels)}")
    print(f"Selected question types={dict(question_types)}")


if __name__ == "__main__":
    main()

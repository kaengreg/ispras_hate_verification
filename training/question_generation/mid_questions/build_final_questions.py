import argparse
import json
from pathlib import Path


def normalized_question(question):
    return " ".join(question.casefold().split())


def load_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def load_generation_metadata(input_path, config_path, source_path):
    if config_path and config_path.exists():
        config = load_json(config_path)
        return config.get("model"), config.get("prompt_sha256")

    for path in (input_path, source_path):
        if path and path.exists():
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    generation = (json.loads(line).get("question_generation") or {})
                    if generation.get("model") and generation.get("prompt_sha256"):
                        return generation["model"], generation["prompt_sha256"]

    raise FileNotFoundError(
        "Generation metadata was not found in input. Pass --generation-config or --generation-source."
    )


def load_review_progress(path, run_signature):
    completed = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_signature") == run_signature:
                completed[record["index"]] = record
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-field", default="reviewed_question")
    parser.add_argument("--id-field", default="sentence_id")
    parser.add_argument("--text-field", default="text")
    generation_metadata = parser.add_mutually_exclusive_group()
    generation_metadata.add_argument( "--generation-source", type=Path, help="Original questions JSONL with per-row question_generation metadata. Use for old generator outputs.")
    generation_metadata.add_argument("--generation-config", type=Path, help="Generation .json file containing model and prompt_sha256. Use for new generator outputs.")
    parser.add_argument("--review-config", type=Path)
    parser.add_argument("--review-progress", type=Path)
    args = parser.parse_args()

    review_config_path = args.review_config or Path(str(args.input) + ".review.json")
    review_progress_path = args.review_progress or Path(str(args.input) + ".review_progress.jsonl")
    if not review_config_path.exists() or not review_progress_path.exists():
        raise FileNotFoundError("Review files were not found. Pass --review-config and --review-progress.")

    review_config = load_json(review_config_path)
    run_signature = review_config.get("run_signature")
    review_records = load_review_progress(review_progress_path, run_signature)

    generation_model, generation_prompt = load_generation_metadata(args.input, args.generation_config, args.generation_source)

    review_model = review_config.get("model")
    review_prompt = review_config.get("prompt_sha256")
    if not review_model or not review_prompt or not run_signature:
        raise ValueError("Review config does not contain model, prompt_sha256 or run_signature")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    seen_questions = set()
    input_rows = 0
    output_rows = 0
    without_question = 0
    duplicates = 0
    review_skipped = 0
    review_errors = 0

    with args.input.open("r", encoding="utf-8") as source, tmp_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            index = input_rows
            input_rows += 1
            record = json.loads(line)
            question = record.get(args.question_field)
            if not isinstance(question, str) or not question.strip():
                review = review_records.get(index)
                if review and review.get("status") == "skipped":
                    review_skipped += 1
                else:
                    without_question += 1
                continue
            question = " ".join(question.split())

            review = review_records.get(index)
            if not review:
                raise ValueError(f"No review progress record for input row {index}")
            review_result = review.get("result") or {}
            review_status = review.get("status")
            if review_status == "skipped":
                review_skipped += 1
                continue
            if review_status == "error":
                review_errors += 1
                continue
            if review_status not in {"kept", "rewritten"}:
                raise ValueError(f"Unexpected review status {review_status!r} for input row {index}")

            question_key = normalized_question(question)
            if question_key in seen_questions:
                duplicates += 1
                continue
            seen_questions.add(question_key)

            output_record = {
                "id": record.get(args.id_field),
                "text": record.get(args.text_field),
                "question": question,
                "source_label": record.get("pred_label"),
                "metadata": {
                    "generation_model": generation_model,
                    "generation_prompt_sha256": generation_prompt,
                    "review_model": review_model,
                    "review_prompt_sha256": review_prompt,
                    "review_status": review_status,
                    "rewrite_rounds": review_result.get("rewrite_rounds", 0),
                },
            }
            if output_record["id"] is None or not output_record["text"] or output_record["source_label"] is None:
                raise ValueError(f"Missing id, source text or source label in input row {index}")

            target.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_rows += 1

    tmp_path.replace(args.output)
    print(
        f"Saved {output_rows} questions to {args.output}. "
        f"Input rows={input_rows}, without question={without_question}, review skipped={review_skipped}, "
        f"review errors={review_errors}, duplicates={duplicates}."
    )


if __name__ == "__main__":
    main()

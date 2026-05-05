import json
import os
import argparse
import hashlib
import random
from tqdm import tqdm

def load_json(path):
    data = []
    
    with open(path, 'r', encoding='utf-8') as fin:
        first_char = fin.read(1)
        fin.seek(0)
        if first_char == "[":
            data = json.load(fin)
        else:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    return data

def get_stance(themes_n_stances, theme: str = 'Россия'):
    if not isinstance(themes_n_stances, list):
        return None 
    
    for x in themes_n_stances:
        if isinstance(x, dict) and x.get("Theme") == theme and x.get("Stance"):
            return x["Stance"]
        
    for x in themes_n_stances:
        if isinstance(x, dict) and x.get("Stance"):
            return x["Stance"]
            
    return None 

def make_uid(item):
    s = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_prompt(question):
    return str(question).strip()

def load_test_questions(path: str) -> set:
    qs = set()
    if not path:
        return qs
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get('prompt') or obj.get('question')
            if q:
                qs.add(str(q).strip())
    return qs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--train_output', required=True, help='Output JSONL for SimPO train pairs')

    parser.add_argument('--sft_test_file', default='', help='Existing SFT test JSONL; must contain "question" (and optionally "prompt"). Used to exclude prompts from SimPO train and build SimPO test.')
    parser.add_argument('--simpo_test_output', default='', help='If --sft_test_file is set: output JSONL with SimPO test pairs (prompt/chosen/rejected).')

    parser.add_argument('--test_file', default='', help='Question-only test JSONL (created if missing) when --sft_test_file is not provided.')

    parser.add_argument('--dedup', action='store_true')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Fraction of unique questions to hold out for test when creating a new split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for created split')
    parser.add_argument('--force_resplit', action='store_true', help='Overwrite test_file and recreate split even if it exists (fallback mode only)')
    parser.add_argument('--theme', default='Россия')

    args = parser.parse_args()

    if args.sft_test_file:
        if not args.simpo_test_output:
            raise ValueError('When --sft_test_file is provided, you must also set --simpo_test_output to write SimPO test pairs.')
    else:
        if not args.test_file:
            raise ValueError('When --sft_test_file is not provided, you must set --test_file (question-only test split file).')

    data = load_json(args.input)

    examples = []

    for item in tqdm(data, desc="Building SimPO pairs"):
        sentence_id = item.get('sentence_id')
        link = item.get('link')

        anti_text = item.get('text')
        anti_question = item.get('generated_question')

        generated_items = item.get("Generated_texts_by_QWEN32") or []
        if not isinstance(generated_items, list) or not generated_items:
            continue

        for gi, gen in enumerate(generated_items):
            if not isinstance(gen, dict):
                continue

            pos_text = gen.get('generated_text')
            pos_question = gen.get('generated_question')

            stance = get_stance(gen.get('themes_n_stances'), theme=args.theme)
            if not stance:
                continue
            stance = str(stance).strip().upper()
            if stance not in {"NEU", "PRO"}:
                continue

            if not pos_text or not pos_question or not anti_text or not anti_question:
                continue

            samples = [
                {
                    "prompt": make_prompt(pos_question),
                    "chosen": str(pos_text).strip(),
                    "rejected": str(anti_text).strip(),
                    "meta": {
                        "sentence_id": sentence_id,
                        "link": link,
                        "theme": args.theme,
                        "positive_stance": stance,
                        "negative_stance": "ANTI",
                        "source": "pos_question",
                        "gen_index": gi,
                    },
                },
                {
                    "prompt": make_prompt(anti_question),
                    "chosen": str(pos_text).strip(),
                    "rejected": str(anti_text).strip(),
                    "meta": {
                        "sentence_id": sentence_id,
                        "link": link,
                        "theme": args.theme,
                        "positive_stance": stance,
                        "negative_stance": "ANTI",
                        "source": "anti_question",
                        "gen_index": gi,
                    },
                },
            ]

            for ex in samples:
                ex["sample_uid"] = make_uid({
                    "sentence_id": ex["meta"]["sentence_id"],
                    "gen_index": ex["meta"]["gen_index"],
                    "source": ex["meta"]["source"],
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                    "pos_stance": ex["meta"]["positive_stance"],
                })
                examples.append(ex)

    # Decide which prompts belong to test (membership is ALWAYS by question/prompt text)
    if args.sft_test_file:
        # Primary path: reuse SFT test membership
        reuse_existing_test = True
        test_questions = load_test_questions(args.sft_test_file)
    else:
        # Fallback path: create/reuse question-only split file
        reuse_existing_test = os.path.exists(args.test_file) and (not args.force_resplit)

        if reuse_existing_test:
            test_questions = load_test_questions(args.test_file)
        else:
            rng = random.Random(args.seed)
            unique_prompts = sorted({ex['prompt'] for ex in examples if ex.get('prompt')})
            n_test = int(round(len(unique_prompts) * float(args.test_ratio)))
            n_test = max(0, min(n_test, len(unique_prompts)))
            test_questions = set(rng.sample(unique_prompts, n_test))

            # Build mapping question -> set of sentence_ids (for easier identification)
            question_to_sentence_ids = {}
            for ex in examples:
                q = ex.get('prompt')
                if q in test_questions:
                    sid = ex.get('meta', {}).get('sentence_id')
                    if q not in question_to_sentence_ids:
                        question_to_sentence_ids[q] = set()
                    if sid is not None:
                        question_to_sentence_ids[q].add(sid)

            # Write ONLY unique questions (+ sentence_id list) to test_file
            with open(args.test_file, 'w', encoding='utf-8') as ftest_q:
                for q in sorted(test_questions):
                    sentence_ids = sorted(question_to_sentence_ids.get(q, []))
                    ftest_q.write(
                        json.dumps({'question': q, 'sentence_id': sentence_ids}, ensure_ascii=False) + "\n"
                    )

    # Write TRAIN SimPO examples, filtering out any prompts that belong to test_questions
    seen_train = set()
    written_train = 0
    skipped_test_prompts = 0

    with open(args.train_output, 'w', encoding='utf-8') as ftrain:
        for ex in tqdm(examples, desc='Writing TRAIN'):
            if ex['prompt'] in test_questions:
                skipped_test_prompts += 1
                continue

            key = (ex['prompt'] + "\n---\n" + ex['chosen'] + "\n---\n" + ex['rejected']).lower()
            if args.dedup and key in seen_train:
                continue
            seen_train.add(key)

            ftrain.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written_train += 1

    written_test_questions = len(test_questions)

    # If we are reusing SFT test membership, also write SimPO test PAIRS for those prompts
    written_test_pairs = 0
    if args.sft_test_file:
        seen_test = set()
        with open(args.simpo_test_output, 'w', encoding='utf-8') as ftest_pairs:
            for ex in tqdm(examples, desc='Writing SimPO TEST'):
                if ex['prompt'] not in test_questions:
                    continue

                key = (ex['prompt'] + "\n---\n" + ex['chosen'] + "\n---\n" + ex['rejected']).lower()
                if args.dedup and key in seen_test:
                    continue
                seen_test.add(key)

                ftest_pairs.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written_test_pairs += 1

    if args.sft_test_file:
        print(
            f"Done. Train SimPO: {written_train} -> {args.train_output} | "
            f"SFT test questions: {written_test_questions} (from {args.sft_test_file}) | "
            f"SimPO test pairs: {written_test_pairs} -> {args.simpo_test_output} | "
            f"Filtered out train examples by test prompts: {skipped_test_prompts}"
        )
    else:
        print(
            f"Done. Train SimPO: {written_train} -> {args.train_output} | "
            f"Test questions: {written_test_questions} -> {args.test_file} | "
            f"Test source: {'reused existing' if reuse_existing_test else 'created new'} | "
            f"Filtered out train examples by test prompts: {skipped_test_prompts}"
        )


if __name__ == "__main__":
    main()
import argparse
import json
import os
import random
import hashlib

SYSTEM_PROMPT = ""

def load_json(path): 
    data = []
    with open(path, 'r', encoding='utf-8') as fin:
        first = fin.read(1)
        fin.seek(0)

        if first == "[":
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

    for item in themes_n_stances:
        if isinstance(item, dict) and item.get("Theme") == theme:
            return item.get("Stance")
    
    for item in themes_n_stances:
        if isinstance(item, dict) and item.get("Stance"):
            return item["Stance"]
        
    return None


def compute_sample_uid(*, sentence_id, source, gen_index, stance, question, answer):
    metadata = {
        "sentence_id": sentence_id,
        "source": source,
        "gen_index": gen_index,
        "stance": stance,
        "question": question,
        "answer": answer,
    }
    s = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_user_prompt(question):
    return str(question).strip()

def build_messages(question):
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': build_user_prompt(question)},
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dedup', action='store_true')
    parser.add_argument('--theme', default='Россия')
    parser.add_argument('--test_output', default=None)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_resplit', action='store_true')

    args = parser.parse_args()

    data = load_json(args.input)

    output_rich = str('_rich.'.join(args.output.split('.')))

    def dedup_key(messages, assistant_text: str) -> str:
        user_text = messages[1]['content'] if len(messages) > 1 else ''
        return (user_text.strip() + "\n---\n" + assistant_text.strip()).lower()

    candidates = []  

    for item in data:
        sentence_id = item.get('sentence_id')
        link = item.get('link')
        theme = args.theme

        for gi, item in enumerate(item.get('Generated_texts_by_QWEN32', [])):
            if not isinstance(item, dict):
                continue

            question = item.get('generated_question')
            answer = item.get('generated_text')
            stance = get_stance(item.get('themes_n_stances'), theme)

            if not question or not answer or not stance:
                continue
            if str(stance).strip().upper() != 'PRO':
                continue

            candidates.append({
                'sentence_id': sentence_id,
                'link': link,
                'theme': theme,
                'source': 'generated_text',
                'gen_index': gi,
                'stance': stance,
                'question': question,
                'answer': answer,
                'themes_n_stances': item.get('themes_n_stances'),
            })

    for c in candidates:
        c['sample_uid'] = compute_sample_uid(sentence_id=c['sentence_id'], source=c['source'], gen_index=c['gen_index'],
                                            stance=c['stance'], question=c['question'],answer=c['answer'])

    use_test = bool(args.test_output)
    test_uids = set()

    if use_test:
        test_exists = os.path.exists(args.test_output)

        if test_exists and (not args.force_resplit):
            with open(args.test_output, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    uid = obj.get('sample_uid')
                    if uid:
                        test_uids.add(uid)
        else:
            rng = random.Random(args.seed)
            unique_uids = sorted({c['sample_uid'] for c in candidates})

            n_test = int(round(len(unique_uids) * float(args.test_ratio)))
            n_test = max(0, min(n_test, len(unique_uids)))
            test_uids = set(rng.sample(unique_uids, n_test))

            written_test = 0
            with open(args.test_output, 'w', encoding='utf-8') as ftest:
                for c in candidates:
                    if c['sample_uid'] not in test_uids:
                        continue

                    test_rec = {
                        'sample_uid': c['sample_uid'],
                        'sentence_id': c['sentence_id'],
                        'link': c['link'],
                        'source': c['source'],
                        'gen_index': c['gen_index'],
                        'theme': c['theme'],
                        'stance': c['stance'],
                        'question': c['question'],
                    }

                    ftest.write(json.dumps(test_rec, ensure_ascii=False) + "\n")
                    written_test += 1

    seen_train = set()
    written_train = 0

    with open(output_rich, 'w', encoding='utf-8') as frich, \
         open(args.output, 'w', encoding='utf-8') as fclean:

        for c in candidates:
            if use_test and (c['sample_uid'] in test_uids):
                continue

            messages = build_messages(c['question'])
            rich_ex = {
                'id': f"{c['sentence_id']}_{'ANTI' if c['source']=='original_text' else 'GEN_' + str(c['gen_index'])}" if c['sentence_id'] is not None else c['sample_uid'],
                'messages': messages + [{'role': 'assistant', 'content': c['answer']}],
                'meta': {
                    'sample_uid': c['sample_uid'],
                    'sentence_id': c['sentence_id'],
                    'link': c['link'],
                    'source': c['source'],
                    'gen_index': c['gen_index'],
                    'themes_n_stances': c['themes_n_stances'],
                    'stance': c['stance'],
                }
            }

            key = dedup_key(messages, c['answer'])
            if args.dedup and key in seen_train:
                continue
            seen_train.add(key)

            frich.write(json.dumps(rich_ex, ensure_ascii=False) + "\n")
            fclean.write(json.dumps({'messages': rich_ex['messages']}, ensure_ascii=False) + "\n")
            written_train += 1

    if use_test:
        print(
            f"Done. Train: {written_train} examples -> {output_rich} / {args.output}. "
            f"Test uids: {len(test_uids)} -> {args.test_output} "
            f"({'reused' if (os.path.exists(args.test_output) and not args.force_resplit) else 'created/overwritten'})"
        )
    else:
        print(f"Done. Wrote {written_train} train examples -> {output_rich} / {args.output}")

if __name__ == '__main__':
    main()
import json
from pathlib import Path 
from sklearn.model_selection import train_test_split
import argparse

STANCES = ["ANTI", "PRO", "NEU"]

#SYSTEM_PROMPT_V1 = ("Ты - модель для классификации высказываний по отношению к указанной цели."
#                 "Отвечай ровно одним словом: ANTI, PRO или NEU.")

SYSTEM_PROMPT_V2 = ("Ты - модель для классификации высказываний по отношению к указанной цели."
                    "Есть три класса: ANTI, PRO и NEU."
                    "Необходимо классифицировать высказывание и вернуть ТОЛЬКО метку одного из трех классов.")

SYSTEM_PROMPT = SYSTEM_PROMPT_V2

def load_json(path):
    data = []
    with open(path, "r", encoding='utf-8') as fin:
        first_char = fin.read(1)
        fin.seek(0) 
        if first_char == '[':
            data = json.load(fin)
        else:
            for line in fin:
                line = line.strip()
                if not line: 
                    continue
                data.append(json.loads(line))
    return data
    
def save_jsonl(path, data):
    with open(path, "w", encoding='utf-8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

def build_messages(example):
    text = example['text']
    target = example['target']
    stance = example['stance']

    if stance not in STANCES:
        raise ValueError(f"UNEXPECTED STANCE LABEL: {stance}")
    
    #user_content_v1 = ("Определи тональность текста/отношение автора по отношению к цели в данном высказывании. \n\n"
    #                f"**Высказывани:** \n\n {text}\n\n "
    #                f"**Цель:**  \n\n{target}\n\n"
    #                "Ответ должен быть ОДНИМ словом: 'ANTI', 'PRO' или 'NEU'.")
    
    user_content_v2 = ("Определи тональность текста/позицию автора по отношению к цели в данном высказывании. \n\n"
                        f"**Высказывание:** \n\n {text}\n\n "
                        f"**Цель:**  \n\n{target}\n\n"
                        "Ответ должен быть ОДНИМ словом: 'ANTI', 'PRO' или 'NEU'.")
    
    user_content = user_content_v2
    
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': stance}]

    return {"messages": messages}
    

def main(input_path, output_dir, test_path, test_size: float = 0.1, random_seed: int = 42):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}")
    data = load_json(input_path)
    print(f"Total examples: {len(data)}")

    allowed_labels = STANCES
    clean_data = []
    for example in data:
        stance = example.get('stance')
        if stance not in STANCES:
            print(f"Skipping example with bad label: {stance} (id={example.get('sentence_id')})")
            continue
        clean_data.append(example)

    print(f"After label filtering: {len(clean_data)} examples")


    if test_path is not None:
        print(f"Loading existing test set from {test_path}")
        
        test_data = load_json(test_path)
        test_ids = set()
        test_keys = set()
        
        for example in test_data:
            sent_id = example['sentence_id']
            if sent_id is not None:
                test_ids.add(sent_id)
            else:
                key = (example['text'], example['target'], example['stance'])
                test_keys.add(key)

        train_data = []
        test_data = []
        matched_test_ids = set()
        matched_test_keys = set()

        for example in clean_data:
            sent_id = example['sentence_id']
            key = (example['text'], example['target'], example['stance'])

            if sent_id is not None and sent_id in test_ids:
                test_data.append(example)
                matched_test_ids.add(sent_id)
            elif sent_id is None and key in test_keys:
                test_data.append(example)
                matched_test_keys.add(key)
            else:
                train_data.append(example)

        missing_ids = test_ids - matched_test_ids
        missing_keys = test_keys - matched_test_keys
        missing_count = len(missing_ids) + len(missing_keys)

        if missing_count > 0:
            print(f"[WARNING]: {missing_count} examples from the provided test set were not found in the input data")

    else:
        if test_size is not None and test_size > 0:
            train_data, test_data = train_test_split(
                clean_data,
                test_size=test_size,
                random_state=random_seed,
                shuffle=True,
                stratify=[example['stance'] for example in clean_data]
            )
        else:
            print("No test partition requested; using all clean data for training")
            train_data = clean_data
            test_data = []
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    train_chat = [build_messages(example) for example in train_data]

    train_chat_path = output_dir/'train_chat.jsonl'
    test_raw_path = output_dir/'test_raw.jsonl'
    train_raw_path = output_dir/'train_raw.jsonl'

    print(f"Saving train dataset to {train_chat_path}")
    save_jsonl(train_chat_path, train_chat)

    print(f"Saving train raw dataset to {train_raw_path}")
    save_jsonl(train_raw_path, train_data)

    if test_size > 0:
        print(f"Saving test dataset to {test_raw_path}")
        save_jsonl(test_raw_path, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/Users/kaengreg/Documents/Работа /НИВЦ/ispras/countries_15_01_full_filtered.json")
    parser.add_argument("--output_dir", type=str, default="data/countries_classification/")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of data to use for test split. Set to 0 to disable test partition.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-path", type=str, default=None)

    args = parser.parse_args()
    main(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
        test_path=args.test_path
    )
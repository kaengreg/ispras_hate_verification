import json 
import argparse
import os 
import torch 
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "Ты — интеллектуальный помощник, который пишет естественные, связные и стилистически корректные "
    "тексты на русском языке, являющиеся уместными ответами на заданные вопросы.\n\n"
)


def build_user_prompt(theme, question):
    return f""" 
    Тебе задан вопрос по теме «{theme}». 

    Твоя задача — по вопросу пользователя и указанной позиции (stance) написать такой ответ, 
    который выглядел бы логичным, развернутым и естественным ответом на этот вопрос.\n\n

    Как правильно интепретировать позицию (stance):\n"
    - PRO — текст должен формировать позитивное или оправдывающее представление о теме.\n
    - ANTI — текст должен формировать критическое или негативное представление о теме.\n
    - NEU — текст должен быть нейтральным и информативным без выраженной оценки.\n\n

    Важно:\n
    - Текст должен выглядеть как обычный ответ, а не как выполнение инструкции по пунктам.\n
    - Нельзя упоминать позицию (stance) явно.\n
    - Нельзя добавлять дисклеймеры, пояснения, мета-комментарии.\n
    - Нельзя ссылаться на то, что тебе даны инструкции или контекст.\n
    - Ответ должен состоять из ОДНОГО предложения.\n
    - Ответ должен напрямую отвечать на вопрос, а не пересказывать его.\n
    - Формулировки должны быть естественными и грамотными.\n

    Позиция: "PRO" \n
    Вопрос: {question} \n\n
    """

def build_messages(theme, question):
    user_prompt = build_user_prompt(theme, question)
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}]


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

def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temp', default=0.7)
    parser.add_argument('--top-p', default=0.9)
    parser.add_argument('--top-k', default=50)
    parser.add_argument('--repetition_penalty', default=1.0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    data = load_json(args.input)

    print(f"Loaded: {len(data)} test items from {args.input}")

    device = "cuda" if torch.cuda.is_available() else "auto"
    dtype = torch.bfloat16 

    print(f"Loading tokenizer: {args.model})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    
    print(f"Loading model: {args.model} (dtype={dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    outputs = []

    def prepare_prompt(item):
        theme = item['theme'] 
        question = item['question']
        messages = build_messages(theme, question) 

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            if args.debug:
                print(f"[Tokenizer chat template]: {tokenizer.chat_template}")
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        return f"{SYSTEM_PROMPT}\n Пользователь: {question}\n Ассистент:"
    

    with torch.no_grad():
        for i in tqdm(range(0, len(data), args.batch_size), desc='Generating answers'):
            batch = data[i:i+args.batch_size]
            prompts = [prepare_prompt(item) for item in batch] 

            tokenized = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

            if device == "cuda":
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
            else:
                tokenized = {k: v.to("cpu") for k, v in tokenized.items()}

            generated = model.generate(
                            **tokenized,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temp,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
            
            input_lens = tokenized['input_ids'].shape[1]

            for j, item in enumerate(batch):
                full_text = tokenizer.decode(generated[j], skip_special_tokens=True)
                if args.debug:
                    print(f"[Full text] {full_text}")

                prompt_text = prompts[j]

                if full_text.startswith(prompt_text):
                    answer = full_text[len(prompt_text):]
                else:
                    answer_ids = generated[j][input_lens:]
                    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

                answer = str(answer).strip()
                answer = " ".join(answer.split())

                if args.debug:
                    print(f"[Answer only] {answer}")
                
                out_row = dict()
                out_row['sentence_id'] = item["sample_uid"]
                out_row["text"] = answer
                out_row["target"] = item["theme"]
                out_row["stance"] = item["stance"]
                out_row["question"] = item["question"]

                outputs.append(out_row)
    
    save_jsonl(args.output, outputs)
    print(f"Saved: {len(outputs)} rows to {args.output}")

if __name__ == "__main__":
    main()
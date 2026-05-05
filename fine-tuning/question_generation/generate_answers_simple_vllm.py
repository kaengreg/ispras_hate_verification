import json 
import argparse
import os 
import torch 
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def build_messages(theme, question): 
    return [{'role': 'system', 'content': ""},
            {'role': 'user', 'content': question}]


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
    #model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    #model.eval()

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens
    )
    llm = LLM(
        model=args.model,
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    outputs = []

    def prepare_prompt(item):
        theme = item['theme']
        question = item['question']
        messages = build_messages(theme, question)

        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return {"prompt_token_ids": prompt_token_ids}
    
    prompts = []
    for item in tqdm(data):
        prompt = prepare_prompt(item)
        prompts.append(prompt)

    gen_outputs = llm.generate(prompts, sampling_params=sampling_params)
    for i, item in tqdm(enumerate(data)):
        generated_text = gen_outputs[i].outputs[0].text
        out_row = dict()
        out_row['sentence_id'] = item["sample_uid"]
        out_row["text"] = generated_text
        out_row["target"] = item["theme"]
        out_row["stance"] = item["stance"]
        out_row["question"] = item["question"]

        outputs.append(out_row)
    
    save_jsonl(args.output, outputs)
    print(f"Saved: {len(outputs)} rows to {args.output}")

if __name__ == "__main__":
    main()
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np 
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text_field", default='text')
    parser.add_argument("--add_special_tokens", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = load_dataset("json", data_files=args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    token_lens = []    

    for sample in dataset['train'][args.text_field]:
        text = str(sample)
        encoded = tokenizer(text, add_special_tokens=args.add_special_tokens, truncation=False)
        token_lens.append(len(encoded["input_ids"]))

    token_lens = np.array(token_lens)

    print(f"Количество примеров: {len(token_lens)}")
    print(f"Среднее число токенов{token_lens.mean():.2f}")
    print(f"Медиана: {np.median(token_lens):.0f}")
    print(f"90-й перцентиль: {np.percentile(token_lens, 90):.0f}")
    print(f"95-й перцентиль: {np.percentile(token_lens, 95):.0f}")
    print(f"99-й перцентиль: {np.percentile(token_lens, 99):.0f}")
    print(f"Максимум: {token_lens.max()}")
    print()
    print(f"Рекомендуемый max_length по p95: {int(np.percentile(token_lens, 95))}")

if __name__ == "__main__":

    main()

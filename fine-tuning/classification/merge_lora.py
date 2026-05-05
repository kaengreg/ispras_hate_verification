import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel 
import argparse

def main(args):
    print(f"Loading base model: {args.base_model}")
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]


    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    print(f"Loading LoRA adapters from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print(f"Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    print(f"Saving merge model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.output_dir)

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, required=True, help='Path or HF name of the base model')
    parser.add_argument('--adapter-dir', type=str, required=True, help='Path to LoRA adapter directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save merged model')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16','fp16','fp32'], help='Model dtype for loading/merging')
    return parser

if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
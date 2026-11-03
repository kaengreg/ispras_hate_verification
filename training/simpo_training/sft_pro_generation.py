from dataclasses import dataclass
import argparse
import torch 
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 

@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    max_length: int = 8192
    pad_to_multiple_of: int = 8 

    def __call__(self, features):

        labels_list = [feat['labels'] for feat in features]

        features_no_labels = [{k: v for k, v in feature.items() if k != 'labels'} for feature in features]

        batch = self.tokenizer.pad(features_no_labels, 
                                   padding=True,
                                   max_length=None, 
                                   pad_to_multiple_of=self.pad_to_multiple_of, 
                                   return_tensors='pt')

        max_len = batch['input_ids'].shape[1]

        padded_labels = []
        for labels in labels_list:
            if len(labels) > max_len:
                labels = labels[:max_len]
            
            if len(labels) < max_len:
                labels = labels + [-100] * (max_len - len(labels))
            padded_labels.append(labels)

        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        return batch 
    

def main(args):
    raw_dataset = load_dataset("json", data_files=args.train_data)['train']

    split_dataset = raw_dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_raw = split_dataset['train']
    val_raw = split_dataset['test']

    print(f"Train examples: {len(train_raw)}, Validation examples: {len(val_raw)}")

    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def tokenize_with_mask_loss(example):
        messages = example['messages']
        prompt_message = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=False)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_length)['input_ids']
        tokenized = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)

        input_ids = tokenized['input_ids']
        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))

        for i in range(prompt_len):
            labels[i] = -100
        
        tokenized['labels'] = labels
        return tokenized 
    
    print("Tokenizing train data...")
    train_tokenized = train_raw.map(tokenize_with_mask_loss, batched=False, remove_columns=train_raw.column_names, desc='Tokenizing train')

    print("Tokenizing val dataset...")
    val_tokenized = val_raw.map(tokenize_with_mask_loss, batched=False, remove_columns=val_raw.column_names, desc='Tokenizing validation')

    print("Loading model...")

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1

    model_kwargs = {'attn_implementation': 'flash_attention_2'} if args.flash_attn else {}

    if (args.deepspeed is None) and (not distributed):
        model_kwargs['device_map'] = 'auto'
    if args.bf16:
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif args.fp16:
        model_kwargs['torch_dtype'] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora_confg = LoraConfig(r=16,
                            lora_alpha=32, 
                            lora_dropout=0.05,
                            task_type="CAUSAL_LM",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]
                                            )

    model = get_peft_model(model, lora_confg)
    model.print_trainable_parameters()

    evaluation_strategy_val = 'steps' if args.eval_steps > 0 else 'no'
    

    training_args_kwargs= dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy='steps',
        eval_strategy=evaluation_strategy_val,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to='none',
        ddp_find_unused_parameters=False,
        logging_dir=args.logging_dir,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.deepspeed is not None:
        training_args_kwargs['deepspeed'] = args.deepspeed

    if args.eval_steps > 0:
        training_args_kwargs['eval_steps'] = args.eval_steps
    else:
        training_args_kwargs['eval_steps'] = None

    training_args = TrainingArguments(**training_args_kwargs)
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, max_length=args.max_seq_length)

    trainer = Trainer(model=model,
                    args=training_args, 
                    train_dataset=train_tokenized, 
                    eval_dataset=val_tokenized, 
                    data_collator=data_collator,
                    tokenizer=tokenizer)
    
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default="RefalMachine/RuadaptQwen3-4B-Instruct")
    parser.add_argument('--train-data', type=str, default='questions_instruct_train.jsonl')
    parser.add_argument('--output-dir', type=str, default="RuadaptQwen3-4B-Instruct_prorussia_lora")
    parser.add_argument('--logging-dir', type=str, default="RuadaptQwen3-4B-Insturct_prorussia_lora/logs")
    parser.add_argument('--max-seq-length', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--flash-attn', action='store_true', default=True)
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True)
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    main(args)

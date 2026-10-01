import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        model_inputs = [{key: value for key, value in feature.items() if key != "labels"} for feature in features]
        batch = self.tokenizer.pad(model_inputs, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([label + [-100] * (max_length - len(label)) for label in labels], dtype=torch.long)
        return batch


def render_example(tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("[ERROR] The rendered prompt is not a prefix of the full chat")

    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    response_start = len(prompt_text)
    prompt_length = next((index for index, (_, end) in enumerate(tokenized["offset_mapping"]) if end > response_start), None)
    if prompt_length is None:
        raise ValueError("[ERROR] Assistant response boundary was not found")
    return tokenized["input_ids"], prompt_length


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("[ERROR] Each example must contain at least two messages")
    if messages[-1].get("role") != "assistant" or not str(messages[-1].get("content") or "").strip():
        raise ValueError("[ERROR] The last message must be a non-empty assistant response")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError(f"[ERROR] Invalid chat message: {message!r}")


def truncate_content(tokenizer, content_ids, token_count):
    return tokenizer.decode(content_ids[:token_count], skip_special_tokens=True)


def truncate_messages_to_fit(example, tokenizer, max_seq_length):
    messages = [dict(message) for message in example["messages"]]
    validate_messages(messages)
    input_ids, prompt_length = render_example(tokenizer, messages)
    if len(input_ids) <= max_seq_length:
        return input_ids, prompt_length, False

    user_indices = [index for index, message in enumerate(messages[:-1]) if message["role"] == "user"]
    if not user_indices:
        raise ValueError("[ERROR] Cannot shorten an example without a user message")
    user_index = max(user_indices, key=lambda index: len(tokenizer(messages[index]["content"], add_special_tokens=False)["input_ids"]))
    content_ids = tokenizer(messages[user_index]["content"], add_special_tokens=False)["input_ids"]

    best = None
    low, high = 0, len(content_ids)
    while low <= high:
        middle = (low + high) // 2
        candidate_messages = [dict(message) for message in messages]
        candidate_messages[user_index]["content"] = truncate_content(tokenizer, content_ids, middle)
        candidate_ids, candidate_prompt_length = render_example(tokenizer, candidate_messages)
        if len(candidate_ids) <= max_seq_length:
            best = candidate_ids, candidate_prompt_length
            low = middle + 1
        else:
            high = middle - 1

    if best is None:
        raise ValueError(f"[ERROR] max_seq_length={max_seq_length} is too small even with an empty user message")
    return best[0], best[1], True


def tokenize_dataset(dataset, tokenizer, max_seq_length, description):
    def tokenize(example):
        input_ids, prompt_length, was_truncated = truncate_messages_to_fit(example, tokenizer, max_seq_length)
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError("[ERROR] Assistant label was lost during tokenization")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels, "was_truncated": was_truncated}

    tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names, desc=description)
    truncated_count = sum(tokenized["was_truncated"])
    print(f"{description}: {len(tokenized)} examples, shortened texts: {truncated_count}")
    return tokenized.remove_columns("was_truncated")


def main(args):
    evaluation_strategy = "steps" if args.eval_steps > 0 else "epoch"
    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    
    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        logging_dir=args.logging_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        train_sampling_strategy="group_by_length" if args.group_by_length else "random",
        seed=args.seed,
        data_seed=args.seed
    )
    if args.eval_steps > 0:
        training_args_kwargs["eval_steps"] = args.eval_steps
    if args.save_steps > 0:
        training_args_kwargs["save_steps"] = args.save_steps
    if args.deepspeed is not None:
        training_args_kwargs["deepspeed"] = args.deepspeed
    training_args = TrainingArguments(**training_args_kwargs)

    datasets = load_dataset("json", data_files={"train": args.train_data, "validation": args.val_data})
    train_raw = datasets["train"]
    val_raw = datasets["validation"]
    print(f"Train examples: {len(train_raw)}, validation examples: {len(val_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    with training_args.main_process_first(desc="Tokenizing datasets"):
        train_tokenized = tokenize_dataset(train_raw, tokenizer, args.max_seq_length, "Tokenizing train")
        val_tokenized = tokenize_dataset(val_raw, tokenizer, args.max_seq_length, "Tokenizing validation")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_kwargs = {"attn_implementation": "flash_attention_3"} if args.flash_attn else {}
    if args.deepspeed is None and world_size == 1:
        model_kwargs["device_map"] = "auto"
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora_config = LoraConfig(r=args.lora_r, 
                             lora_alpha=args.lora_alpha, 
                             lora_dropout=args.lora_dropout, 
                             task_type="CAUSAL_LM", 
                             target_modules=args.lora_target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(model=model, args=training_args, train_dataset=train_tokenized, eval_dataset=val_tokenized, data_collator=DataCollatorForCausalLM(tokenizer), processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="RefalMachine/RuadaptQwen3-4B-Instruct")
    parser.add_argument("--train-data", default="data/guard_train/splits/multitask/train_chat.jsonl")
    parser.add_argument("--val-data", default="data/guard_train/splits/multitask/val_chat.jsonl")
    parser.add_argument("--output-dir", default="RuadaptQwen3-4B-Instruct_multitask_lora")
    parser.add_argument("--logging-dir", default="RuadaptQwen3-4B-Instruct_multitask_lora/logs")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

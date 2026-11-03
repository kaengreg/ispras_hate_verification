import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
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


def validate_data_files(paths, split):
    paths = [paths] if isinstance(paths, str) else paths
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{split} data must be a path or a non-empty list of file entries")
    files = []
    for entry in paths:
        entry = {"path": entry} if isinstance(entry, str) else entry
        if not isinstance(entry, dict) or set(entry) - {"path", "max_examples"}:
            raise ValueError(f"Invalid {split} file entry: {entry!r}")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Each {split} file entry must contain a non-empty path")
        if not Path(path).is_file():
            raise FileNotFoundError(f"{split} data file not found: {path}")
        limit = entry.get("max_examples")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError(f"max_examples for {path} must be a positive integer")
        files.append(entry)
    return files


def load_chat_dataset(files, split, seed):
    datasets = []
    for entry in files:
        path = entry["path"]
        dataset = load_dataset("json", data_files=path, split="train")
        if "messages" not in dataset.column_names:
            raise ValueError(f"{split} file is missing the messages column: {path}")
        
        if not len(dataset):
            raise ValueError(f"{split} file is empty: {path}")
        
        dataset = dataset.select_columns(["messages"])
        total = len(dataset)
        limit = entry.get("max_examples")

        if limit is not None and limit < total:
            indices = sorted(random.Random(seed).sample(range(total), limit))
            dataset = dataset.select(indices)
            
        print(f"{split}: {path}: using {len(dataset)} of {total} examples")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def load_json_config(path):
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("The top-level config value must be a JSON object")
    required = {"model_name", "train_data", "val_data", "output_dir", "max_seq_length", "flash_attn",
                "group_by_length", "trainer", "lora"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    unknown = set(config) - required
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    if not isinstance(config["trainer"], dict) or not isinstance(config["lora"], dict):
        raise ValueError("The trainer and lora config sections must be JSON objects")
    for field in ("flash_attn", "group_by_length"):
        if not isinstance(config[field], bool):
            raise ValueError(f"Config field {field!r} must be a JSON boolean")
    for field in ("output_dir", "train_sampling_strategy"):
        if field in config["trainer"]:
            raise ValueError(f"Set {field!r} through the top-level config fields")
    if config["lora"].get("task_type", "CAUSAL_LM") != "CAUSAL_LM":
        raise ValueError("Classification SFT requires lora.task_type='CAUSAL_LM'")
    return config


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the complete training configuration JSON")
    return parser


def main(args):
    config = load_json_config(args.config)
    train_files = validate_data_files(config["train_data"], "train")
    val_files = validate_data_files(config["val_data"], "validation")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        train_sampling_strategy="group_by_length" if config["group_by_length"] else "random",
        **config["trainer"],
    )

    data_seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed
    with training_args.main_process_first(desc="Loading datasets"):
        train_raw = load_chat_dataset(train_files, "train", data_seed)
        val_raw = load_chat_dataset(val_files, "validation", data_seed)
    print(f"Train examples: {len(train_raw)}, validation examples: {len(val_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    with training_args.main_process_first(desc="Tokenizing datasets"):
        train_tokenized = tokenize_dataset(train_raw, tokenizer, config["max_seq_length"], "Tokenizing train")
        val_tokenized = tokenize_dataset(val_raw, tokenizer, config["max_seq_length"], "Tokenizing validation")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_kwargs = {"attn_implementation": "flash_attention_3"} if config["flash_attn"] else {}
    if training_args.deepspeed is None and world_size == 1:
        model_kwargs["device_map"] = "auto"
    if training_args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **model_kwargs)
    if training_args.gradient_checkpointing:
        checkpointing_kwargs = training_args.gradient_checkpointing_kwargs or {"use_reentrant": False}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
        model.config.use_cache = False

    lora_config = LoraConfig(**{**config["lora"], "task_type": "CAUSAL_LM"})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=train_tokenized, 
                      eval_dataset=val_tokenized, 
                      data_collator=DataCollatorForCausalLM(tokenizer), 
                      processing_class=tokenizer)
    
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main(build_arg_parser().parse_args())

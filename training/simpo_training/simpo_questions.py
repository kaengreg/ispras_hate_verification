import argparse
import dataclasses
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import CPOConfig, CPOTrainer


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the complete training configuration JSON")
    return parser


def load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError("The top-level config value must be a JSON object")
    return config


def load_preference_dataset(path: str, name: str) -> Dataset:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"{name} data file not found: {path}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    validate_preference_dataset(dataset, name)
    return dataset


def validate_preference_dataset(dataset: Dataset, name: str) -> None:
    required = {"prompt", "chosen", "rejected"}
    missing_columns = required - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"{name} dataset is missing columns: {sorted(missing_columns)}")
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty")

    errors = Counter()
    examples = []
    seen_pairs = set()
    duplicate_pairs = 0
    for index, row in enumerate(dataset):
        values = {key: row.get(key) for key in required}
        row_has_error = False
        for key, value in values.items():
            if not isinstance(value, str):
                errors[f"non_string_{key}"] += 1
                row_has_error = True
            elif not value.strip():
                errors[f"empty_{key}"] += 1
                row_has_error = True
        if not row_has_error:
            if values["chosen"].strip() == values["rejected"].strip():
                errors["chosen_equals_rejected"] += 1
                row_has_error = True
            pair = tuple(values[key].strip().casefold() for key in ("prompt", "chosen", "rejected"))
            if pair in seen_pairs:
                duplicate_pairs += 1
            else:
                seen_pairs.add(pair)
        if row_has_error and len(examples) < 5:
            examples.append(index)

    if errors:
        raise ValueError(f"Invalid {name} preference data: counts={dict(errors)}, example row indices={examples}")
    print(f"{name}: {len(dataset)} rows, {len(seen_pairs)} unique pairs, {duplicate_pairs} duplicate pairs")


def split_by_prompt(dataset: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if val_split == 0:
        return dataset, None
    if not 0 < val_split < 1:
        raise ValueError("--val_split must be in [0, 1)")

    prompts = [str(value).strip().casefold() for value in dataset["prompt"]]
    unique_prompts = sorted(set(prompts))

    if len(unique_prompts) < 2:
        raise ValueError("At least two unique prompts are required for a train/eval split")

    num_eval = max(1, min(round(len(unique_prompts) * val_split), len(unique_prompts) - 1))
    eval_prompts = set(random.Random(seed).sample(unique_prompts, num_eval))
    train_indices = [i for i, prompt in enumerate(prompts) if prompt not in eval_prompts]
    eval_indices = [i for i, prompt in enumerate(prompts) if prompt in eval_prompts]
    return dataset.select(train_indices), dataset.select(eval_indices)


def ensure_no_prompt_overlap(train_dataset: Dataset, eval_dataset: Dataset) -> None:
    train_prompts = {str(value).strip().casefold() for value in train_dataset["prompt"]}
    eval_prompts = {str(value).strip().casefold() for value in eval_dataset["prompt"]}
    overlap = train_prompts & eval_prompts
    if overlap:
        examples = sorted(overlap)[:3]
        raise ValueError(f"Train/eval prompt leakage: {len(overlap)} overlapping prompts; examples={examples}")


def to_conversational(dataset: Dataset) -> Dataset:
    def convert(row):
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    return dataset.map(convert, remove_columns=dataset.column_names, desc="Converting to conversational preferences")


def main(args):
    config = load_json_config(args.config)
    trainer_json = config.get("trainer", {})
    lora_json = config.get("lora", {})
    cpo_json = config.get("cpo", {})
    if not all(isinstance(section, dict) for section in (trainer_json, lora_json, cpo_json)):
        raise ValueError("The trainer, lora and cpo config sections must be JSON objects")

    required_config = {"model_name", "data", "output_dir"}
    missing_config = required_config - set(config)
    if missing_config:
        raise ValueError(f"Config is missing required fields: {sorted(missing_config)}")
    model_name = str(config["model_name"])
    data_path = str(config["data"])
    output_dir = str(config["output_dir"])

    seed = int(trainer_json.get("seed", 42))
    dataset = load_preference_dataset(data_path, "train")
    eval_data_path = config.get("eval_data")
    if eval_data_path:
        train_ds = dataset
        eval_ds = load_preference_dataset(eval_data_path, "eval")
        ensure_no_prompt_overlap(train_ds, eval_ds)
    else:
        train_ds, eval_ds = split_by_prompt(dataset, float(config.get("val_split", 0.1)), seed)

    print(f"Split: train={len(train_ds)}, eval={len(eval_ds) if eval_ds is not None else 0}")

    train_ds = to_conversational(train_ds)
    if eval_ds is not None:
        eval_ds = to_conversational(eval_ds)

    tokenizer_trust_remote_code = bool(config.get("tokenizer_trust_remote_code", True))
    model_trust_remote_code = bool(config.get("model_trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=tokenizer_trust_remote_code, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = bool(trainer_json.get("bf16", False))
    fp16 = bool(trainer_json.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None

    load_in_4bit = bool(config.get("load_in_4bit", False))
    load_in_8bit = bool(config.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("4-bit and 8-bit loading cannot both be enabled")

    attn_implementation = config.get("attn_implementation")
    if attn_implementation not in {None, "eager", "sdpa", "flash_attention_2", "flash_attention_3"}:
        raise ValueError("attn_implementation must be one of: eager, sdpa, flash_attention_2, flash_attention_3")
    model_kwargs = {"trust_remote_code": model_trust_remote_code}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if load_in_4bit or load_in_8bit:
        compute_dtype = dtype or torch.float32
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = lora_json.get("target_modules")
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("LoRA target_modules must be a non-empty list")

    peft_config = LoraConfig(
        r=int(lora_json.get("r", 16)),
        lora_alpha=int(lora_json.get("lora_alpha", 32)),
        lora_dropout=float(lora_json.get("lora_dropout", 0.05)),
        bias=str(lora_json.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=lora_json.get("modules_to_save"),
    )

    gradient_checkpointing = bool(trainer_json.get("gradient_checkpointing", False))
    dataloader_num_workers = int(trainer_json.get("dataloader_num_workers", 0))
    dataloader_persistent_workers = bool(
        trainer_json.get("dataloader_persistent_workers", False)
    ) and dataloader_num_workers > 0

    cpo_fields = {field.name for field in dataclasses.fields(CPOConfig)}
    eval_strategy = str(trainer_json.get("eval_strategy", "steps")) if eval_ds is not None else "no"

    cfg_kwargs = {
        "output_dir": output_dir,
        "report_to": trainer_json.get("report_to", "tensorboard"),
        "per_device_train_batch_size": int(trainer_json.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(trainer_json.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(trainer_json.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(trainer_json.get("num_train_epochs", 1)),
        "learning_rate": float(trainer_json.get("learning_rate", 1e-5)),
        "logging_steps": int(trainer_json.get("logging_steps", 50)),
        "save_steps": int(trainer_json.get("save_steps", 1000)),
        "eval_steps": int(trainer_json.get("eval_steps", 1000)),
        "eval_strategy": eval_strategy,
        "save_strategy": str(trainer_json.get("save_strategy", "steps")),
        "seed": seed,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": bool(trainer_json.get("tf32", False)),
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": trainer_json.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        "optim": str(trainer_json.get("optim", "adamw_torch")),
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_persistent_workers": dataloader_persistent_workers,
        "dataset_num_proc": cpo_json.get("dataset_num_proc"),
        "max_length": int(cpo_json.get("max_length", 8192)),
        "loss_type": str(cpo_json.get("loss_type", "simpo")),
        "cpo_alpha": float(cpo_json.get("cpo_alpha", 0.0)),
        "beta": float(cpo_json.get("beta", 0.1)),
        "simpo_gamma": float(cpo_json.get("simpo_gamma", 0.5)),
    }
    if "warmup_steps" in trainer_json:
        cfg_kwargs["warmup_steps"] = int(trainer_json["warmup_steps"])
    else:
        cfg_kwargs["warmup_ratio"] = float(trainer_json.get("warmup_ratio", 0.03))

    for section in (trainer_json, cpo_json):
        for key, value in section.items():
            if key in cpo_fields and key not in cfg_kwargs:
                cfg_kwargs[key] = value
    cfg_kwargs = {key: value for key, value in cfg_kwargs.items() if key in cpo_fields}

    if cfg_kwargs["loss_type"] != "simpo":
        raise ValueError("This script requires cpo.loss_type='simpo'")
    if cfg_kwargs["cpo_alpha"] != 0.0:
        raise ValueError("Pure SimPO requires cpo.cpo_alpha=0.0")
    if cfg_kwargs["beta"] <= 0 or cfg_kwargs["simpo_gamma"] < 0:
        raise ValueError("SimPO requires beta > 0 and simpo_gamma >= 0")

    simpo_ratio = cfg_kwargs["simpo_gamma"] / cfg_kwargs["beta"]
    print(
        f"Model={model_name}; attention={attn_implementation or 'model_default'}; "
        f"bf16={bf16}; fp16={fp16}; 4bit={load_in_4bit}; 8bit={load_in_8bit}"
    )
    print(
        f"SimPO: beta={cfg_kwargs['beta']}; gamma={cfg_kwargs['simpo_gamma']}; "
        f"gamma/beta={simpo_ratio:.4f}; max_length={cfg_kwargs['max_length']}"
    )
    if bool(config.get("debug", False)):
        print(f"Train sample before templating: {train_ds[0]}")

    trainer = CPOTrainer(
        model=model,
        args=CPOConfig(**cfg_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_from_checkpoint = trainer_json.get("resume_from_checkpoint")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

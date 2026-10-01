import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from trl.experimental.cpo import CPOConfig, CPOTrainer
from peft import LoraConfig, get_peft_model
import dataclasses

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps",type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")


    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--report_to", type=str, default="tensorboard")

    parser.add_argument("--debug", action="store_true")

    
    return parser

def load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(args):
    dataset = load_dataset('json', data_files=args.data)['train']

    cfg_json = load_json_config(args.config)
    trainer_json = cfg_json.get("trainer", {}) if isinstance(cfg_json, dict) else {}
    lora_json = cfg_json.get("lora", {}) if isinstance(cfg_json, dict) else {}
    cpo_json = cfg_json.get("cpo", {}) if isinstance(cfg_json, dict) else {}

    model_name = cfg_json.get("model_name") if isinstance(cfg_json, dict) else None
    if not model_name:
        model_name = args.model
    if not model_name:
        raise ValueError("Model is not specified. Provide --model or set model_name in --config JSON.")

    if args.debug:
        print(f"[Dataset samples] {dataset[:5]}")

    if args.val_split and args.val_split > 0:
        split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds, eval_ds = split['train'], split['test']
    else:
        train_ds = dataset
        eval_ds = None 

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    def preprocess(example):
        raw_prompt = example.get("prompt")
        chosen_text = example.get("chosen")
        rejected_text = example.get("rejected")

        user_messages = [{"role": "user", "content": raw_prompt}]
        prompt = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)

        chosen = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": chosen_text}],
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": rejected_text}],
            tokenize=False,
            add_generation_prompt=False,
        )

        eos = tokenizer.eos_token or ""
        if eos:
            if not chosen.endswith(eos):
                chosen += eos
            if not rejected.endswith(eos):
                rejected += eos

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)

    if eval_ds is not None:
        eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    model_kwargs = {}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    bf16_flag = bool(trainer_json.get("bf16", args.bf16))
    fp16_flag = bool(trainer_json.get("fp16", args.fp16))
    if bf16_flag:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16_flag:
        model_kwargs["torch_dtype"] = torch.float16

    if bool(cfg_json.get("load_in_8bit", False)):
        model_kwargs["load_in_8bit"] = True
    if bool(cfg_json.get("load_in_4bit", False)):
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    gradient_ckpt = bool(cfg_json.get("gradient_checkpointing", args.gradient_checkpointing))
    if gradient_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if isinstance(lora_json.get("target_modules", None), list):
        target_modules = lora_json.get("target_modules")
    else:
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]

    peft_config = LoraConfig(
        r=int(lora_json.get("r", args.lora_rank)),
        lora_alpha=int(lora_json.get("lora_alpha", args.lora_alpha)),
        lora_dropout=float(lora_json.get("lora_dropout", args.lora_dropout)),
        bias=str(lora_json.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=lora_json.get("modules_to_save", None),
    )

    model = get_peft_model(model, peft_config)

    modules_to_save = lora_json.get("modules_to_save", []) or []
    if getattr(model.config, "tie_word_embeddings", False) and "lm_head" in modules_to_save:
        assert not ("lm_head" in modules_to_save and "embed_tokens" in modules_to_save), "Cannot include both 'lm_head' and 'embed_tokens' in modules_to_save when tie_word_embeddings=True"

        try:
            model.base_model.model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight
        except Exception:
            model.base_model.model.embed_tokens.weight = model.base_model.model.lm_head.modules_to_save["default"].weight

    cpo_fields = {f.name for f in dataclasses.fields(CPOConfig)}

    simpo_gamma_cli = args.simpo_gamma

    cfg_kwargs = dict(
        output_dir=args.output_dir,
        report_to=args.report_to,

        per_device_train_batch_size=int(trainer_json.get("per_device_train_batch_size", args.batch_size)),
        per_device_eval_batch_size=int(trainer_json.get("per_device_eval_batch_size", args.batch_size)),
        gradient_accumulation_steps=int(trainer_json.get("gradient_accumulation_steps", args.grad_accum)),
        num_train_epochs=float(trainer_json.get("num_train_epochs", args.epochs)),
        learning_rate=float(trainer_json.get("learning_rate", args.lr)),
        logging_steps=int(trainer_json.get("logging_steps", args.logging_steps)),
        save_steps=int(trainer_json.get("save_steps", args.save_steps)),
        eval_steps=int(trainer_json.get("eval_steps", args.eval_steps)),
        eval_strategy=str(trainer_json.get("eval_strategy", "steps")),
        save_strategy="steps",
        seed=int(trainer_json.get("seed", args.seed)),

        bf16=bool(trainer_json.get("bf16", args.bf16)),
        fp16=bool(trainer_json.get("fp16", args.fp16)),

        max_length=int(cpo_json.get("max_length", args.max_length)),

        loss_type=str(cpo_json.get("loss_type", "simpo")),
        cpo_alpha=float(cpo_json.get("cpo_alpha", 0.0)),
        beta=float(cpo_json.get("beta", args.beta)),
        simpo_gamma=float(cpo_json.get("simpo_gamma", simpo_gamma_cli)),
    )

    if "warmup_steps" in trainer_json:
        cfg_kwargs["warmup_steps"] = int(trainer_json["warmup_steps"])
    else:
        cfg_kwargs["warmup_ratio"] = float(trainer_json.get("warmup_ratio", args.warmup_ratio))

    for k, v in trainer_json.items():
        if k in cpo_fields and k not in cfg_kwargs:
            cfg_kwargs[k] = v
    for k, v in cpo_json.items():
        if k in cpo_fields and k not in cfg_kwargs:
            cfg_kwargs[k] = v

    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in cpo_fields}

    cfg = CPOConfig(**cfg_kwargs)

    trainer = CPOTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer)

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__=="__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)

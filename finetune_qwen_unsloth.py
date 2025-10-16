# finetune_qwen_unsloth.py
# Minimal Unsloth + TRL SFT for Qwen3/Qwen2.5 8B with LoRA (4-bit)
# Usage:
#   pip install "unsloth>=2024.10.0" "transformers>=4.43.0" "accelerate>=0.33.0" trl datasets peft bitsandbytes
#   python finetune_qwen_unsloth.py \
#       --model_id Qwen/Qwen3-8B-Instruct \
#       --train_jsonl ./train.jsonl \
#       --output_dir ./qwen3_8b_lora \
#       --epochs 1 \
#       --max_steps 200
from unsloth import FastLanguageModel
import argparse, json, os, math
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
#from unsloth import FastLanguageModel  # Unsloth magic

# ------------- Args -------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str,
                    default="Qwen/Qwen3-8B",
                    help="Use your exact model, e.g. Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--train_jsonl", type=str, required=True,
                    help="Path to JSONL with fields: instruction,input,output")
    ap.add_argument("--output_dir", type=str, default="./qwen_lora_out")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="If > 0, overrides epochs")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--bf16", action="store_true", help="Use bf16 if available")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ------------- Prompting -------------
def build_chat(sample: Dict[str, str]) -> List[Dict[str, str]]:
    """Create a Qwen-style chat list for tokenizer.apply_chat_template."""
    instruction = (sample.get("instruction") or "").strip()
    inp = (sample.get("input") or "").strip()
    out = (sample.get("output") or sample.get("label") or "").strip()

    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": out},
    ]

def format_example(ex, tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    # Turn chat into a single LM sequence (labels shift handled by trainer)
    messages = build_chat(ex)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # include assistant completion in labels
    )
    return {"text": text}

# ------------- Main -------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model & tokenizer via Unsloth (4-bit QLoRA friendly)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=None,            # Auto-select (bf16 if supported)
        load_in_4bit=True,     # 24GB-friendly
        token=None,            # HF token if needed; or set HF_TOKEN env var
    )

    # Qwen tokenizer specifics
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing & FlashAttn via Unsloth already handled
    # Add LoRA adapters
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,                   # e.g. 8
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,      # e.g. 16
        lora_dropout=args.lora_dropout,  # e.g. 0.05
        bias="none",
    )
    # Load dataset
    # Your JSONL must have fields: instruction, input, output
    raw = load_dataset("json", data_files={"train": args.train_jsonl})
    if args.eval_fraction and args.eval_fraction > 0:
        raw = raw["train"].train_test_split(test_size=args.eval_fraction, seed=args.seed)
        ds_train, ds_eval = raw["train"], raw["test"]
    else:
        ds_train, ds_eval = raw["train"], None

    ds_train = ds_train.map(lambda ex: format_example(ex, tokenizer), remove_columns=ds_train.column_names)
    if ds_eval:
        ds_eval = ds_eval.map(lambda ex: format_example(ex, tokenizer), remove_columns=ds_eval.column_names)

    # Collator: standard LM collator (labels = input_ids)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training args
    steps_per_epoch = None
    if args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = 1
    else:
        # rough steps/epoch estimate (for logging)
        steps_per_epoch = math.ceil(len(ds_train) / (args.batch_size * args.grad_accum))
        max_steps = -1
        num_train_epochs = args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=not args.bf16,
        optim="paged_adamw_8bit",          # 8-bit optimizer to save VRAM
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        torch_compile=False,
        report_to="none",
        run_name="qwen_unsloth_lora",
    )

    # TRL SFT Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = 16, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 1 , # Set this for 1 full training run.
            max_steps = 5000,
            learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 4545,
            report_to = "wandb", # Use this for WandB etc
            #logging_steps=10,
            run_name="qwen3-8b-16-10-sft",
        ),
    )

    trainer.train()

    # Save the adapter (small) + merged full if you want
    trainer.save_model(args.output_dir)  # saves PEFT adapter
    tokenizer.save_pretrained(args.output_dir)

    # Optionally merge LoRA into base weights for export (VRAM-heavy)
    # from peft import AutoPeftModelForCausalLM
    # merged = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16)
    # merged = merged.merge_and_unload()
    # merged.save_pretrained(os.path.join(args.output_dir, "merged"), safe_serialization=True)

    # -------- Quick sanity inference --------
    print("\n=== Inference sanity check ===")
    from transformers import TextStreamer
    # Switch to inference mode (Unsloth helper)
    FastLanguageModel.for_inference(model)
    prompt_chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the benefits of LoRA for fine-tuning large models."},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )
    print("\nDone. Adapters saved at:", args.output_dir)

if __name__ == "__main__":
    main()


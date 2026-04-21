"""Qwen + LoRA SFT for 5-class intent labeling.

End-to-end minimal pipeline:
  1. Load JSONL of (prior_turns, current_turn, session, label, reason) records
  2. Render into chat messages (system → user → assistant JSON)
  3. Apply LoRA to the base model
  4. Train with trl.SFTTrainer
  5. Save adapter
  6. Run a smoke-test inference on one sample

Usage:
    uv run --extra train python scripts/train_qwen_lora.py \\
        --data   data/training/seed_v1.jsonl \\
        --model  Qwen/Qwen2.5-0.5B-Instruct \\
        --output models/qwen-intent-lora-v1 \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from sorting_hat.labeling.prompts import SYSTEM_PROMPT, USER_TEMPLATE


def build_user(ex: dict) -> str:
    prior = ex.get("prior_turns") or []
    if prior:
        prior_str = "\n".join(
            f"[{p.get('role', '?')}] {(p.get('text') or '').strip()}" for p in prior
        )
    else:
        prior_str = "(no prior turns)"
    ct = ex["current_turn"]
    sess = ex.get("session") or {}
    return USER_TEMPLATE.format(
        interview_mode=sess.get("interview_mode") or "-",
        programming_language=sess.get("programming_language") or "-",
        goal_position=sess.get("goal_position") or "-",
        goal_company=sess.get("goal_company") or "-",
        prior_turns=prior_str,
        role=ct.get("role") or "-",
        source=ct.get("source") or "-",
        text=(ct.get("text") or "").strip(),
    )


def build_assistant(ex: dict) -> str:
    out = {
        "label": ex["label"],
        "confidence": ex.get("confidence", 0.95),
        "reason": ex.get("reason", ""),
        "secondary_label": ex.get("secondary_label"),
    }
    return json.dumps(out, ensure_ascii=False)


def load_dataset(path: pathlib.Path) -> Dataset:
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        ex = json.loads(line)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user(ex)},
                    {"role": "assistant", "content": build_assistant(ex)},
                ]
            }
        )
    return Dataset.from_list(rows)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="default tiny for smoke test; switch to Qwen/Qwen3-4B for real run")
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "models" / "qwen-intent-lora-v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--no-smoke-test", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    # MPS bf16 support is spotty; use fp32 there for reliability on smoke tests
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading tokenizer + model ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print("[2/5] building dataset ...", file=sys.stderr)
    ds = load_dataset(args.data)
    print(f"    samples: {len(ds)}", file=sys.stderr)

    print("[3/5] configuring LoRA ...", file=sys.stderr)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(device == "cuda"),
        fp16=False,
        report_to=[],
        gradient_checkpointing=False,
        max_length=args.max_seq_len,
        completion_only_loss=True,          # mask prompt, only loss on assistant response
        assistant_only_loss=True,           # belt-and-suspenders
    )

    print("[4/5] starting training ...", file=sys.stderr)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=sft_cfg,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"[5/5] saved adapter to {args.output}", file=sys.stderr)

    if args.no_smoke_test:
        return

    print("\n--- smoke-test inference on first training sample ---", file=sys.stderr)
    first = ds[0]
    messages = first["messages"][:-1]  # system + user only (drop gold assistant)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    with torch.no_grad():
        out = trainer.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPROMPT (last 400 chars):\n...{prompt[-400:]}", file=sys.stderr)
    print(f"\nGOLD:     {first['messages'][-1]['content']}", file=sys.stderr)
    print(f"PREDICTED: {generated.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()

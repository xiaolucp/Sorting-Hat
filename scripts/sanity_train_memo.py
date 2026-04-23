"""Sanity check: read rows directly from training JSONL (combined_v2.jsonl format),
render them exactly like train_qwen_lora.py does, and ask the adapter to predict
the gold label. Pure memorization test — if this isn't near-perfect, training
didn't do what we expected.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sorting_hat.labeling.prompts import SYSTEM_PROMPT, USER_TEMPLATE

LABELS = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]


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


def parse_label(raw: str):
    raw = raw.strip()
    i = raw.find("{")
    j = raw.rfind("}")
    if i == -1 or j == -1:
        return None, raw[:200]
    try:
        return json.loads(raw[i : j + 1]).get("label"), None
    except Exception:
        return None, raw[i : j + 1][:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", type=pathlib.Path, required=True,
                    help="training-format jsonl (combined_v2.jsonl)")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()

    # stratified sample from training jsonl
    by_label = defaultdict(list)
    for line in args.training.open():
        if not line.strip():
            continue
        d = json.loads(line)
        by_label[d["label"]].append(d)
    rng = random.Random(args.seed)
    per = max(1, args.n // len(LABELS))
    sample = []
    for lbl in LABELS:
        pool = by_label.get(lbl, [])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    rng.shuffle(sample)
    print(f"sampled {len(sample)}: {Counter(r['label'] for r in sample)}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()

    prompts = []
    for ex in sample:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user(ex)},
        ]
        p = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(p)

    preds = []
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i : i + args.batch_size]
        enc = tok(chunk, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        prefix_len = enc.input_ids.shape[1]
        for j, row_out in enumerate(out):
            gen = tok.decode(row_out[prefix_len:], skip_special_tokens=True)
            pred_label, raw = parse_label(gen)
            gold = sample[i + j]["label"]
            preds.append({"gold": gold, "pred": pred_label, "text": sample[i+j]["current_turn"]["text"][:80], "raw": raw})
        done = len(preds)
        print(f"  {done}/{len(prompts)}", file=sys.stderr, flush=True)

    correct = sum(1 for p in preds if p["pred"] == p["gold"])
    print(f"\nTRAINING MEMORIZATION acc={correct}/{len(preds)}={correct/len(preds):.3f}")
    cm = {g: Counter() for g in LABELS}
    for p in preds:
        pr = p["pred"] if p["pred"] in LABELS else "UNP"
        cm[p["gold"]][pr] += 1
    print("\nconfusion (gold → pred):")
    header = "           " + " ".join(f"{l[:8]:>10s}" for l in LABELS + ["UNP"])
    print(header)
    for g in LABELS:
        row = f"{g[:10]:<10s} " + " ".join(f"{cm[g].get(p,0):>10d}" for p in LABELS + ["UNP"])
        print(row)

    print("\nfirst 5 mismatches:")
    m = 0
    for p in preds:
        if p["pred"] != p["gold"] and m < 5:
            print(f"  gold={p['gold']:16s} pred={p['pred']!s:16s} | {p['text']}")
            if p["raw"]:
                print(f"    raw: {p['raw']}")
            m += 1


if __name__ == "__main__":
    main()

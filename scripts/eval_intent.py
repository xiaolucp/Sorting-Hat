"""Quick eval: run a trained LoRA adapter against a held-out labeled day,
compare to gpt-4o labels (the labeling-pipeline output).

Uses the same prompts + formatting as train_qwen_lora.py / build_training_data.py.

Usage:
    uv run --extra train python scripts/eval_intent.py \\
        --base Qwen/Qwen3-4B \\
        --adapter models/qwen3-4b-intent-lora-v3 \\
        --labeled data/labeled/turns_2026-04-02.jsonl \\
        --raw-dir data/raw \\
        --n 200 \\
        --out runs/eval_v3_04-02.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sorting_hat.labeling.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT, USER_TEMPLATE

LABELS = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]


def load_raw_sessions(raw_dir: pathlib.Path) -> dict:
    out = {}
    for p in sorted(raw_dir.glob("frai_sessions_*.json")):
        for s in json.loads(p.read_text()):
            sid = s.get("session_id")
            if sid:
                out[sid] = s
    return out


def find_turn_index(session: dict, turn_id: str, text: str):
    turns = session.get("turns") or []
    for i, t in enumerate(turns):
        if t.get("trace_id") == turn_id:
            return i
    sid = session.get("session_id")
    if turn_id.startswith(f"{sid}:"):
        try:
            idx = int(turn_id.split(":", 1)[1])
            if 0 <= idx < len(turns):
                return idx
        except ValueError:
            pass
    text = text.strip()
    for i, t in enumerate(turns):
        if t.get("role") == "interviewer" and (t.get("text") or "").strip() == text:
            return i
    return None


def build_user(row: dict, prior_turns: list[dict]) -> str:
    if prior_turns:
        prior_str = "\n".join(
            f"[{p.get('role','?')}] {(p.get('text') or '').strip()}" for p in prior_turns
        )
    else:
        prior_str = "(no prior turns)"
    return USER_TEMPLATE.format(
        interview_mode=row.get("interview_mode") or "-",
        programming_language=row.get("programming_language") or "-",
        goal_position=row.get("goal_position") or "-",
        goal_company=row.get("goal_company") or "-",
        prior_turns=prior_str,
        role=row.get("role") or "interviewer",
        source=row.get("source") or "-",
        text=(row.get("text") or "").strip(),
    )


def parse_label(raw: str) -> dict:
    raw = raw.strip()
    # strip possible code fence
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    # find first { ... last }
    i = raw.find("{")
    j = raw.rfind("}")
    if i == -1 or j == -1:
        return {"label": None, "raw": raw}
    try:
        return json.loads(raw[i : j + 1])
    except Exception:
        return {"label": None, "raw": raw[: 200]}


def stratified_sample(
    rows: list[dict], n_per_class: dict[str, int], seed: int
) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out = []
    for lbl, k in n_per_class.items():
        pool = by_label.get(lbl, [])
        rng.shuffle(pool)
        out.extend(pool[:k])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", type=pathlib.Path, default=None,
                    help="LoRA adapter dir; omit to eval the base model as-is")
    ap.add_argument("--labeled", type=pathlib.Path, required=True)
    ap.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/raw"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-confidence", type=float, default=0.9)
    ap.add_argument("--context-window", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable Qwen3 <think> block in chat template (default off)")
    ap.add_argument("--prompt", choices=["short", "full"], default="full",
                    help="system prompt variant: 'short' for fine-tuned LoRA, 'full' for base model")
    args = ap.parse_args()

    print(f"[1/4] loading labeled + raw sessions ...", file=sys.stderr)
    rows = []
    for line in args.labeled.open():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("confidence", 1.0) < args.min_confidence:
            continue
        if d.get("label") not in LABELS:
            continue
        rows.append(d)
    print(f"  {len(rows)} rows with conf>={args.min_confidence}", file=sys.stderr)

    # stratified sample: up to n/5 per class, fall back to available
    per = max(1, args.n // len(LABELS))
    n_per = {l: per for l in LABELS}
    sample = stratified_sample(rows, n_per, args.seed)[: args.n]
    print(f"  sampled {len(sample)}: {Counter(r['label'] for r in sample)}", file=sys.stderr)

    sessions = load_raw_sessions(args.raw_dir)
    print(f"  {len(sessions)} raw sessions indexed", file=sys.stderr)

    print(f"[2/4] loading model {args.base}"
          + (f" + adapter {args.adapter}" if args.adapter else " (base, no adapter)")
          + " ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # causal-LM batch generation requires left padding
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    if args.adapter is not None:
        model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    else:
        model = base.eval()

    print(f"[3/4] building prompts ...", file=sys.stderr)
    # Pre-build every (row, prompt) so we can batch the generate call cleanly.
    jobs: list[tuple[dict, str]] = []
    for row in sample:
        sid = row.get("session_id")
        session = sessions.get(sid)
        if not session:
            continue
        idx = find_turn_index(session, row.get("turn_id", ""), row.get("text", ""))
        if idx is None:
            continue
        turns = session.get("turns") or []
        prior_raw = turns[max(0, idx - args.context_window * 4) : idx]
        prior_turns = [
            {"role": p.get("role") or "?", "text": (p.get("text") or "").strip()}
            for p in prior_raw
            if (p.get("text") or "").strip()
        ]
        user = build_user(row, prior_turns)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SHORT if args.prompt == "short" else SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        jobs.append((row, prompt))
    print(f"  {len(jobs)} prompts built (skipped {len(sample) - len(jobs)})", file=sys.stderr)

    print(f"  generating, batch_size={args.batch_size} ...", file=sys.stderr)
    preds = []
    t0 = time.time()
    for i in range(0, len(jobs), args.batch_size):
        chunk = jobs[i : i + args.batch_size]
        prompts = [p for _, p in chunk]
        enc = tok(prompts, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        # left-padded → every row shares the same prefix length
        prefix_len = enc.input_ids.shape[1]
        for (row, _), row_out in zip(chunk, out):
            gen = tok.decode(row_out[prefix_len:], skip_special_tokens=True)
            parsed = parse_label(gen)
            preds.append(
                {
                    "turn_id": row.get("turn_id"),
                    "text": row.get("text"),
                    "gold": row["label"],
                    "gold_reason": row.get("reason"),
                    "pred": parsed.get("label"),
                    "pred_conf": parsed.get("confidence"),
                    "pred_reason": parsed.get("reason"),
                    "raw": gen if parsed.get("label") is None else None,
                }
            )
        done = len(preds)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0.0
        eta = (len(jobs) - done) / rate if rate else 0.0
        print(f"  {done}/{len(jobs)}  {rate:.2f} samp/s  eta {eta:.0f}s", file=sys.stderr, flush=True)

    print(f"[4/4] scoring ...", file=sys.stderr)
    correct = sum(1 for p in preds if p["pred"] == p["gold"])
    total = len(preds)
    acc = correct / total if total else 0.0

    # confusion matrix
    cm = {g: Counter() for g in LABELS}
    cm["UNPARSED"] = Counter()
    for p in preds:
        g = p["gold"]
        pr = p["pred"] if p["pred"] in LABELS else "UNPARSED"
        if pr == "UNPARSED":
            cm["UNPARSED"][g] += 1
        else:
            cm[g][pr] += 1

    # per-class P/R
    per_class = {}
    for lbl in LABELS:
        tp = sum(1 for p in preds if p["gold"] == lbl and p["pred"] == lbl)
        fp = sum(1 for p in preds if p["gold"] != lbl and p["pred"] == lbl)
        fn = sum(1 for p in preds if p["gold"] == lbl and p["pred"] != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[lbl] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}

    summary = {
        "n": total,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix_gold_to_pred": {g: dict(cm[g]) for g in cm},
        "unparsed": sum(1 for p in preds if p["pred"] not in LABELS),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "predictions": preds}, f, ensure_ascii=False, indent=2)

    print("\n================ RESULT ================")
    print(f"n={total}  accuracy={acc:.3f}  unparsed={summary['unparsed']}")
    print("\nper-class:")
    for lbl, s in per_class.items():
        print(f"  {lbl:18s}  P={s['precision']:.3f}  R={s['recall']:.3f}  (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    print("\nconfusion (gold rows → pred cols):")
    header = "           " + " ".join(f"{l[:8]:>10s}" for l in LABELS + ["UNPARSED"])
    print(header)
    for g in LABELS:
        row = f"{g[:10]:<10s} " + " ".join(f"{cm[g].get(p,0):>10d}" for p in LABELS + ["UNPARSED"])
        print(row)
    print(f"\nwrote full report → {args.out}")


if __name__ == "__main__":
    main()

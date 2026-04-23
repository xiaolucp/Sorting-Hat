"""Run v5 LoRA model on every labeled interviewer turn of a day.

Writes predictions to data/predictions/turns_v5_YYYY-MM-DD.jsonl with the same
turn_id keys as the gpt-4o labeled file, plus `pred_label` / `pred_confidence`
/ `pred_reason` fields.

Usage:
    uv run --no-sync python scripts/predict_day.py \\
        --base Qwen/Qwen3-4B \\
        --adapter models/qwen3-4b-intent-lora-v5-short \\
        --date 2026-04-11 \\
        --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sorting_hat.labeling.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHORT,
    USER_TEMPLATE,
)

LABELS = {"coding", "system_design", "project_qa", "chat", "no_answer_needed"}


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
    prior_str = (
        "\n".join(
            f"[{p.get('role','?')}] {(p.get('text') or '').strip()}"
            for p in prior_turns
            if (p.get("text") or "").strip()
        )
        or "(no prior turns)"
    )
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
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    i = raw.find("{")
    j = raw.rfind("}")
    if i == -1 or j == -1:
        return {}
    try:
        return json.loads(raw[i : j + 1])
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--date", required=True, help="e.g. 2026-04-11")
    ap.add_argument("--raw-dir", type=pathlib.Path, default=ROOT / "data" / "raw")
    ap.add_argument("--labeled-dir", type=pathlib.Path, default=ROOT / "data" / "labeled")
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "data" / "predictions")
    ap.add_argument("--prompt", choices=["short", "full"], default="short")
    ap.add_argument("--context-window", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--resume", action="store_true",
                    help="skip turn_ids already present in the output file")
    args = ap.parse_args()

    labeled_path = args.labeled_dir / f"turns_{args.date}.jsonl"
    out_path = args.out_dir / f"turns_v5_{args.date}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not labeled_path.exists():
        print(f"[error] not found: {labeled_path}", file=sys.stderr)
        sys.exit(1)

    print(f"loading labeled turns from {labeled_path} ...", file=sys.stderr)
    rows = [json.loads(l) for l in labeled_path.open() if l.strip()]
    print(f"  {len(rows)} rows", file=sys.stderr)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                try:
                    done_ids.add(json.loads(line).get("turn_id"))
                except Exception:
                    pass
        print(f"  resuming: {len(done_ids)} already predicted, will skip", file=sys.stderr)
    rows = [r for r in rows if r.get("turn_id") not in done_ids]
    print(f"  to predict: {len(rows)}", file=sys.stderr)

    sessions = load_raw_sessions(args.raw_dir)
    print(f"  indexed {len(sessions)} raw sessions", file=sys.stderr)

    sys_prompt = SYSTEM_PROMPT_SHORT if args.prompt == "short" else SYSTEM_PROMPT
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    print(f"model loaded: base={args.base}  adapter={args.adapter}  prompt={args.prompt}",
          file=sys.stderr)

    # Build all prompts
    jobs = []
    for row in rows:
        sid = row.get("session_id")
        session = sessions.get(sid)
        if not session:
            continue
        idx = find_turn_index(session, row.get("turn_id", ""), row.get("text", ""))
        if idx is None:
            continue
        turns = session.get("turns") or []
        prior_raw = turns[max(0, idx - args.context_window * 4) : idx]
        prior = [
            {"role": p.get("role") or "?", "text": (p.get("text") or "").strip()}
            for p in prior_raw
            if (p.get("text") or "").strip()
        ]
        user = build_user(row, prior)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        jobs.append((row, prompt))
    print(f"  prompts built: {len(jobs)}", file=sys.stderr)

    # Generate in batches, stream-write each result immediately (crash-safe)
    t0 = time.time()
    with out_path.open("a") as fout:
        for i in range(0, len(jobs), args.batch_size):
            chunk = jobs[i : i + args.batch_size]
            prompts = [p for _, p in chunk]
            enc = tok(
                prompts, padding=True, return_tensors="pt",
                truncation=True, max_length=args.max_seq_len,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            prefix_len = enc.input_ids.shape[1]
            for (row, _), row_out in zip(chunk, out):
                gen = tok.decode(row_out[prefix_len:], skip_special_tokens=True)
                parsed = parse_label(gen)
                rec = {
                    "turn_id": row.get("turn_id"),
                    "session_id": row.get("session_id"),
                    "text": row.get("text"),
                    "role": row.get("role"),
                    "gold_label": row.get("label"),
                    "gold_confidence": row.get("confidence"),
                    "gold_reason": row.get("reason"),
                    "pred_label": parsed.get("label"),
                    "pred_confidence": parsed.get("confidence"),
                    "pred_reason": parsed.get("reason"),
                    "pred_secondary": parsed.get("secondary_label"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done = i + len(chunk)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(jobs) - done) / rate if rate else 0
            print(f"  {done}/{len(jobs)}  {rate:.2f} samp/s  eta {eta:.0f}s",
                  file=sys.stderr, flush=True)

    print(f"\nwrote predictions → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

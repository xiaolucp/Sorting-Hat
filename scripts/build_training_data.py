"""Convert flat labeled JSONL (data/labeled/turns_*.jsonl) into training format
that train_qwen_lora.py expects: {prior_turns, current_turn, session, label,
confidence, reason, secondary_label}.

Rebuilds prior_turns by looking up the raw session and slicing the same window
label_sessions.py used (context_window * 4 raw turns prior, no merging here —
we pass the raw prior turns through so the training example mirrors inference
input exactly).

Usage:
    uv run python scripts/build_training_data.py \\
        --labeled data/labeled/turns_2026-04-20.snapshot.jsonl \\
        --raw-dir data/raw \\
        --out data/training/from_labeled_2026-04-20.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict


def load_raw_sessions(raw_dir: pathlib.Path) -> dict[str, dict]:
    """session_id -> session dict, across all frai_sessions_*.json files."""
    out: dict[str, dict] = {}
    for p in sorted(raw_dir.glob("frai_sessions_*.json")):
        data = json.loads(p.read_text())
        for s in data:
            sid = s.get("session_id")
            if sid:
                out[sid] = s
    return out


def find_turn_index(session: dict, turn_id: str, text: str) -> int | None:
    """Match labeled turn back to its index in the raw session turns."""
    sid = session.get("session_id")
    turns = session.get("turns") or []
    # Primary: trace_id match
    for i, t in enumerate(turns):
        if t.get("trace_id") == turn_id:
            return i
    # Fallback: "{sid}:{idx}" form
    if turn_id.startswith(f"{sid}:"):
        try:
            idx = int(turn_id.split(":", 1)[1])
            if 0 <= idx < len(turns):
                return idx
        except ValueError:
            pass
    # Last resort: text match (first interviewer turn with same text)
    for i, t in enumerate(turns):
        if t.get("role") == "interviewer" and (t.get("text") or "").strip() == text.strip():
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", type=pathlib.Path, required=True,
                    help="flat labeled JSONL (LabeledTurn rows)")
    ap.add_argument("--raw-dir", type=pathlib.Path,
                    default=pathlib.Path("data/raw"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--context-window", type=int, default=6,
                    help="same as label_sessions.py; prior_raw = 4*this before current")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop rows below this confidence (0 keeps all)")
    args = ap.parse_args()

    if not args.labeled.exists():
        print(f"[error] not found: {args.labeled}", file=sys.stderr)
        sys.exit(1)

    print(f"loading raw sessions from {args.raw_dir} ...", file=sys.stderr)
    sessions = load_raw_sessions(args.raw_dir)
    print(f"  {len(sessions)} sessions indexed", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_no_session = skipped_no_turn = skipped_low_conf = 0
    label_counts: dict[str, int] = defaultdict(int)

    with args.out.open("w") as fout:
        for line in args.labeled.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("confidence", 1.0) < args.min_confidence:
                skipped_low_conf += 1
                continue

            sid = row.get("session_id")
            session = sessions.get(sid)
            if not session:
                skipped_no_session += 1
                continue

            idx = find_turn_index(session, row.get("turn_id", ""), row.get("text", ""))
            if idx is None:
                skipped_no_turn += 1
                continue

            turns = session["turns"]
            prior_raw = turns[max(0, idx - args.context_window * 4): idx]
            prior_turns = [
                {
                    "role": p.get("role") or "?",
                    "text": (p.get("text") or "").strip(),
                }
                for p in prior_raw
                if (p.get("text") or "").strip()
            ]

            rec = {
                "prior_turns": prior_turns,
                "current_turn": {
                    "role": row.get("role") or "interviewer",
                    "text": row.get("text") or "",
                    "source": row.get("source"),
                    "trace_id": row.get("turn_id"),
                },
                "session": {
                    "interview_mode": row.get("interview_mode"),
                    "programming_language": row.get("programming_language"),
                    "goal_position": row.get("goal_position"),
                    "goal_company": session.get("goal_company"),
                },
                "label": row.get("label"),
                "confidence": row.get("confidence"),
                "reason": row.get("reason"),
                "secondary_label": row.get("secondary_label"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            label_counts[row.get("label", "?")] += 1

    print(f"\nwritten={written} skipped_no_session={skipped_no_session} "
          f"skipped_no_turn={skipped_no_turn} skipped_low_conf={skipped_low_conf}",
          file=sys.stderr)
    print(f"label counts: {dict(label_counts)}", file=sys.stderr)
    print(f"output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

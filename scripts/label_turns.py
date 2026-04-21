"""Label interviewer turns from a `frai_turns_YYYY-MM-DD.jsonl` file.

For each interviewer turn, builds the prior-turn context window from the same
session, calls the LLM labeler, and writes a JSONL of labeled turns.

Usage:
    uv run python scripts/label_turns.py \\
        --input  data/raw/frai_turns_2026-04-19.jsonl \\
        --output data/labeled/turns_2026-04-19.jsonl \\
        --model  claude-opus-4-7 \\
        --limit  200          # optional — test on a subset first
        --min-len 15          # skip very short interviewer utterances
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for env in (ROOT / ".env.local", ROOT / ".env"):
    if env.exists():
        load_dotenv(env)
        break

from sorting_hat.labeling import IntentLabeler, SessionContext, TurnInput  # noqa: E402


def iter_turns(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--limit", type=int, default=None, help="max interviewer turns to label")
    ap.add_argument("--min-len", type=int, default=10, help="skip interviewer turns shorter than this")
    ap.add_argument("--context-window", type=int, default=6, help="prior turns to include as context")
    ap.add_argument("--roles", nargs="+", default=["interviewer"], help="roles to label")
    args = ap.parse_args()

    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    api_base = os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_API_BASE")
    assert api_key, "No API key found (check LITELLM_API_KEY in .env.local)"
    print(f"api_base: {api_base or '(default provider)'}  model: {args.model}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Group all turns by session_id so we can build per-turn prior context
    print(f"loading {args.input.name} ...", file=sys.stderr)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in iter_turns(args.input):
        by_session[r["session_id"]].append(r)
    for sid in by_session:
        by_session[sid].sort(key=lambda r: r.get("transcript_timestamp") or "")

    total_turns = sum(len(v) for v in by_session.values())
    print(f"sessions={len(by_session)}  total_turns={total_turns:,}", file=sys.stderr)

    # Build the label queue: (turn, prior_turns, session_context)
    queue = []
    for sid, turns in by_session.items():
        session = SessionContext(
            session_id=sid,
            interview_mode=turns[0].get("interview_mode"),
            programming_language=turns[0].get("programming_language"),
            goal_position=turns[0].get("goal_position"),
            goal_company=turns[0].get("goal_company"),
        )
        for idx, t in enumerate(turns):
            text = (t.get("transcript") or "").strip()
            if t.get("role") not in args.roles:
                continue
            if len(text) < args.min_len:
                continue
            prior_raw = turns[max(0, idx - args.context_window) : idx]
            prior = [
                TurnInput(
                    turn_id=f"{sid}:{i}",
                    text=p.get("transcript") or "",
                    role=p.get("role") or "?",
                    source=p.get("source"),
                    timestamp=p.get("transcript_timestamp"),
                )
                for i, p in enumerate(prior_raw)
            ]
            turn = TurnInput(
                turn_id=t.get("trace_id") or f"{sid}:{idx}",
                text=text,
                role=t.get("role") or "interviewer",
                source=t.get("source"),
                timestamp=t.get("transcript_timestamp"),
            )
            queue.append((turn, prior, session))

    if args.limit:
        queue = queue[: args.limit]
    print(f"labeling {len(queue):,} turns → {args.output.name}", file=sys.stderr)

    labeler = IntentLabeler(model=args.model)
    written = 0
    errors = 0
    with args.output.open("w") as out:
        for rec in tqdm(labeler.label_many(iter(queue)), total=len(queue)):
            try:
                out.write(rec.model_dump_json() + "\n")
                written += 1
            except Exception as e:
                errors += 1
                print(f"write error: {e}", file=sys.stderr)

    print(f"done. written={written}  errors={errors}", file=sys.stderr)


if __name__ == "__main__":
    main()

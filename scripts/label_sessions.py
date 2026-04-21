"""Label interviewer turns from frai_sessions_YYYY-MM-DD.json files.

Reads the session JSON files already in data/raw/, labels every interviewer
turn (with prior-turn context), and writes per-turn labeled JSONL to
data/labeled/turns_YYYY-MM-DD.jsonl.

Usage:
    uv run python scripts/label_sessions.py --date 2026-04-20
    uv run python scripts/label_sessions.py --date 2026-04-20 --limit 50
    uv run python scripts/label_sessions.py --date 2026-04-20 --model azure/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for env in (ROOT / ".env.local", ROOT / ".env"):
    if env.exists():
        load_dotenv(env)
        break

from sorting_hat.labeling import IntentLabeler, SessionContext, TurnInput  # noqa: E402


def iter_sessions(path: pathlib.Path):
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


def build_queue(
    sessions: list[dict],
    roles: list[str],
    min_len: int,
    context_window: int,
    modes: list[str] | None = None,
) -> list[tuple[TurnInput, list[TurnInput], SessionContext]]:
    queue = []
    for s in sessions:
        if modes and s.get("interview_mode") not in modes:
            continue
        sid = s["session_id"]
        turns = s.get("turns") or []
        session = SessionContext(
            session_id=sid,
            interview_mode=s.get("interview_mode"),
            programming_language=s.get("programming_language"),
            goal_position=s.get("goal_position"),
            goal_company=s.get("goal_company"),
        )
        for idx, t in enumerate(turns):
            text = (t.get("text") or "").strip()
            if t.get("role") not in roles:
                continue
            if len(text) < min_len:
                continue
            # Fetch more raw turns than context_window; _format_prior_turns will
            # merge consecutive same-role fragments first, then take the last
            # context_window merged slots — so we cover more actual history.
            prior_raw = turns[max(0, idx - context_window * 4): idx]
            prior = [
                TurnInput(
                    turn_id=f"{sid}:{i}",
                    text=(p.get("text") or "").strip(),
                    role=p.get("role") or "?",
                    source=p.get("source"),
                    timestamp=p.get("timestamp"),
                )
                for i, p in enumerate(prior_raw)
                if (p.get("text") or "").strip()
            ]
            turn = TurnInput(
                turn_id=t.get("trace_id") or f"{sid}:{idx}",
                text=text,
                role=t.get("role") or "interviewer",
                source=t.get("source"),
                timestamp=t.get("timestamp"),
            )
            queue.append((turn, prior, session))
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="e.g. 2026-04-20")
    ap.add_argument("--model", default="azure/gpt-4o")
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on total turns (for testing)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random sample N turns per day (stratified by session)")
    ap.add_argument("--min-len", type=int, default=15,
                    help="skip very short turns (raised default to 15 to cut filler)")
    ap.add_argument("--context-window", type=int, default=6)
    ap.add_argument("--roles", nargs="+", default=["interviewer"])
    ap.add_argument("--modes", nargs="+", default=None,
                    help="filter by interview_mode, e.g. --modes copilot code")
    ap.add_argument("--workers", type=int, default=8, help="parallel workers")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    src = ROOT / "data" / "raw" / f"frai_sessions_{args.date}.json"
    if not src.exists():
        print(f"[error] not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "data" / "labeled"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"turns_{args.date}.jsonl"
    if args.force and out.exists():
        out.unlink()

    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    api_base = os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_API_BASE")
    assert api_key, "No API key found"
    print(f"api_base={api_base or '(direct)'}  model={args.model}", file=sys.stderr)

    sessions = iter_sessions(src)
    print(f"sessions={len(sessions)}", file=sys.stderr)

    queue = build_queue(sessions, args.roles, args.min_len, args.context_window, args.modes)
    if args.sample and len(queue) > args.sample:
        import random
        random.seed(42)
        queue = random.sample(queue, args.sample)
        print(f"sampled {len(queue)} from {len(queue) + (len(queue) - args.sample)} candidates", file=sys.stderr)
    if args.limit:
        queue = queue[:args.limit]

    # Resume: skip already-labeled turn_ids (file stays open for appending)
    done_ids: set[str] = set()
    if out.exists() and not args.force:
        for line in out.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("turn_id"):
                    done_ids.add(rec["turn_id"])
            except Exception:
                pass
        if done_ids:
            print(f"resuming: skipping {len(done_ids)} already-labeled turns", file=sys.stderr)

    remaining = [(t, p, s) for t, p, s in queue if t.turn_id not in done_ids]
    print(f"labeling {len(remaining):,} turns → {out.name}", file=sys.stderr)

    labeler = IntentLabeler(model=args.model, api_key=api_key, api_base=api_base)
    written = errors = 0
    write_lock = threading.Lock()
    mode = "a" if done_ids else "w"

    with out.open(mode) as f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool, \
         tqdm(total=len(remaining)) as bar:

        futures = {pool.submit(labeler.label_turn, t, p, s): (t, s)
                   for t, p, s in remaining}

        for fut in as_completed(futures):
            turn_obj, session_obj = futures[fut]
            bar.update(1)
            try:
                result = fut.result()
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                from sorting_hat.labeling.schema import LabeledTurn
                rec = LabeledTurn(
                    turn_id=turn_obj.turn_id,
                    text=turn_obj.text,
                    role=turn_obj.role,
                    source=turn_obj.source,
                    timestamp=turn_obj.timestamp,
                    session_id=session_obj.session_id,
                    interview_mode=session_obj.interview_mode,
                    programming_language=session_obj.programming_language,
                    goal_position=session_obj.goal_position,
                    label=result.label,
                    confidence=result.confidence,
                    reason=result.reason,
                    secondary_label=result.secondary_label,
                    model=args.model,
                    labeled_at=now,
                )
                with write_lock:
                    f.write(rec.model_dump_json() + "\n")
                    f.flush()
                written += 1
            except Exception as e:
                errors += 1
                print(f"\n[skip] {turn_obj.turn_id}: {e}", file=sys.stderr)

    print(f"done. written={written}  errors={errors}  total={written + len(done_ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()

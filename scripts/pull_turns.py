"""Pull FRAI sessions and flatten to per-turn rows.

Output format matches the reference schema (one JSON row per turn):
    data/raw/frai_turns_YYYY-MM-DD.jsonl

Each row: session-level metadata + turn-level fields from `requests` JSON array.
Non-mock sessions only. Preserves all 20+ turn fields (role, transcript,
chat_id, chat_id_index, request_id, trace_id, timestamps, etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
from datetime import date, datetime, timedelta

import pandas as pd
from google.cloud import bigquery

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CRED = ROOT / "bq-breakglass_zhenlu-liuzl.json"

MAX_BYTES = 20 * 1024**3


def iso(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def extract_turns(raw: str | None, session_meta: dict) -> list[dict]:
    """Parse `requests` JSON array, emit one row per valid turn with session meta.

    `requests` is known to store each logical turn twice (two copies with
    microsecond-level timestamp skew, sometimes differing source). We dedup
    adjacent (role, transcript) after sorting by transcript_timestamp.
    """
    if not raw:
        return []
    try:
        arr = json.loads(raw)
    except Exception:
        return []
    collected = []
    for item in arr:
        obj = json.loads(item) if isinstance(item, str) else item
        if not isinstance(obj, dict):
            continue
        if obj.get("source") in {"system", "show_answers", "chat_history"}:
            continue
        text = (obj.get("transcript") or "").strip() or None
        role = obj.get("role")
        if text is None or role not in ("interviewer", "interviewee", "ai"):
            continue
        row = dict(session_meta)
        row.update({
            "role": role,
            "transcript": text,
            "transcript_timestamp": obj.get("timestamp"),
            "transcript_start_timestamp": obj.get("transcript_start_timestamp"),
            "transcript_end_timestamp": obj.get("transcript_end_timestamp"),
            "request_id": obj.get("request_id"),
            "trace_id": obj.get("trace_id"),
            "chat_id": obj.get("chat_id"),
            "chat_id_index": obj.get("chat_id_index"),
            "link_question_id": obj.get("link_question_id"),
            "is_completed": obj.get("is_completed"),
            "source": obj.get("source"),
            "model_type": obj.get("model_type"),
            "concated_transcript": obj.get("concated_transcript"),
            "already_answered": obj.get("already_answered"),
            "already_evaluated": obj.get("already_evaluated"),
            "regenerate": obj.get("regenerate"),
        })
        collected.append(row)

    # Sort by timestamp then dedup adjacent (role, transcript) — kills the 2×
    # system-written copies that land next to each other post-sort.
    collected.sort(key=lambda r: r.get("transcript_timestamp") or "")
    rows = []
    last_key = None
    for r in collected:
        key = (r["role"], r["transcript"])
        if key == last_key:
            continue
        last_key = key
        rows.append(r)
    return rows


def pull_day(client: bigquery.Client, day: date, force: bool = False) -> pathlib.Path | None:
    d = day.isoformat()
    next_d = (day + timedelta(days=1)).isoformat()
    out = OUT_DIR / f"frai_turns_{d}.jsonl"
    if out.exists() and not force:
        print(f"  [skip] {out.name} exists")
        return out

    sql = f"""
      WITH latest AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp DESC) AS rn
        FROM `finalround-dev-552cd.frai.sessions_partitioned_view`
        WHERE timestamp >= TIMESTAMP '{d} 00:00:00'
          AND timestamp <  TIMESTAMP '{next_d} 00:00:00'
          AND interview_mode IS NOT NULL AND interview_mode != 'mock'
      )
      SELECT session_id, user_id, timestamp AS session_snapshot_at,
             interview_mode, programming_language, goal_position, goal_company,
             mock, created_timestamp, activated_timestamp, terminated_timestamp,
             requests
      FROM latest WHERE rn = 1 AND LENGTH(requests) > 10
    """
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES))
    df = job.to_dataframe()

    with out.open("w") as f:
        n_turns = 0
        for _, r in df.iterrows():
            meta = {
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "session_snapshot_at": iso(r["session_snapshot_at"]),
                "interview_mode": r["interview_mode"],
                "programming_language": r["programming_language"] or None,
                "goal_position": r["goal_position"] or None,
                "goal_company": r["goal_company"] or None,
                "mock": bool(r["mock"]) if pd.notna(r["mock"]) else None,
                "created_timestamp": iso(r["created_timestamp"]) if pd.notna(r["created_timestamp"]) else None,
                "activated_timestamp": iso(r["activated_timestamp"]) if pd.notna(r["activated_timestamp"]) else None,
                "terminated_timestamp": iso(r["terminated_timestamp"]) if pd.notna(r["terminated_timestamp"]) else None,
            }
            for turn in extract_turns(r["requests"], meta):
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
                n_turns += 1

    print(f"  [ok]   {out.name}  sessions={len(df)}  turns={n_turns}  "
          f"size={out.stat().st_size/1e6:.1f} MB  bytes={job.total_bytes_processed/1e9:.2f} GB")
    return out


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", type=lambda s: date.fromisoformat(s))
    p.add_argument("end",   type=lambda s: date.fromisoformat(s))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(CRED))
    client = bigquery.Client()

    for d in daterange(args.start, args.end):
        print(f"-> {d.isoformat()}")
        pull_day(client, d, force=args.force)


if __name__ == "__main__":
    main()

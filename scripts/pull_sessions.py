"""Pull one day of FRAI sessions from BigQuery, join with amplitude code_detect events,
clean + structure + save to data/raw/frai_sessions_YYYY-MM-DD.json.

Usage:
    uv run --with google-cloud-bigquery,pandas,pyarrow,db-dtypes \\
        python scripts/pull_sessions.py 2026-04-01 2026-04-20
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd
from google.cloud import bigquery

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CRED = ROOT / "bq-breakglass_zhenlu-liuzl.json"

BAD_SRC = {"system", "show_answers", "chat_history"}
VALID_ROLES = ("interviewer", "interviewee")
MAX_BYTES = 20 * 1024**3


def clean_turns(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        arr = json.loads(raw)
    except Exception:
        return []
    # Collect all raw turns (no dedup yet — requests is known to contain each
    # turn twice with slightly different sources / microsecond timestamps)
    collected: list[dict] = []
    for item in arr:
        obj = json.loads(item) if isinstance(item, str) else item
        if not isinstance(obj, dict):
            continue
        role = obj.get("role")
        text = (obj.get("transcript") or "").strip()
        src = obj.get("source")
        if not text or role not in VALID_ROLES or src in BAD_SRC:
            continue
        collected.append({
            "role": role,
            "text": text,
            "timestamp": obj.get("timestamp"),
            "source": src,
            "trace_id": obj.get("trace_id"),
            "is_completed": obj.get("is_completed"),
        })
    # Sort by timestamp first, then dedup adjacent (role, text) — the two
    # system-written copies have near-identical timestamps so sort brings them together.
    collected.sort(key=lambda x: x["timestamp"] or "")
    out: list[dict] = []
    last_key = None
    for t in collected:
        key = (t["role"], t["text"])
        if key == last_key:
            continue
        last_key = key
        out.append(t)
    return out


def _clean_nan(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, list):
        return [_clean_nan(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean_nan(vv) for k, vv in v.items()}
    return v


def pull_day(client: bigquery.Client, day: date, force: bool = False) -> pathlib.Path | None:
    d = day.isoformat()
    next_d = (day + timedelta(days=1)).isoformat()
    out = OUT_DIR / f"frai_sessions_{d}.json"
    if out.exists() and not force:
        print(f"  [skip] {out.name} exists")
        return out

    # 1) sessions
    sql_sess = f"""
      WITH latest AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp DESC) AS rn
        FROM `finalround-dev-552cd.frai.sessions_partitioned_view`
        WHERE timestamp >= TIMESTAMP '{d} 00:00:00'
          AND timestamp <  TIMESTAMP '{next_d} 00:00:00'
          AND interview_mode IS NOT NULL AND interview_mode != 'mock'
      )
      SELECT session_id, user_id, interview_mode, mock,
             programming_language, goal_position, goal_company,
             created_timestamp, activated_timestamp, terminated_timestamp,
             requests
      FROM latest
      WHERE rn = 1 AND LENGTH(requests) > 10
    """
    job = client.query(sql_sess, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES))
    df = job.to_dataframe()

    # 2) code_detect events
    sql_evt = f"""
      SELECT JSON_VALUE(event_properties, '$.interview_id') AS interview_id,
             event_type, event_time,
             JSON_VALUE(event_properties, '$.problem_type') AS problem_type,
             JSON_VALUE(event_properties, '$.reason')       AS reject_reason
      FROM `finalround-dev-552cd.amplitude.events`
      WHERE _airbyte_extracted_at >= TIMESTAMP '{d} 00:00:00'
        AND _airbyte_extracted_at <  TIMESTAMP_ADD(TIMESTAMP '{next_d} 00:00:00', INTERVAL 2 DAY)
        AND event_time >= TIMESTAMP '{d} 00:00:00'
        AND event_time <  TIMESTAMP '{next_d} 00:00:00'
        AND event_type LIKE 'code_detect_%'
    """
    job2 = client.query(sql_evt, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES))
    evt_df = job2.to_dataframe()

    evt_map = defaultdict(list)
    for _, r in evt_df.iterrows():
        iid = r["interview_id"]
        if not iid:
            continue
        pt = r["problem_type"] if isinstance(r["problem_type"], str) else None
        evt_map[iid].append({
            "event_type": r["event_type"],
            "problem_type": pt,
            "event_time": r["event_time"].isoformat() if pd.notna(r["event_time"]) else None,
            "reject_reason": r["reject_reason"] if isinstance(r["reject_reason"], str) else None,
        })

    # 3) assemble
    sessions: list[dict] = []
    for _, r in df.iterrows():
        turns = clean_turns(r["requests"])
        if not turns:
            continue
        sid = r["session_id"]
        sessions.append({
            "session_id": sid,
            "user_id": r["user_id"],
            "interview_mode": r["interview_mode"],
            "programming_language": r["programming_language"] or None,
            "goal_position": r["goal_position"] or None,
            "goal_company": r["goal_company"] or None,
            "created_timestamp": r["created_timestamp"].isoformat() if pd.notna(r["created_timestamp"]) else None,
            "activated_timestamp": r["activated_timestamp"].isoformat() if pd.notna(r["activated_timestamp"]) else None,
            "terminated_timestamp": r["terminated_timestamp"].isoformat() if pd.notna(r["terminated_timestamp"]) else None,
            "n_turns": len(turns),
            "has_interviewer": any(t["role"] == "interviewer" for t in turns),
            "code_detect_events": evt_map.get(sid, []),
            "turns": turns,
        })

    sessions = _clean_nan(sessions)
    out.write_text(json.dumps(sessions, ensure_ascii=False, indent=2, allow_nan=False))

    bytes_total = job.total_bytes_processed + job2.total_bytes_processed
    print(f"  [ok]   {out.name}  sessions={len(sessions)}  "
          f"events={len(evt_df)}  bytes={bytes_total/1e9:.2f} GB")
    return out


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", type=lambda s: date.fromisoformat(s), help="YYYY-MM-DD")
    p.add_argument("end",   type=lambda s: date.fromisoformat(s), help="YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="re-pull even if file exists")
    args = p.parse_args()

    import os
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(CRED))
    client = bigquery.Client()

    for d in daterange(args.start, args.end):
        print(f"-> {d.isoformat()}")
        pull_day(client, d, force=args.force)


if __name__ == "__main__":
    main()

"""Pull Langfuse clean_traces for a date range, group by session_id,
join with amplitude code_detect events, save per-day JSON.

Output format:
    data/raw/langfuse_sessions_YYYY-MM-DD.json
    [
      {
        "session_id": "...",
        "user_id": "...",
        "first_timestamp": "...",
        "last_timestamp": "...",
        "n_traces": N,
        "output_language": "English",
        "code_detect_events": [...],
        "traces": [
          {"timestamp": "...", "interviewer_query": "..."},
          ...
        ]
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
from google.cloud import bigquery

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CRED = ROOT / "bq-breakglass_zhenlu-liuzl.json"

MAX_BYTES = 20 * 1024**3


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
    out = OUT_DIR / f"langfuse_sessions_{d}.json"
    if out.exists() and not force:
        print(f"  [skip] {out.name} exists")
        return out

    # Langfuse clean_traces — trace-level; group by session later
    sql_traces = f"""
      SELECT timestamp, session_id, user_id, output_language, interviewer_query
      FROM `finalround-dev-552cd.langfuse_v3.clean_traces`
      WHERE timestamp >= TIMESTAMP '{d} 00:00:00'
        AND timestamp <  TIMESTAMP '{next_d} 00:00:00'
        AND session_id IS NOT NULL
        AND interviewer_query IS NOT NULL
        AND interviewer_query != ''
      ORDER BY session_id, timestamp
    """
    job = client.query(sql_traces, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES))
    df = job.to_dataframe()

    # amplitude code_detect for same day
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

    # Group traces by session_id
    sessions_map: dict[str, dict] = {}
    for _, r in df.iterrows():
        sid = r["session_id"]
        if sid not in sessions_map:
            sessions_map[sid] = {
                "session_id": sid,
                "user_id": r["user_id"],
                "output_language": r["output_language"] if isinstance(r["output_language"], str) else None,
                "traces": [],
            }
        sessions_map[sid]["traces"].append({
            "timestamp": r["timestamp"].isoformat() if pd.notna(r["timestamp"]) else None,
            "interviewer_query": r["interviewer_query"],
        })

    sessions = []
    for sid, s in sessions_map.items():
        traces = s["traces"]
        traces.sort(key=lambda x: x["timestamp"] or "")
        s["first_timestamp"] = traces[0]["timestamp"] if traces else None
        s["last_timestamp"]  = traces[-1]["timestamp"] if traces else None
        s["n_traces"] = len(traces)
        s["code_detect_events"] = evt_map.get(sid, [])
        sessions.append(s)

    # Sort by first_timestamp asc
    sessions.sort(key=lambda s: s.get("first_timestamp") or "")
    sessions = _clean_nan(sessions)
    out.write_text(json.dumps(sessions, ensure_ascii=False, indent=2, allow_nan=False))

    bytes_total = job.total_bytes_processed + job2.total_bytes_processed
    print(f"  [ok]   {out.name}  sessions={len(sessions)}  traces={len(df)}  "
          f"amplitude_events={len(evt_df)}  bytes={bytes_total/1e9:.2f} GB")
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

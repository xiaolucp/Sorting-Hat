"""Aggregate the coding_session_transcripts CSV into per-session JSON matching
the viewer's format. One row per session, with a `turns` array sorted by time.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "coding_session_transcripts_20251120_20251130.csv"
DEFAULT_OUT = ROOT / "data" / "raw" / "frai_sessions_coding_nov2025.json"

VALID_ROLES = {"interviewer", "interviewee", "ai"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"reading {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  rows={len(df):,}  sessions={df['session_id'].nunique()}")

    df["transcript"] = df["transcript"].astype(str).str.strip()
    df = df[df["transcript"].str.len() > 0]
    df = df[df["role"].isin(VALID_ROLES)]
    df = df.sort_values(["session_id", "transcript_timestamp"])

    sessions = []
    for sid, grp in df.groupby("session_id", sort=False):
        first = grp.iloc[0]
        turns = []
        last_key = None
        for _, r in grp.iterrows():
            key = (r["role"], r["transcript"])
            if key == last_key:
                continue
            last_key = key
            turns.append({
                "role": r["role"],
                "text": r["transcript"],
                "timestamp": r["transcript_timestamp"],
                "source": None,
                "trace_id": r.get("trace_id") if pd.notna(r.get("trace_id")) else None,
                "is_completed": bool(r["is_completed"]) if pd.notna(r.get("is_completed")) else None,
                "chat_id": r.get("chat_id") if pd.notna(r.get("chat_id")) else None,
                "chat_id_index": int(r["chat_id_index"]) if pd.notna(r.get("chat_id_index")) else None,
                "transcript_start_timestamp": r.get("transcript_start_timestamp")
                    if pd.notna(r.get("transcript_start_timestamp")) else None,
                "transcript_end_timestamp": r.get("transcript_end_timestamp")
                    if pd.notna(r.get("transcript_end_timestamp")) else None,
            })
        if not turns:
            continue
        sessions.append({
            "session_id": sid,
            "user_id": first["user_id"],
            "interview_mode": first["interview_mode"],
            "programming_language": first["programming_language"] if pd.notna(first.get("programming_language")) else None,
            "goal_position": first["goal_position"] if pd.notna(first.get("goal_position")) else None,
            "goal_company": first["goal_company"] if pd.notna(first.get("goal_company")) else None,
            "mock": bool(first["mock"]) if pd.notna(first.get("mock")) else None,
            "created_timestamp": first["transcript_timestamp"],
            "activated_timestamp": None,
            "terminated_timestamp": grp.iloc[-1]["transcript_timestamp"],
            "n_turns": len(turns),
            "has_interviewer": any(t["role"] == "interviewer" for t in turns),
            "code_detect_events": [],
            "turns": turns,
        })

    sessions.sort(key=lambda s: s.get("created_timestamp") or "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sessions, ensure_ascii=False, indent=2, allow_nan=False))

    # stats
    from collections import Counter
    mode = Counter(s["interview_mode"] for s in sessions)
    roles = Counter(t["role"] for s in sessions for t in s["turns"])
    turn_counts = sorted(s["n_turns"] for s in sessions)
    int_sessions = sum(1 for s in sessions if s["has_interviewer"])
    print(f"\nwrote {args.out.name}  {args.out.stat().st_size/1e6:.1f} MB  sessions={len(sessions)}")
    print(f"  interview_mode: {dict(mode)}")
    print(f"  roles: {dict(roles)}")
    print(f"  turns/session p50={turn_counts[len(turn_counts)//2]}  "
          f"mean={sum(turn_counts)/len(turn_counts):.0f}  max={max(turn_counts)}")
    print(f"  sessions with interviewer turns: {int_sessions}/{len(sessions)}")


if __name__ == "__main__":
    main()

"""Streamlit viewer for Langfuse traces — focuses on ASR text in Interviewer Query / Conversation History."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT / ".env.local", ROOT / ".env"):
    if candidate.exists():
        load_dotenv(candidate)
        break

PK = os.getenv("LANGFUSE_PUBLIC_KEY")
SK = os.getenv("LANGFUSE_SECRET_KEY")
BASE = (os.getenv("LANGFUSE_BASE_URL") or "").rstrip("/")


def fetch_traces(from_ts: datetime, to_ts: datetime, *, user_id: str | None,
                 name: str | None, limit: int) -> list[dict]:
    params = {
        "fromTimestamp": from_ts.isoformat().replace("+00:00", "Z"),
        "toTimestamp": to_ts.isoformat().replace("+00:00", "Z"),
        "limit": limit,
    }
    if user_id:
        params["userId"] = user_id
    if name:
        params["name"] = name
    r = requests.get(f"{BASE}/api/public/traces", params=params, auth=(PK, SK), timeout=60)
    r.raise_for_status()
    return r.json().get("data", [])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trace_detail(tid: str) -> dict:
    r = requests.get(f"{BASE}/api/public/traces/{tid}", auth=(PK, SK), timeout=60)
    r.raise_for_status()
    return r.json()


def extract_block(text: str, tag: str) -> str | None:
    m = re.search(
        rf"<{re.escape(tag)}>.*?<content>\s*(.*?)\s*</content>\s*</{re.escape(tag)}>",
        text, re.DOTALL,
    )
    return m.group(1).strip() if m else None


def parse_user_content(trace: dict) -> dict[str, str | None]:
    msgs = (trace.get("input") or {}).get("messages") or []
    user_msg = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    return {
        "interviewer_query": extract_block(user_msg, "Interviewer Query"),
        "conversation_history": extract_block(user_msg, "Conversation History"),
        "candidate": extract_block(user_msg, "Candidate"),
        "job_description": extract_block(user_msg, "Job Description"),
    }


st.set_page_config(page_title="Langfuse ASR Viewer", layout="wide")
st.title("Langfuse ASR Viewer")

if not (PK and SK and BASE):
    st.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in .env.local")
    st.stop()

with st.sidebar:
    st.header("Filters")
    today = datetime.now(timezone.utc).date()
    date_from = st.date_input("From date (UTC)", today - timedelta(days=1))
    hour_from = st.slider("From hour", 0, 23, 0)
    duration = st.slider("Duration (hours)", 1, 24, 1)
    trace_name = st.selectbox(
        "Trace name",
        ["litellm-acompletion", "", "litellm-aembedding"],
        help="litellm-acompletion is the copilot answer call (has ASR).",
    )
    user_id = st.text_input("User ID (optional)")
    limit = st.slider("Limit", 10, 200, 50)
    only_with_query = st.checkbox("Only show traces with <Interviewer Query>", value=True)

from_ts = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour_from)
to_ts = from_ts + timedelta(hours=duration)

st.caption(f"Window (UTC): `{from_ts.isoformat()}` → `{to_ts.isoformat()}`")

if st.button("Load traces", type="primary"):
    with st.spinner("Fetching trace list..."):
        try:
            traces = fetch_traces(
                from_ts, to_ts,
                user_id=user_id or None,
                name=trace_name or None,
                limit=limit,
            )
        except requests.HTTPError as e:
            st.error(f"Langfuse API error: {e.response.status_code} — try a tighter time window.")
            st.stop()
    st.session_state["traces"] = traces

traces = st.session_state.get("traces", [])
st.caption(f"Fetched **{len(traces)}** traces")

if not traces:
    st.info("Click **Load traces** to fetch.")
    st.stop()

# Counters for the filter checkbox — count how many have interviewer query after detail fetch
rows_to_show = []
for t in traces:
    tid = t.get("id")
    ts = (t.get("timestamp") or "")[:19]
    uid = (t.get("userId") or "")
    name = t.get("name") or ""
    rows_to_show.append({"tid": tid, "ts": ts, "uid": uid, "name": name, "_raw": t})

for row in rows_to_show:
    tid = row["tid"]
    header = f"`{row['ts']}` · **{row['name']}** · user `{row['uid'][:24]}` · tid `{tid[:12]}`"
    with st.expander(header):
        try:
            d = fetch_trace_detail(tid)
        except requests.HTTPError as e:
            st.warning(f"detail fetch failed: {e}")
            continue
        parts = parse_user_content(d)
        iq = parts["interviewer_query"]
        ch = parts["conversation_history"]

        if only_with_query and not iq:
            st.caption("_no <Interviewer Query> block — hidden (uncheck sidebar to show all)_")
            continue

        col1, col2 = st.columns([2, 1])
        with col1:
            if iq:
                st.markdown("#### Interviewer Query (current ASR)")
                st.success(iq)
            if ch:
                st.markdown("#### Conversation History")
                st.text(ch[:5000] + ("\n... [truncated]" if len(ch) > 5000 else ""))
            out = (d.get("output") or {}).get("content") if isinstance(d.get("output"), dict) else d.get("output")
            if out:
                st.markdown("#### AI Output")
                st.text(str(out)[:3000])
        with col2:
            st.caption("Metadata")
            st.json({
                "id": d.get("id"),
                "timestamp": d.get("timestamp"),
                "userId": d.get("userId"),
                "sessionId": d.get("sessionId"),
                "latency": d.get("latency"),
                "totalCost": d.get("totalCost"),
                "tags": d.get("tags"),
                "environment": d.get("environment"),
            }, expanded=False)
            if parts["candidate"]:
                st.caption("Candidate")
                st.text(parts["candidate"][:600])
            if parts["job_description"]:
                st.caption("Job Description")
                st.text(parts["job_description"][:600])

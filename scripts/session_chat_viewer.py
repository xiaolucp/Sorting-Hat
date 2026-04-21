"""Chat-bubble viewer for FRAI session data. Renders interviewer on the left,
interviewee on the right (iMessage-style), with relative time offsets.

Expects data files at `data/raw/frai_sessions_*.json` produced by the session
pull pipeline. Optionally overlays intent labels from
`data/labeled/turns_YYYY-MM-DD.jsonl` with per-label background colours.
This is a display-only service — no mutation.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
LABELED_DIR = ROOT / "data" / "labeled"

ROLE_SIDE = {"interviewer": "left", "interviewee": "right"}
ROLE_COLOR = {
    "interviewer": "#ececec",
    "interviewee": "#0b93f6",
}
ROLE_TEXT_COLOR = {
    "interviewer": "#111",
    "interviewee": "#fff",
}

# Per-label bubble background colours (interviewer turns only)
# (bubble_bg, bubble_fg, badge_bg)  — keep high contrast so colours are obvious
LABEL_COLOR: dict[str, tuple[str, str, str]] = {
    "coding":           ("#c6f6d5", "#1a5c35", "#16a34a"),
    "system_design":    ("#bfdbfe", "#1e3a8a", "#2563eb"),
    "project_qa":       ("#e9d5ff", "#4c1d95", "#7c3aed"),
    "chat":             ("#fef08a", "#713f12", "#ca8a04"),
    "no_answer_needed": ("#e2e8f0", "#64748b", "#94a3b8"),
}
LABEL_DISPLAY = {
    "coding":           "💻 coding",
    "system_design":    "🏗️ system design",
    "project_qa":       "📁 project Q&A",
    "chat":             "💬 chat",
    "no_answer_needed": "🔇 no answer",
}
ROLE_AVATAR = {
    "interviewer": "🧑‍💼",
    "interviewee": "🧑‍💻",
}
SOURCE_ICON = {
    "microphone": "🎤",
    "video-display": "📺",
    "tavus": "🤖",
    "system": "⚙️",
    "show_answers": "📝",
    "chat_history": "🗂️",
    None: "·",
    "": "·",
}

EVENT_ICON = {
    "code_detect_triggered": "🔍",
    "code_detect_screenshot_uploaded": "📸",
    "code_detect_auto_detected": "🤖",
    "code_detect_generation_completed": "✨",
    "code_detect_problem_accepted": "✅",
    "code_detect_problem_rejected": "❌",
    "code_detect_panel_toggled": "🪟",
    "code_detect_error": "⚠️",
}

CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 8px; background: #fff; }
.event-row { display: flex; justify-content: center; margin: 12px 0; }
.event {
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid transparent;
}
.event.code            { background: #e6f7ed; color: #0f7a3f; border-color: #b8e2c8; }
.event.system_design   { background: #e6f0ff; color: #1456b0; border-color: #b8d0ee; }
.event.accepted        { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
.event.rejected        { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.event.neutral         { background: #f3f4f6; color: #374151; border-color: #e5e7eb; }
.event .tag { padding: 1px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
.event .tag.code          { background: #0f7a3f; color: #fff; }
.event .tag.system_design { background: #1456b0; color: #fff; }
.chat-row { display: flex; margin: 8px 0; width: 100%; align-items: flex-end; gap: 8px; }
.chat-row.left  { justify-content: flex-start; }
.chat-row.right { justify-content: flex-end; }
.avatar {
  width: 36px; height: 36px;
  border-radius: 50%;
  background: #f2f2f5;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.35rem;
  flex-shrink: 0;
  line-height: 1;
}
.bubble-wrap { display: flex; flex-direction: column; max-width: 70%; }
.bubble {
  padding: 8px 12px;
  border-radius: 16px;
  line-height: 1.45;
  font-size: 0.95rem;
  box-shadow: 0 1px 1.5px rgba(0,0,0,0.08);
  white-space: pre-wrap;
  word-wrap: break-word;
}
.chat-row.left  .bubble { border-bottom-left-radius: 4px; }
.chat-row.right .bubble { border-bottom-right-radius: 4px; }
.meta { font-size: 0.72rem; color: #888; margin-bottom: 2px; display: flex; gap: 4px; align-items: center; }
.chat-row.left  .meta { justify-content: flex-start; padding-left: 4px; }
.chat-row.right .meta { justify-content: flex-end;   padding-right: 4px; }
.src-icon { font-size: 0.9rem; }
.label-badge {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 700;
  color: #fff;
  margin-left: 4px;
  vertical-align: middle;
}
.confidence { font-size: 0.68rem; color: #aaa; margin-left: 3px; }
.reason-tip { cursor: help; }
</style>
"""


def parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def fmt_offset(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def render_bubble(
    role: str,
    text: str,
    offset: str,
    source: str | None,
    label_info: dict | None = None,
) -> str:
    side = ROLE_SIDE.get(role, "left")
    bubble_bg = ROLE_COLOR.get(role, "#ddd")
    bubble_fg = ROLE_TEXT_COLOR.get(role, "#111")
    avatar = ROLE_AVATAR.get(role, "💬")
    src_icon = SOURCE_ICON.get(source, "·")
    src_title = html.escape(source or "unknown")
    esc_text = html.escape(text).replace("\n", "<br>")

    # Override bubble colour if we have a label for this turn
    badge_html = ""
    if label_info:
        lname = label_info.get("label", "")
        colors = LABEL_COLOR.get(lname)
        if colors:
            bubble_bg, bubble_fg, badge_bg = colors
        conf = label_info.get("confidence", 0)
        reason = html.escape((label_info.get("reason") or "")[:300])
        display = html.escape(LABEL_DISPLAY.get(lname, lname))
        badge_html = (
            f'<span class="label-badge reason-tip" style="background:{badge_bg}" title="{reason}">'
            f'{display}</span>'
            f'<span class="confidence">{conf:.0%}</span>'
        )

    avatar_html = f'<div class="avatar" title="{html.escape(role)}">{avatar}</div>'
    meta_html = (
        f'<div class="meta">'
        f'<span>{html.escape(offset)}</span>'
        f'<span class="src-icon" title="{src_title}">{src_icon}</span>'
        f'{badge_html}'
        f'</div>'
    )
    bubble_html = (
        f'<div class="bubble {side}" style="background:{bubble_bg};color:{bubble_fg};">'
        f'{esc_text}</div>'
    )
    wrap_html = f'<div class="bubble-wrap">{meta_html}{bubble_html}</div>'

    if side == "left":
        return f'<div class="chat-row left">{avatar_html}{wrap_html}</div>'
    return f'<div class="chat-row right">{wrap_html}{avatar_html}</div>'


def render_event(event_type: str, problem_type: str | None, offset: str,
                 reject_reason: str | None) -> str:
    icon = EVENT_ICON.get(event_type, "🎯")
    short = event_type.replace("code_detect_", "")
    classes = ["event"]
    if event_type == "code_detect_problem_accepted":
        classes.append("accepted")
    elif event_type == "code_detect_problem_rejected":
        classes.append("rejected")
    else:
        classes.append("neutral")

    pt_badge = ""
    if problem_type == "CODE":
        pt_badge = '<span class="tag code">CODE</span>'
    elif problem_type == "SYSTEM_DESIGN":
        pt_badge = '<span class="tag system_design">SYSTEM_DESIGN</span>'

    reason_attr = ""
    if reject_reason:
        reason_attr = f' title="{html.escape(reject_reason[:400])}"'

    return (
        f'<div class="event-row">'
        f'<span class="{" ".join(classes)}"{reason_attr}>'
        f'<span>{icon}</span>'
        f'<span>[{html.escape(offset)}]</span>'
        f'<span>{html.escape(short)}</span>'
        f'{pt_badge}'
        f'</span></div>'
    )


def _pt_set(events: list[dict]) -> set[str]:
    out: set[str] = set()
    for e in events:
        pt = e.get("problem_type")
        if isinstance(pt, str) and pt:
            out.add(pt)
        else:
            out.add("rejected")
    return out


def session_label(s: dict, labels: dict[str, dict] | None = None) -> str:
    mode = s.get("interview_mode", "?")
    nt = s.get("n_turns", 0)
    goal = (s.get("goal_position") or "")[:30]
    pts = sorted(_pt_set(s.get("code_detect_events", [])))
    pt_badge = f"  [{'/'.join(pts)}]" if pts else ""
    label_summary = ""
    if labels:
        from collections import Counter
        c = Counter(
            labels[t.get("trace_id", "")]["label"]
            for t in s.get("turns", [])
            if t.get("trace_id") in labels
        )
        if c:
            short = {"coding": "cod", "system_design": "sd", "project_qa": "pqa",
                     "chat": "chat", "no_answer_needed": "nan"}
            label_summary = "  🏷 " + " ".join(f"{short.get(k,k)}:{v}" for k, v in c.most_common())
    return f"{s['session_id'][:10]} · {mode} · {nt}t · {goal}{pt_badge}{label_summary}"


def load_labels(date_str: str) -> dict[str, dict]:
    """Load labeled turns for a given date. Returns trace_id → label record."""
    path = LABELED_DIR / f"turns_{date_str}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            tid = rec.get("turn_id")
            if tid:
                out[tid] = rec
        except Exception:
            pass
    return out


st.set_page_config(page_title="FRAI Session Chat Viewer", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("FRAI Session Chat Viewer")

files = sorted(DATA_DIR.glob("frai_sessions_*.json"))
if not files:
    st.error(f"No data files in {DATA_DIR}")
    st.stop()

with st.sidebar:
    st.header("Data")
    chosen = st.selectbox("File", [f.name for f in files], index=len(files) - 1)
    sessions = json.loads((DATA_DIR / chosen).read_text())
    st.caption(f"{len(sessions)} sessions loaded")

    # Derive date string from filename, e.g. "frai_sessions_2026-04-20.json" → "2026-04-20"
    date_str = chosen.removeprefix("frai_sessions_").removesuffix(".json")
    labels: dict[str, dict] = load_labels(date_str)
    if labels:
        st.caption(f"✅ {len(labels):,} labeled turns loaded")
    else:
        st.caption("⚠️ No labels — run `label_sessions.py --date {date_str}`")

    st.header("Filters")
    only_interviewer = st.checkbox("Has interviewer turn", value=True)
    only_code_detect = st.checkbox("Has code_detect event", value=False)
    problem_type = st.selectbox(
        "problem_type", ["(any)", "CODE", "SYSTEM_DESIGN", "none/rejected"]
    )
    modes_present = sorted({s.get("interview_mode") for s in sessions if s.get("interview_mode")})
    mode_filter = st.multiselect("interview_mode", modes_present)

    label_filter: list[str] = []
    only_labeled = False
    if labels:
        only_labeled = st.checkbox("Has labeled turns", value=False)
        all_labels = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]
        label_filter = st.multiselect("intent label (any turn)", all_labels)

def _session_trace_ids(s: dict) -> set[str]:
    out = set()
    for t in s.get("turns", []):
        tid = t.get("trace_id")
        if tid:
            out.add(tid)
    return out


filtered = []
for s in sessions:
    if only_interviewer and not s.get("has_interviewer"):
        continue
    if only_code_detect and not s.get("code_detect_events"):
        continue
    if mode_filter and s.get("interview_mode") not in mode_filter:
        continue
    if problem_type != "(any)":
        pts = {e.get("problem_type") for e in s.get("code_detect_events", [])}
        if problem_type == "none/rejected":
            if any(pts):
                continue
        elif problem_type not in pts:
            continue
    tids = _session_trace_ids(s)
    has_any_label = bool(labels) and any(tid in labels for tid in tids)
    if only_labeled and not has_any_label:
        continue
    if label_filter and labels:
        session_labels = {labels[tid]["label"] for tid in tids if tid in labels}
        if not any(lbl in session_labels for lbl in label_filter):
            continue
    filtered.append(s)

with st.sidebar:
    st.caption(f"**{len(filtered)}** sessions match")
    if not filtered:
        st.stop()

    search = st.text_input("Search (session_id / goal / mode)", placeholder="e.g. 5VCWKrp or kubernetes")
    if search:
        s_low = search.lower()
        filtered = [
            s for s in filtered
            if s_low in s["session_id"].lower()
            or s_low in (s.get("goal_position") or "").lower()
            or s_low in (s.get("interview_mode") or "").lower()
        ]
        st.caption(f"after search: **{len(filtered)}**")
        if not filtered:
            st.stop()

    sort_by = st.selectbox(
        "Sort by",
        ["time asc", "time desc", "turns desc", "turns asc", "session_id"],
    )

    def _session_ts(s: dict) -> str:
        return (s.get("created_timestamp") or s.get("activated_timestamp") or "") or ""

    if sort_by == "time asc":
        filtered = sorted(filtered, key=_session_ts)
    elif sort_by == "time desc":
        filtered = sorted(filtered, key=_session_ts, reverse=True)
    elif sort_by == "turns desc":
        filtered = sorted(filtered, key=lambda s: -s.get("n_turns", 0))
    elif sort_by == "turns asc":
        filtered = sorted(filtered, key=lambda s: s.get("n_turns", 0))
    else:
        filtered = sorted(filtered, key=lambda s: s["session_id"])

    sel = st.radio("Session", range(len(filtered)), format_func=lambda i: session_label(filtered[i], labels))

if not filtered:
    st.info("No sessions match current filter.")
    st.stop()

sess = filtered[sel]

st.subheader(f"Session `{sess['session_id']}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mode", sess.get("interview_mode") or "—")
c2.metric("Turns", sess.get("n_turns"))
c3.metric("Has interviewer", "✓" if sess.get("has_interviewer") else "✗")
c4.metric("code_detect", len(sess.get("code_detect_events", [])))

mc = st.columns(3)
mc[0].write(f"**Language:** {sess.get('programming_language') or '—'}")
mc[1].write(f"**Goal:** {sess.get('goal_position') or '—'}")
mc[2].write(f"**Company:** {sess.get('goal_company') or '—'}")

st.markdown("### Dialogue")
label_legend = ""
if labels:
    label_legend = (
        " · intent colours: "
        "🟢 coding · 🔵 system design · 🟣 project Q&A · 🟡 chat · ⬜ no answer"
    )
st.caption(
    "🧑‍💼 interviewer (left) · 🧑‍💻 interviewee (right) · "
    "🎤 mic · 📺 video-display · 🤖 tavus · ⚙️ system · "
    "📸 screenshot · ✅ CODE/SD accepted · ❌ rejected · "
    "[M:SS] = offset from first utterance"
    + label_legend
)

turns = sess.get("turns", [])
t0 = None
for t in turns:
    dt = parse_iso(t.get("timestamp"))
    if dt:
        t0 = dt
        break

# Merge turns + code_detect events into one timeline sorted by timestamp
timeline: list[tuple[datetime | None, str, dict]] = []
for t in turns:
    timeline.append((parse_iso(t.get("timestamp")), "turn", t))
for e in sess.get("code_detect_events", []):
    timeline.append((parse_iso(e.get("event_time")), "event", e))

# Stable sort: items with no timestamp go to the end
FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)
timeline.sort(key=lambda x: x[0] if x[0] is not None else FAR_FUTURE)

blocks: list[str] = []
for dt, kind, item in timeline:
    off = fmt_offset((dt - t0).total_seconds()) if (dt and t0) else "—:—"
    if kind == "turn":
        trace_id = item.get("trace_id")
        label_info = labels.get(trace_id) if (trace_id and labels) else None
        blocks.append(render_bubble(item.get("role") or "unknown",
                                    item.get("text") or "",
                                    off,
                                    item.get("source"),
                                    label_info))
    else:
        blocks.append(render_event(item.get("event_type", ""),
                                   item.get("problem_type"),
                                   off,
                                   item.get("reject_reason")))

body = CSS + "".join(blocks)
# Height heuristic: ~70px per block
height = max(300, min(4000, 70 * len(timeline) + 100))
components.html(body, height=height, scrolling=True)

"""Side-by-side eval comparison viewer.

Shows the 100-turn 04-07 eval set with:
  • current turn text + prior context
  • labels from gpt-4o (gold), gpt-5, base Qwen3-4B, Qwen3-VL-Emb-2B classifier
  • per-row agreement highlights

Run:
    uv run --no-sync streamlit run scripts/eval_comparison_viewer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

LABELS = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]
LABEL_DISPLAY = {
    "coding":           "💻 coding",
    "system_design":    "🏗️ system_design",
    "project_qa":       "📁 project_qa",
    "chat":             "💬 chat",
    "no_answer_needed": "🔇 no_answer",
    None:               "—",
    "None":             "—",
}
LABEL_BG = {
    "coding":           "#c6f6d5",
    "system_design":    "#bfdbfe",
    "project_qa":       "#e9d5ff",
    "chat":             "#fef08a",
    "no_answer_needed": "#e2e8f0",
    None:               "#f3f4f6",
    "None":             "#f3f4f6",
}
LABEL_FG = {
    "coding":           "#1a5c35",
    "system_design":    "#1e3a8a",
    "project_qa":       "#4c1d95",
    "chat":             "#713f12",
    "no_answer_needed": "#64748b",
    None:               "#888",
    "None":             "#888",
}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_all(_cache_key: tuple):
    # _cache_key is a tuple of (path, mtime) pairs; Streamlit re-fetches whenever any file changes.
    eval_path = ROOT / "data/training/eval_04-07_partial.jsonl"
    embed_path = ROOT / "runs/embed_v2_preds_04-07.json"
    gpt5_path = ROOT / "runs/gpt5_vs_gpt4o_04-07.json"
    base_path = ROOT / "runs/base_qwen_aligned_04-07.json"
    v5_path = ROOT / "runs/eval_v5_final_04-07.json"

    eval_rows = [json.loads(l) for l in eval_path.open() if l.strip()]
    embed_preds = {r["turn_id"]: r for r in json.load(embed_path.open())}
    gpt5_preds = {r["turn_id"]: r for r in json.load(gpt5_path.open())}
    base_data = json.load(base_path.open())
    base_preds = dict(zip(base_data["turn_ids"], base_data["preds"]))
    v5_data = json.load(v5_path.open())
    # eval_intent.py stores {"summary":..., "predictions":[{turn_id, pred, pred_conf, pred_reason, ...}, ...]}
    v5_preds = {p["turn_id"]: p for p in v5_data.get("predictions", [])}

    merged = []
    for r in eval_rows:
        tid = r["current_turn"]["trace_id"]
        em = embed_preds.get(tid, {})
        g5 = gpt5_preds.get(tid, {})
        v5 = v5_preds.get(tid, {})
        merged.append({
            "turn_id": tid,
            "text": r["current_turn"].get("text") or "",
            "prior": r.get("prior_turns") or [],
            "session": r.get("session") or {},
            "gold_4o": r.get("label"),
            "gold_4o_reason": r.get("reason"),
            "gpt5": g5.get("gpt5"),
            "gpt5_reason": g5.get("gpt5_reason"),
            "base_qwen": base_preds.get(tid),
            "embed": em.get("pred"),
            "embed_conf": em.get("pred_conf"),
            "embed_probs": em.get("all_probs"),
            "v5": v5.get("pred"),
            "v5_conf": v5.get("pred_conf"),
            "v5_reason": v5.get("pred_reason"),
        })
    return merged


def label_pill(label, fontsize="0.72rem"):
    if not label:
        label = None
    bg = LABEL_BG.get(label, "#eee")
    fg = LABEL_FG.get(label, "#666")
    txt = LABEL_DISPLAY.get(label, str(label))
    return f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:12px;font-size:{fontsize};font-weight:600;display:inline-block;'>{txt}</span>"


def main():
    st.set_page_config(page_title="Intent eval comparison (04-07)", layout="wide")
    st.title("Intent eval comparison — 2026-04-07 (n=100)")
    _paths = [
        ROOT / "data/training/eval_04-07_partial.jsonl",
        ROOT / "runs/embed_v2_preds_04-07.json",
        ROOT / "runs/gpt5_vs_gpt4o_04-07.json",
        ROOT / "runs/base_qwen_aligned_04-07.json",
        ROOT / "runs/eval_v5_final_04-07.json",
    ]
    rows = load_all(tuple((str(p), _mtime(p)) for p in _paths))

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        label_filter = st.multiselect(
            "gpt-4o label (gold)",
            options=LABELS,
            default=LABELS,
        )
        disagree_only = st.checkbox("Only show disagreements (any model ≠ gpt-4o)", value=False)
        focus = st.selectbox("Focus comparison", ["all models", "gpt-4o vs v5 LoRA", "gpt-4o vs gpt-5", "gpt-4o vs embedding", "gpt-4o vs base_qwen"])

        st.header("Summary")
        n_total = len(rows)
        accs = {
            "v5 LoRA ↔ gpt-4o":   sum(1 for r in rows if r["v5"] == r["gold_4o"]) / n_total,
            "gpt-5 ↔ gpt-4o":     sum(1 for r in rows if r["gpt5"] == r["gold_4o"]) / n_total,
            "embedding ↔ gpt-4o": sum(1 for r in rows if r["embed"] == r["gold_4o"]) / n_total,
            "base_qwen ↔ gpt-4o": sum(1 for r in rows if r["base_qwen"] == r["gold_4o"]) / n_total,
        }
        for k, v in accs.items():
            st.metric(k, f"{v:.1%}")

    # Filter
    filtered = [r for r in rows if r["gold_4o"] in label_filter]
    if disagree_only:
        def has_disagreement(r):
            return any(r[k] != r["gold_4o"] for k in ("gpt5", "base_qwen", "embed", "v5"))
        filtered = [r for r in filtered if has_disagreement(r)]
    st.caption(f"showing {len(filtered)} / {len(rows)} turns")

    # Render each row
    for r in filtered:
        with st.container(border=True):
            # Header row with all labels (5 columns: gold + 4 predictors)
            cols = st.columns([1, 1, 1, 1, 1])
            cols[0].markdown(f"**gpt-4o (gold)**<br>{label_pill(r['gold_4o'])}", unsafe_allow_html=True)
            v5_conf = r.get("v5_conf")
            v5_conf_txt = f" <span style='font-size:.7rem;color:#888;'>({v5_conf:.2f})</span>" if v5_conf is not None else ""
            cols[1].markdown(
                f"**v5 LoRA (Qwen3-4B)**<br>{label_pill(r['v5'])}{v5_conf_txt}",
                unsafe_allow_html=True,
            )
            cols[2].markdown(f"**gpt-5**<br>{label_pill(r['gpt5'])}", unsafe_allow_html=True)
            emb_conf = r.get("embed_conf")
            emb_conf_txt = f" <span style='font-size:.7rem;color:#888;'>({emb_conf:.2f})</span>" if emb_conf is not None else ""
            cols[3].markdown(
                f"**Qwen3-VL-Emb-2B**<br>{label_pill(r['embed'])}{emb_conf_txt}",
                unsafe_allow_html=True,
            )
            cols[4].markdown(f"**base Qwen3-4B**<br>{label_pill(r['base_qwen'])}", unsafe_allow_html=True)

            # Current turn (bold)
            st.markdown(
                f"<div style='background:#fffbe8;padding:10px 14px;border-left:3px solid #f59e0b;margin:10px 0 6px 0;border-radius:4px;'>"
                f"<b>Current turn:</b> {r['text']}</div>",
                unsafe_allow_html=True,
            )

            # Reasons + probs expander
            with st.expander("details: reasons / probs / session / prior"):
                sess = r["session"]
                st.caption(
                    f"mode={sess.get('interview_mode') or '-'}  "
                    f"lang={sess.get('programming_language') or '-'}  "
                    f"role={sess.get('goal_position') or '-'}  "
                    f"company={sess.get('goal_company') or '-'}"
                )
                st.markdown(f"**gpt-4o reason:** {r['gold_4o_reason'] or '—'}")
                st.markdown(f"**v5 LoRA reason:** {r.get('v5_reason') or '—'}")
                st.markdown(f"**gpt-5 reason:**  {r['gpt5_reason'] or '—'}")
                if r.get("embed_probs"):
                    st.markdown(
                        "**embedding all-class probabilities:** "
                        + "  ".join(
                            f"{LABEL_DISPLAY.get(l)} <code>{r['embed_probs'].get(l, 0):.2f}</code>"
                            for l in LABELS
                        ),
                        unsafe_allow_html=True,
                    )
                if r["prior"]:
                    st.markdown("**prior turns (last 6):**")
                    for p in r["prior"][-6:]:
                        who = p.get("role") or "?"
                        txt = (p.get("text") or "").strip()
                        bg = "#e8f4f8" if who == "interviewer" else "#f0fff4"
                        st.markdown(
                            f"<div style='background:{bg};padding:6px 10px;margin:3px 0;"
                            f"border-radius:6px;font-size:.88rem;'>"
                            f"<b>{who}:</b> {txt}</div>",
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()

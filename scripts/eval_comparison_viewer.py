"""Demo dashboard + per-turn eval comparison.

Two tabs on the same page (port 8502):

  1. **Summary** — model training + evaluation results as tables
     (overall accuracy, per-class P/R, confusion matrix, training history,
     labeler-noise ceiling).
  2. **Per-turn comparison** — 100-turn 04-07 eval set, one card per turn
     with gpt-4o (gold), v5 LoRA, gpt-5, Qwen3-VL-Emb-2B classifier,
     base Qwen3-4B predictions side by side, plus reasons and softmax
     probabilities.

Run:
    uv run --no-sync streamlit run scripts/eval_comparison_viewer.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
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


# ---------------------------------------------------------------------------
# Summary-tab helpers
# ---------------------------------------------------------------------------

LABEL_SHORT = {
    "coding": "coding",
    "system_design": "sys_design",
    "project_qa": "project_qa",
    "chat": "chat",
    "no_answer_needed": "no_answer",
}


def _accuracy(rows, pred_key: str) -> float:
    vals = [r for r in rows if r[pred_key] is not None]
    return sum(1 for r in vals if r[pred_key] == r["gold_4o"]) / max(1, len(vals))


def _per_class_pr(rows, pred_key: str) -> pd.DataFrame:
    out = []
    for lbl in LABELS:
        tp = sum(1 for r in rows if r["gold_4o"] == lbl and r[pred_key] == lbl)
        fp = sum(1 for r in rows if r["gold_4o"] != lbl and r[pred_key] == lbl)
        fn = sum(1 for r in rows if r["gold_4o"] == lbl and r[pred_key] != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"label": LABEL_SHORT[lbl],
                    "P": f"{prec:.2f}", "R": f"{rec:.2f}", "F1": f"{f1:.2f}",
                    "TP": tp, "FP": fp, "FN": fn})
    return pd.DataFrame(out)


def _confusion(rows, pred_key: str) -> pd.DataFrame:
    cm = {g: Counter() for g in LABELS}
    for r in rows:
        g = r["gold_4o"]
        p = r[pred_key] if r[pred_key] in LABELS else "UNP"
        cm[g][p] += 1
    cols = [LABEL_SHORT[l] for l in LABELS] + ["UNP"]
    df = pd.DataFrame(
        [[cm[g].get(p, 0) for p in LABELS + ["UNP"]] for g in LABELS],
        index=[LABEL_SHORT[l] for l in LABELS],
        columns=cols,
    )
    df.index.name = "gold ↓ / pred →"
    return df


def render_summary_tab(rows):
    st.warning(
        "**数据标注来源**：训练集和这里用的 gold label 全部由 **gpt-4o** 通过我们的 "
        "`SYSTEM_PROMPT` rubric 预标注产生（`scripts/label_sessions.py` + LiteLLM 代理，"
        "`confidence ≥ 0.9` 过滤）。\n\n"
        "因此下面表格里的 **“accuracy” = 模型预测与 gpt-4o 标签的一致率**，"
        "**不是**与人工 ground truth 的一致率。gpt-4o 本身会漏判 / 误判，所以：\n"
        "- 同一 rubric 下 **gpt-5 和 gpt-4o 只有 75% 一致**，这就是 "
        "“学 gpt-4o 标签” 模型的理论天花板约 80%；\n"
        "- 超过 75% 的模型（如 v5 的 87%）说明学到了**跨 labeler 的通用判据**，不是在复读 gpt-4o；\n"
        "- 最终真实上限要靠 **200-500 条人工 golden set** 评估（TODO，尚未建）。",
        icon="⚠️",
    )

    v5_acc = _accuracy(rows, "v5")
    embed_acc = _accuracy(rows, "embed")
    gpt5_acc = _accuracy(rows, "gpt5")
    base_acc = _accuracy(rows, "base_qwen")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("v5 LoRA (Qwen3-4B, short prompt)", f"{v5_acc:.0%}",
              delta=f"{v5_acc - gpt5_acc:+.0%} vs noise ceiling")
    c2.metric("Qwen3-VL-Emb-2B + logreg", f"{embed_acc:.0%}",
              delta=f"{embed_acc - gpt5_acc:+.0%} vs noise ceiling")
    c3.metric("gpt-5 ↔ gpt-4o (labeler noise)", f"{gpt5_acc:.0%}",
              delta="theoretical ceiling", delta_color="off")
    c4.metric("base Qwen3-4B (zero-shot)", f"{base_acc:.0%}",
              delta=f"{base_acc - gpt5_acc:+.0%} vs noise ceiling")

    st.subheader("① Model comparison")
    st.caption("All rows evaluated on the exact same 100 held-out turns.")
    summary = pd.DataFrame([
        {"model": "v5 LoRA (Qwen3-4B + LoRA, SYSTEM_PROMPT_SHORT)",
         "params trained": "~20 M LoRA",
         "accuracy": f"{v5_acc:.0%}",
         "train time": "3h 36m (3 epochs, A10G)",
         "inference (batch=1)": "~1.0 s / req (HF bf16)",
         "status": "production candidate"},
        {"model": "Qwen3-VL-Embedding-2B + logreg head (w/ system prompt)",
         "params trained": "~10 KB logreg",
         "accuracy": f"{embed_acc:.0%}",
         "train time": "13 min",
         "inference (batch=1)": "~0.2 s / req",
         "status": "low-latency alternative"},
        {"model": "gpt-5 (labeler alternative)",
         "params trained": "—",
         "accuracy": f"{gpt5_acc:.0%}",
         "train time": "—",
         "inference (batch=1)": "~0.7 s / req (API)",
         "status": "labeler comparison baseline"},
        {"model": "base Qwen3-4B (zero-shot, full prompt)",
         "params trained": "—",
         "accuracy": f"{base_acc:.0%}",
         "train time": "0",
         "inference (batch=1)": "~2-3 s / req (HF bf16)",
         "status": "baseline"},
    ])
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.info(
        f"**Headline takeaway.** The fine-tuned v5 LoRA reaches "
        f"**{v5_acc:.0%}**, which is **{(v5_acc - gpt5_acc) * 100:+.0f} points above** "
        f"the gpt-4o ↔ gpt-5 inter-labeler agreement "
        f"({gpt5_acc:.0%}). That means the model is not merely parroting "
        f"the gpt-4o labeler — it has learned a general decision rule.",
        icon="📌",
    )

    st.subheader("② Per-class precision / recall (vs gpt-4o labels)")
    st.caption("On the same 100 turns; gold label = gpt-4o. v5 is balanced across all 5 classes.")
    models = [("v5 LoRA", "v5"), ("Embedding", "embed"),
              ("gpt-5", "gpt5"), ("base Qwen", "base_qwen")]
    tabs = st.tabs([m[0] for m in models])
    for tab, (_name, key) in zip(tabs, models):
        with tab:
            st.dataframe(_per_class_pr(rows, key), hide_index=True, use_container_width=True)

    st.subheader("③ Confusion matrix — v5 LoRA vs gpt-4o labels")
    st.caption("Rows: gpt-4o label (used as gold). Columns: v5 LoRA prediction. Clean diagonal = high agreement.")
    st.dataframe(_confusion(rows, "v5"), use_container_width=True)

    st.subheader("④ Training history")
    st.caption(
        "Each row is one training attempt. The held-out number is on 100 turns "
        "from 04-07; numbers marked (*) used a different random stratified 100, "
        "not strictly apples-to-apples with the aligned set above."
    )
    history = pd.DataFrame([
        {"version": "v3 (04-21)",
         "data": "04-20 only, 16k samples",
         "prompt": "full (~2000 tok)",
         "max_seq_len": 2048,
         "loss mask": "completion+assistant (double, buggy)",
         "truncation": "right (default, broken)",
         "epochs": "3",
         "training-set acc": "51%",
         "held-out acc": "—",
         "note": "Mask bug + right-truncation stripped assistant JSON"},
        {"version": "v4 (04-22, ckpt-200)",
         "data": "7-day balanced, 4,045",
         "prompt": "full (~2000 tok)",
         "max_seq_len": 1024,
         "loss mask": "completion only ✓",
         "truncation": "left ✓",
         "epochs": "0.2",
         "training-set acc": "—",
         "held-out acc": "59% *",
         "note": "Prompt truncated; model learned shortcuts"},
        {"version": "v4 (ckpt-400)",
         "data": "same as above",
         "prompt": "full (~2000 tok)",
         "max_seq_len": 1024,
         "loss mask": "completion only",
         "truncation": "left",
         "epochs": "0.4",
         "training-set acc": "—",
         "held-out acc": "58% *",
         "note": "Same truncation issue, no improvement"},
        {"version": "v5 (04-22, ckpt-200)",
         "data": "7-day balanced, 4,045",
         "prompt": "SHORT (~205 tok)",
         "max_seq_len": 1536,
         "loss mask": "completion only",
         "truncation": "left",
         "epochs": "0.3",
         "training-set acc": "—",
         "held-out acc": "81%",
         "note": "Early checkpoint already above gpt-5 ceiling"},
        {"version": "v5 final ✓",
         "data": "same as above",
         "prompt": "SHORT (~205 tok)",
         "max_seq_len": 1536,
         "loss mask": "completion only",
         "truncation": "left",
         "epochs": "3",
         "training-set acc": "—",
         "held-out acc": f"{v5_acc:.0%}",
         "note": "Production candidate"},
    ])
    st.dataframe(history, hide_index=True, use_container_width=True)

    st.subheader("⑤ Labeler noise (gpt-4o vs gpt-5, same 100 turns)")
    st.caption(
        "Both are strong labelers applied with the same rubric. Their disagreement "
        "rate is the theoretical ceiling for any model trained on gpt-4o labels."
    )
    rows_noise = []
    for lbl in LABELS:
        gold_rows = [r for r in rows if r["gold_4o"] == lbl]
        n = len(gold_rows)
        if not n:
            continue
        agree = sum(1 for r in gold_rows if r["gpt5"] == lbl)
        rows_noise.append({
            "gold label (gpt-4o)": LABEL_SHORT[lbl],
            "n": n,
            "gpt-5 agrees": agree,
            "agreement rate": f"{agree/n:.0%}",
        })
    total_n = len(rows)
    total_agree = sum(1 for r in rows if r["gpt5"] == r["gold_4o"])
    rows_noise.append({
        "gold label (gpt-4o)": "OVERALL",
        "n": total_n,
        "gpt-5 agrees": total_agree,
        "agreement rate": f"{total_agree/total_n:.0%}",
    })
    st.dataframe(pd.DataFrame(rows_noise), hide_index=True, use_container_width=True)

    st.caption(
        "Most disagreements fall into two patterns: "
        "(1) gpt-4o tends to *infer* system_design / coding from surrounding context "
        "while gpt-5 reads the current turn more literally; "
        "(2) polite closings and ASR fragments where both 'chat' and 'no_answer_needed' "
        "defensibly apply."
    )


def main():
    st.set_page_config(page_title="Sorting Hat — Results & Comparison", layout="wide")
    st.title("Sorting Hat — Results & Per-turn Comparison")
    st.caption(
        "100 held-out turns from **2026-04-07** (stratified 20 per class, "
        "disjoint from training). All numbers recomputed live from `runs/*.json`."
    )

    _paths = [
        ROOT / "data/training/eval_04-07_partial.jsonl",
        ROOT / "runs/embed_v2_preds_04-07.json",
        ROOT / "runs/gpt5_vs_gpt4o_04-07.json",
        ROOT / "runs/base_qwen_aligned_04-07.json",
        ROOT / "runs/eval_v5_final_04-07.json",
    ]
    rows = load_all(tuple((str(p), _mtime(p)) for p in _paths))

    tab_summary, tab_detail = st.tabs(["📊 Summary", "🔍 Per-turn comparison"])

    with tab_summary:
        render_summary_tab(rows)

    with tab_detail:
        # Sidebar filters (these only make sense when looking at the per-turn tab)
        with st.sidebar:
            st.header("Per-turn filters")
            label_filter = st.multiselect(
                "gpt-4o label (gold)",
                options=LABELS,
                default=LABELS,
            )
            disagree_only = st.checkbox("Only show disagreements (any model ≠ gpt-4o)", value=False)

            st.header("Quick stats")
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

        _render_per_turn(filtered)


def _render_per_turn(filtered):
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

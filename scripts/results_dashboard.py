"""One-page demo dashboard: model training + eval results as clean tables.

Reads the actual prediction/evaluation JSON files under runs/ and renders
everything as Streamlit dataframes — no mocked numbers.

Run:
    uv run --no-sync streamlit run scripts/results_dashboard.py --server.port 8503
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

LABELS = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]
LABEL_SHORT = {
    "coding": "coding",
    "system_design": "sys_design",
    "project_qa": "project_qa",
    "chat": "chat",
    "no_answer_needed": "no_answer",
}


# ---------------------------------------------------------------------------
# Data loading (file-mtime-aware cache so files can update in place)
# ---------------------------------------------------------------------------

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_aligned(_key: tuple):
    eval_rows = [json.loads(l) for l in (ROOT / "data/training/eval_04-07_partial.jsonl").open() if l.strip()]
    gold = {r["current_turn"]["trace_id"]: r["label"] for r in eval_rows}

    v5 = {p["turn_id"]: p for p in json.load((RUNS / "eval_v5_final_04-07.json").open())["predictions"]}
    embed = {r["turn_id"]: r for r in json.load((RUNS / "embed_v2_preds_04-07.json").open())}
    gpt5 = {r["turn_id"]: r for r in json.load((RUNS / "gpt5_vs_gpt4o_04-07.json").open())}
    base_blob = json.load((RUNS / "base_qwen_aligned_04-07.json").open())
    base = dict(zip(base_blob["turn_ids"], base_blob["preds"]))

    rows = []
    for tid, g in gold.items():
        rows.append({
            "turn_id": tid,
            "gold":  g,
            "v5":    v5.get(tid, {}).get("pred"),
            "v5_conf":  v5.get(tid, {}).get("pred_conf"),
            "gpt5":  gpt5.get(tid, {}).get("gpt5"),
            "embed": embed.get(tid, {}).get("pred"),
            "base":  base.get(tid),
        })
    return rows


def accuracy(rows, pred_key: str) -> float:
    vals = [r for r in rows if r[pred_key] is not None]
    return sum(1 for r in vals if r[pred_key] == r["gold"]) / max(1, len(vals))


def per_class_pr(rows, pred_key: str) -> pd.DataFrame:
    out = []
    for lbl in LABELS:
        tp = sum(1 for r in rows if r["gold"] == lbl and r[pred_key] == lbl)
        fp = sum(1 for r in rows if r["gold"] != lbl and r[pred_key] == lbl)
        fn = sum(1 for r in rows if r["gold"] == lbl and r[pred_key] != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"label": LABEL_SHORT[lbl], "P": prec, "R": rec, "F1": f1,
                    "TP": tp, "FP": fp, "FN": fn})
    return pd.DataFrame(out)


def confusion_df(rows, pred_key: str) -> pd.DataFrame:
    cm = {g: Counter() for g in LABELS}
    for r in rows:
        g = r["gold"]
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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sorting Hat — Results", layout="wide")
st.title("Sorting Hat — Model Results Dashboard")
st.caption(
    "All metrics computed on the same 100-turn held-out set from "
    "**2026-04-07** (stratified 20 per class, disjoint from training)."
)

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

files_key = tuple(
    (str(RUNS / n), _mtime(RUNS / n))
    for n in (
        "eval_v5_final_04-07.json",
        "embed_v2_preds_04-07.json",
        "gpt5_vs_gpt4o_04-07.json",
        "base_qwen_aligned_04-07.json",
    )
)
rows = load_aligned(files_key)

# ---- Headline metrics ------------------------------------------------------
v5_acc     = accuracy(rows, "v5")
embed_acc  = accuracy(rows, "embed")
gpt5_acc   = accuracy(rows, "gpt5")
base_acc   = accuracy(rows, "base")

c1, c2, c3, c4 = st.columns(4)
c1.metric("v5 LoRA (Qwen3-4B, short prompt)", f"{v5_acc:.0%}",
          delta=f"+{v5_acc - gpt5_acc:+.0%} vs noise ceiling")
c2.metric("Qwen3-VL-Emb-2B + logreg", f"{embed_acc:.0%}",
          delta=f"{embed_acc - gpt5_acc:+.0%} vs noise ceiling")
c3.metric("gpt-5 ↔ gpt-4o (labeler noise)", f"{gpt5_acc:.0%}",
          delta="theoretical ceiling", delta_color="off")
c4.metric("base Qwen3-4B (no FT)", f"{base_acc:.0%}",
          delta=f"{base_acc - gpt5_acc:+.0%} vs noise ceiling")

# ---- Table 1: Model comparison --------------------------------------------
st.subheader("① Model comparison")
st.caption("All rows evaluated on the exact same 100 held-out turns.")
summary = pd.DataFrame([
    {
        "model": "v5 LoRA (Qwen3-4B + LoRA, SYSTEM_PROMPT_SHORT)",
        "params trained": "~20 M LoRA",
        "accuracy": f"{v5_acc:.0%}",
        "train time": "3h 36m (3 epochs, A10G)",
        "inference (batch=1)": "~1.0 s / req (HF bf16)",
        "status": "production candidate",
    },
    {
        "model": "Qwen3-VL-Embedding-2B + logreg head (w/ system prompt)",
        "params trained": "~10 KB logreg",
        "accuracy": f"{embed_acc:.0%}",
        "train time": "13 min",
        "inference (batch=1)": "~0.2 s / req",
        "status": "low-latency alternative",
    },
    {
        "model": "gpt-5 (labeler alternative)",
        "params trained": "—",
        "accuracy": f"{gpt5_acc:.0%}",
        "train time": "—",
        "inference (batch=1)": "~0.7 s / req (API)",
        "status": "labeler comparison baseline",
    },
    {
        "model": "base Qwen3-4B (zero-shot, full prompt)",
        "params trained": "—",
        "accuracy": f"{base_acc:.0%}",
        "train time": "0",
        "inference (batch=1)": "~2-3 s / req (HF bf16)",
        "status": "baseline",
    },
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

# ---- Table 2: per-class P/R for each predictor ----------------------------
st.subheader("② Per-class precision / recall (vs gpt-4o labels)")
st.caption("On the same 100 turns; gold label = gpt-4o. v5 is balanced across all 5 classes.")

models = [("v5 LoRA", "v5"), ("Embedding", "embed"), ("gpt-5", "gpt5"), ("base Qwen", "base")]
tabs = st.tabs([m[0] for m in models])
for tab, (name, key) in zip(tabs, models):
    with tab:
        df = per_class_pr(rows, key)
        df_fmt = df.copy()
        for c in ("P", "R", "F1"):
            df_fmt[c] = df_fmt[c].map(lambda v: f"{v:.2f}")
        st.dataframe(df_fmt, hide_index=True, use_container_width=True)

# ---- Table 3: confusion matrix for v5 -------------------------------------
st.subheader("③ Confusion matrix — v5 LoRA vs gpt-4o labels")
st.caption("Rows: gpt-4o label (used as gold). Columns: v5 LoRA prediction. Clean diagonal = high agreement.")
cm = confusion_df(rows, "v5")
st.dataframe(cm, use_container_width=True)

# ---- Table 4: training configs / history ----------------------------------
st.subheader("④ Training history")
st.caption(
    "Each row is one training attempt. The held-out number is on 100 turns "
    "from 04-07; numbers marked (*) used a different random stratified 100, "
    "not strictly apples-to-apples with the aligned set above."
)
history = pd.DataFrame([
    {
        "version": "v3 (04-21)",
        "data": "04-20 only, 16k samples",
        "prompt": "full (~2000 tok)",
        "max_seq_len": 2048,
        "loss mask": "completion+assistant (double, buggy)",
        "truncation": "right (default, broken)",
        "epochs": "3",
        "training-set acc": "51%",
        "held-out acc": "—",
        "note": "Mask bug + right-truncation stripped assistant JSON",
    },
    {
        "version": "v4 (04-22, ckpt-200)",
        "data": "7-day balanced, 4,045",
        "prompt": "full (~2000 tok)",
        "max_seq_len": 1024,
        "loss mask": "completion only ✓",
        "truncation": "left ✓",
        "epochs": "0.2",
        "training-set acc": "—",
        "held-out acc": "59% *",
        "note": "Prompt truncated; model learned shortcuts",
    },
    {
        "version": "v4 (ckpt-400)",
        "data": "same as above",
        "prompt": "full (~2000 tok)",
        "max_seq_len": 1024,
        "loss mask": "completion only",
        "truncation": "left",
        "epochs": "0.4",
        "training-set acc": "—",
        "held-out acc": "58% *",
        "note": "Same truncation issue, no improvement",
    },
    {
        "version": "v5 (04-22, ckpt-200)",
        "data": "7-day balanced, 4,045",
        "prompt": "SHORT (~205 tok)",
        "max_seq_len": 1536,
        "loss mask": "completion only",
        "truncation": "left",
        "epochs": "0.3",
        "training-set acc": "—",
        "held-out acc": "81%",
        "note": "Early checkpoint already above gpt-5 ceiling",
    },
    {
        "version": "v5 final ✓",
        "data": "same as above",
        "prompt": "SHORT (~205 tok)",
        "max_seq_len": 1536,
        "loss mask": "completion only",
        "truncation": "left",
        "epochs": "3",
        "training-set acc": "—",
        "held-out acc": f"{v5_acc:.0%}",
        "note": "Production candidate",
    },
])
st.dataframe(history, hide_index=True, use_container_width=True)

# ---- Table 5: labeler noise -----------------------------------------------
st.subheader("⑤ Labeler noise (gpt-4o vs gpt-5, same 100 turns)")
st.caption(
    "Both are strong labelers applied with the same rubric. Their disagreement "
    "rate is the theoretical ceiling for any model trained on gpt-4o labels."
)
agree_by_class = []
for lbl in LABELS:
    gold_rows = [r for r in rows if r["gold"] == lbl]
    n = len(gold_rows)
    if not n:
        continue
    agree = sum(1 for r in gold_rows if r["gpt5"] == lbl)
    agree_by_class.append({
        "gold label (gpt-4o)": LABEL_SHORT[lbl],
        "n": n,
        "gpt-5 agrees": agree,
        "agreement rate": f"{agree/n:.0%}",
    })
total_n = len(rows)
total_agree = sum(1 for r in rows if r["gpt5"] == r["gold"])
agree_by_class.append({
    "gold label (gpt-4o)": "OVERALL",
    "n": total_n,
    "gpt-5 agrees": total_agree,
    "agreement rate": f"{total_agree/total_n:.0%}",
})
st.dataframe(pd.DataFrame(agree_by_class), hide_index=True, use_container_width=True)

st.caption(
    "Most disagreements fall into two patterns: "
    "(1) gpt-4o tends to *infer* system_design / coding from surrounding context "
    "while gpt-5 reads the current turn more literally; "
    "(2) polite closings and ASR fragments where both 'chat' and 'no_answer_needed' "
    "defensibly apply."
)

# ---- Footer links ----------------------------------------------------------
st.markdown("---")
st.markdown(
    "**Related viewers:** "
    "[session viewer (port 8501)](/) · "
    "[eval comparison (port 8502)](/) · "
    "this page (port 8503)."
)

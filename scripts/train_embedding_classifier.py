"""Frozen Qwen3-VL-Embedding-2B + logistic regression classifier head.

Pipeline:
  1. For each sample, build a text string: last N prior turns + current turn
  2. Encode with the embedding model → 2048-d vector
  3. Train a logistic regression (sklearn) to map 2048d → 5 labels
  4. Eval on an held-out jsonl

The embedding model is frozen — only the linear head trains. Fast to iterate.

Usage:
    uv run --no-sync python scripts/train_embedding_classifier.py \\
      --train data/training/train_all7_balanced_clean.jsonl \\
      --eval  data/training/eval_04-07_partial.jsonl \\
      --out   models/qwen3vl-emb-clf-v1.npz \\
      --prior-turns 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

LABELS = ["coding", "system_design", "project_qa", "chat", "no_answer_needed"]
L2I = {l: i for i, l in enumerate(LABELS)}

# Minimal ~200-token rubric: core label semantics only, no examples, no decision tree.
# The logreg head does the actual classification — the rubric just steers embedding geometry.
SHORT_PROMPT = (
    "Task: classify one interview turn by intent. Labels:\n"
    "- coding: asks the candidate to WRITE / implement / trace code.\n"
    "- system_design: asks to DESIGN an end-to-end multi-component system.\n"
    "- project_qa: drills into the candidate's past projects / experience / background.\n"
    "- chat: concept Q&A, definition, small talk, logistics, clarification.\n"
    "- no_answer_needed: filler, interviewer self-talk, mic check, ASR fragment, no coherent ask."
)


def row_to_text(row: dict, prior_turns: int, system_prompt: str | None = None) -> str:
    """Flatten a training-format row into a single text string for embedding."""
    s = row.get("session") or {}
    ct = row["current_turn"]
    prior = (row.get("prior_turns") or [])[-prior_turns:] if prior_turns > 0 else []
    prior_str = "\n".join(
        f"[{p.get('role','?')}] {(p.get('text') or '').strip()}" for p in prior if (p.get("text") or "").strip()
    )
    header = (
        f"mode={s.get('interview_mode') or '-'} "
        f"lang={s.get('programming_language') or '-'} "
        f"role={s.get('goal_position') or '-'}"
    )
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    parts.append(header)
    if prior_str:
        parts.append("prior:\n" + prior_str)
    parts.append(f"[{ct.get('role') or 'interviewer'}] {(ct.get('text') or '').strip()}")
    return "\n".join(parts)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def encode_all(model, texts: list[str], batch_size: int) -> np.ndarray:
    t0 = time.time()
    out = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  encoded {len(texts)} texts in {time.time()-t0:.0f}s  shape={out.shape}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--eval", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prior-turns", type=int, default=4,
                    help="how many recent prior turns to concatenate into the input text")
    ap.add_argument("--include-system-prompt", action="store_true",
                    help="prepend the labeling SYSTEM_PROMPT (~2000 tokens) to each input")
    ap.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()

    print(f"[1/5] loading model {args.model} ...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda")
    # enable bf16 for speed if supported
    try:
        model = model.bfloat16()
    except Exception:
        pass

    print(f"[2/5] loading data ...", file=sys.stderr)
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)
    print(f"    train: {len(train_rows)}  label dist: {Counter(r['label'] for r in train_rows)}", file=sys.stderr)
    print(f"    eval : {len(eval_rows)}  label dist: {Counter(r['label'] for r in eval_rows)}", file=sys.stderr)

    sys_prompt = None
    if args.include_system_prompt:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        from sorting_hat.labeling.prompts import SYSTEM_PROMPT
        sys_prompt = SYSTEM_PROMPT
    train_texts = [row_to_text(r, args.prior_turns, sys_prompt) for r in train_rows]
    eval_texts = [row_to_text(r, args.prior_turns, sys_prompt) for r in eval_rows]
    train_y = np.array([L2I[r["label"]] for r in train_rows])
    eval_y = np.array([L2I[r["label"]] for r in eval_rows])

    print(f"[3/5] encoding train ({len(train_texts)}) ...", file=sys.stderr)
    X_train = encode_all(model, train_texts, args.batch_size)
    print(f"[3/5] encoding eval  ({len(eval_texts)}) ...", file=sys.stderr)
    X_eval = encode_all(model, eval_texts, args.batch_size)

    print(f"[4/5] fitting {args.head} head ...", file=sys.stderr)
    if args.head == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, train_y)
        pred = clf.predict(X_eval)
    else:  # mlp
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=200, early_stopping=True,
                            random_state=0, verbose=False)
        clf.fit(X_train, train_y)
        pred = clf.predict(X_eval)

    # report
    print(f"[5/5] eval ...", file=sys.stderr)
    acc = (pred == eval_y).mean()
    print(f"\n================ RESULT ================")
    print(f"accuracy on {args.eval.name}: {acc:.3f}   (n={len(eval_y)})")

    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for g, p in zip(eval_y, pred):
        cm[g, p] += 1
    print("\nper-class:")
    for i, lbl in enumerate(LABELS):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        prec = tp / (tp+fp) if tp+fp else 0
        rec  = tp / (tp+fn) if tp+fn else 0
        print(f"  {lbl:18s}  P={prec:.3f}  R={rec:.3f}  (tp={tp} fp={fp} fn={fn})")

    print("\nconfusion (gold rows → pred cols):")
    header = "           " + " ".join(f"{l[:8]:>10s}" for l in LABELS)
    print(header)
    for i, lbl in enumerate(LABELS):
        row = f"{lbl[:10]:<10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(LABELS)))
        print(row)

    # save head
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.head == "logreg":
        np.savez(str(args.out),
                 W=clf.coef_, b=clf.intercept_, labels=np.array(LABELS),
                 model_name=args.model, prior_turns=args.prior_turns,
                 include_system_prompt=bool(args.include_system_prompt))
    else:
        # sklearn MLP has multiple weight matrices; just pickle
        import pickle
        with open(str(args.out) + ".pkl", "wb") as f:
            pickle.dump({"clf": clf, "labels": LABELS, "model_name": args.model,
                         "prior_turns": args.prior_turns}, f)
    print(f"\nwrote head → {args.out}")


if __name__ == "__main__":
    main()

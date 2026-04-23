"""Load a saved Qwen3-VL-Embedding + logreg classifier and write predictions on an eval jsonl."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_embedding_classifier import LABELS, row_to_text  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=pathlib.Path, required=True, help="saved .npz from train_embedding_classifier")
    ap.add_argument("--eval", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--include-system-prompt", action="store_true",
                    help="override: force include system_prompt (use if .npz predates this flag)")
    args = ap.parse_args()

    blob = np.load(str(args.head), allow_pickle=True)
    W = blob["W"]            # (C, D)
    b = blob["b"]            # (C,)
    labels = list(blob["labels"])
    model_name = str(blob["model_name"])
    prior_turns = int(blob["prior_turns"])
    include_sys = (
        bool(blob["include_system_prompt"]) if "include_system_prompt" in blob.files
        else args.include_system_prompt
    )

    print(f"head: {args.head}", file=sys.stderr)
    print(f"  model={model_name}  prior_turns={prior_turns}  include_sys={include_sys}", file=sys.stderr)

    sys_prompt = None
    if include_sys:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
        from sorting_hat.labeling.prompts import SYSTEM_PROMPT
        sys_prompt = SYSTEM_PROMPT

    rows = [json.loads(l) for l in args.eval.open() if l.strip()]
    texts = [row_to_text(r, prior_turns, sys_prompt) for r in rows]
    print(f"encoding {len(texts)} samples ...", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
    try:
        model = model.bfloat16()
    except Exception:
        pass
    X = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                     show_progress_bar=True, normalize_embeddings=True)

    logits = X @ W.T + b            # (N, C)
    # softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    pred_idx = probs.argmax(axis=1)

    out_records = []
    for r, pi, pr in zip(rows, pred_idx, probs):
        out_records.append({
            "turn_id": r["current_turn"].get("trace_id"),
            "text": r["current_turn"].get("text"),
            "gold": r["label"],
            "gold_reason": r.get("reason"),
            "pred": labels[pi],
            "pred_conf": float(pr[pi]),
            "all_probs": {labels[j]: float(pr[j]) for j in range(len(labels))},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    acc = sum(1 for r in out_records if r["pred"] == r["gold"]) / len(out_records)
    print(f"accuracy={acc:.3f}  wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

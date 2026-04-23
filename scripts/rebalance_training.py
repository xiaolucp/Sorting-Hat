"""Rebalance a training JSONL so each label ratio is in [min_ratio, max_ratio].

Strategy (no duplication — only downsampling):
  1. Let m = count of the minority class.
  2. Set N_max = floor(m / min_ratio); this is the largest total where minority
     still meets the min_ratio floor.
  3. Cap every class at floor(N_max * max_ratio) (the upper ratio).
  4. If the sum of capped counts still exceeds N_max, scale the non-minority
     classes proportionally down so total = N_max.

Usage:
    uv run python scripts/rebalance_training.py \\
        --in  data/training/combined_v2.jsonl \\
        --out data/training/combined_v3_balanced.jsonl \\
        --min-ratio 0.10 --max-ratio 0.30
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict


def plan(counts: dict[str, int], min_ratio: float, max_ratio: float) -> dict[str, int]:
    m_lbl = min(counts, key=counts.get)
    m = counts[m_lbl]
    n_max = int(m / min_ratio)
    cap = int(n_max * max_ratio)

    # First pass: cap each class at (max_ratio * n_max) or available
    targets = {lbl: min(c, cap) for lbl, c in counts.items()}
    targets[m_lbl] = m  # always take all of minority class

    # If the rough sum exceeds n_max, scale non-minority classes proportionally
    total = sum(targets.values())
    if total > n_max:
        other_sum = total - m
        budget = n_max - m
        scale = budget / other_sum
        for lbl in targets:
            if lbl == m_lbl:
                continue
            targets[lbl] = max(int(n_max * min_ratio), int(targets[lbl] * scale))

    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--min-ratio", type=float, default=0.10)
    ap.add_argument("--max-ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (0 < args.min_ratio < args.max_ratio < 1):
        print("need 0 < min_ratio < max_ratio < 1", file=sys.stderr)
        sys.exit(1)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for line in args.in_path.open():
        if not line.strip():
            continue
        d = json.loads(line)
        by_label[d["label"]].append(d)

    counts = {l: len(v) for l, v in by_label.items()}
    print("input distribution:", file=sys.stderr)
    total_in = sum(counts.values())
    for l, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {l:18s} {c:6d}  ({c/total_in:.1%})", file=sys.stderr)
    print(f"  total: {total_in}", file=sys.stderr)

    targets = plan(counts, args.min_ratio, args.max_ratio)
    total_out = sum(targets.values())
    print(f"\ntarget distribution (min_ratio={args.min_ratio:.2f} max_ratio={args.max_ratio:.2f}):",
          file=sys.stderr)
    for l, t in sorted(targets.items(), key=lambda x: -x[1]):
        src = counts[l]
        print(f"  {l:18s} {t:6d}  ({t/total_out:.1%})  ← from {src}", file=sys.stderr)
    print(f"  total: {total_out}", file=sys.stderr)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w") as fout:
        chosen = []
        for lbl, rows in by_label.items():
            rng.shuffle(rows)
            chosen.extend(rows[: targets[lbl]])
        rng.shuffle(chosen)
        for row in chosen:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nwrote {written} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

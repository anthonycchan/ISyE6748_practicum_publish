#!/usr/bin/env python3
import re
import sys
import os
from typing import List, Tuple, Optional, Dict, Any

def read_text() -> str:
    path = sys.argv[1] if len(sys.argv) > 1 else "data.txt"
    if not os.path.exists(path):
        print(f"Error: '{path}' not found. Usage: python parse_cp_ocsvm.py [path/to/data.txt]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Match a block that begins with either:
#  (a) "Factor: <f> Bottleneck: <b> Rank: <...>"
#  (b) "Rank: <...>"
block_re = re.compile(
    r"^(?:"
    r"Factor:\s*(\d+)\s+Bottleneck:\s*(\d+)\s+Rank:\s*([^\n\r]+)"  # g1=factor, g2=bottleneck, g3=rank_str
    r"|"
    r"Rank:\s*([^\n\r]+)"                                          # g4=rank_str (no factor/bottleneck)
    r")\s*(.*?)\s*(?=^(?:Factor:|Rank:)|\Z)",
    re.S | re.M
)

# Accept "Accuracy: 0.6425 AUC 0.624075" OR "Accuracy 0.6425 AUC 0.624075"
acc_auc_line_re = re.compile(
    r"Accuracy:?\s*([0-9.]+)\s*AUC:?\s*([0-9.]+)",
    re.S
)

# Fallback: "[...] FINAL AUC=0.624 | Acc@best=0.613" OR "accuracy=0.625"
final_auc_bestacc_re = re.compile(
    r"FINAL\s+AUC\s*=\s*([0-9.]+).*?(?:Acc@best\s*=\s*([0-9.]+)|Accuracy\s*=\s*([0-9.]+)|accuracy\s*=\s*([0-9.]+))",
    re.S
)

def parse_rank_nums(rank_str: str) -> Tuple[int, ...]:
    """Extract integers from '10' or '64 64 16' etc., return as a tuple."""
    return tuple(int(x) for x in re.findall(r"\d+", rank_str))

def rank_complexity(nums: Tuple[int, ...]) -> int:
    """A simple 'smaller-is-better' complexity to break ties: sum of dimensions."""
    return sum(nums) if nums else 10**9

def parse_blocks(text: str) -> List[Dict[str, Any]]:
    rows = []
    for m in block_re.finditer(text):
        if m.group(3) is not None:  # Factor/Bottleneck/Rank case
            factor = int(m.group(1))
            bottleneck = int(m.group(2))
            rank_str_raw = m.group(3)
        else:                        # Rank-only case
            factor = None
            bottleneck = None
            rank_str_raw = m.group(4)

        block = m.group(5)
        rank_str = " ".join(rank_str_raw.split())
        nums = parse_rank_nums(rank_str)

        # Prefer an explicit "Accuracy ... AUC ..." pair if present
        acc_auc = acc_auc_line_re.search(block)
        if acc_auc:
            acc = float(acc_auc.group(1))
            auc = float(acc_auc.group(2))
        else:
            # Fall back to "FINAL AUC=... | Acc@best=..." or "... accuracy=..."
            fa = final_auc_bestacc_re.search(block)
            if not fa:
                continue
            auc = float(fa.group(1))
            acc_str = fa.group(2) or fa.group(3) or fa.group(4)
            if acc_str is None:
                continue
            acc = float(acc_str)

        rows.append({
            "rank_str": rank_str,          # e.g., "65" or "64 64 16"
            "rank_nums": nums,             # e.g., (65,) or (64,64,16)
            "factor": factor,              # None if not present
            "bottleneck": bottleneck,      # None if not present
            "accuracy": acc,
            "auc": auc
        })
    return rows

def describe_row(r: Dict[str, Any]) -> str:
    desc = f"rank {r['rank_str']}"
    if r["factor"] is not None and r["bottleneck"] is not None:
        desc += f" (factor={r['factor']}, bottleneck={r['bottleneck']})"
    return desc

def main():
    text = read_text()
    rows = parse_blocks(text)
    if not rows:
        print("No (rank, accuracy, auc) triples found.")
        sys.exit(2)

    # Output CSV with factor/bottleneck columns (empty if absent)
    print("rank,factor,bottleneck,accuracy,auc")
    for r in rows:
        f = "" if r["factor"] is None else str(r["factor"])
        b = "" if r["bottleneck"] is None else str(r["bottleneck"])
        print(f"{r['rank_str']},{f},{b},{r['accuracy']},{r['auc']}")

    # Best-by-accuracy (tie-break: higher AUC, then smaller complexity)
    best_acc = max(
        rows,
        key=lambda r: (r["accuracy"], r["auc"], -rank_complexity(r["rank_nums"]))
    )

    # Best-by-AUC (tie-break: higher accuracy, then smaller complexity)
    best_auc = max(
        rows,
        key=lambda r: (r["auc"], r["accuracy"], -rank_complexity(r["rank_nums"]))
    )

    print()
    print(
        f"Best accuracy -> {describe_row(best_acc)} "
        f"(accuracy={best_acc['accuracy']:.4f}, auc={best_acc['auc']:.6f})"
    )
    print(
        f"Best AUC      -> {describe_row(best_auc)} "
        f"(auc={best_auc['auc']:.6f}, accuracy={best_auc['accuracy']:.4f})"
    )

if __name__ == "__main__":
    main()

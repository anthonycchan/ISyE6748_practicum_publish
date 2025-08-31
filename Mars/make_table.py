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

block_re = re.compile(
    r"^Rank:\s*([^\n\r]+)\s*(.*?)\s*(?=^Rank:|\Z)",
    re.S | re.M
)

acc_auc_line_re = re.compile(
    r"Accuracy:\s*([0-9.]+)\s*AUC\s*([0-9.]+)",
    re.S
)

final_auc_bestacc_re = re.compile(
    r"FINAL\s+AUC\s*=\s*([0-9.]+).*?(?:Acc@best\s*=\s*([0-9.]+)|Accuracy\s*=\s*([0-9.]+))",
    re.S
)

def parse_rank_nums(rank_str: str) -> Tuple[int, ...]:
    """Extract integers from '10' or '64 64 16' etc., return as a tuple."""
    nums = [int(x) for x in re.findall(r"\d+", rank_str)]
    return tuple(nums)

def rank_complexity(nums: Tuple[int, ...]) -> int:
    """A simple 'smaller-is-better' complexity: sum of dimensions."""
    return sum(nums) if nums else 10**9

def parse_blocks(text: str) -> List[Dict[str, Any]]:
    rows = []
    for m in block_re.finditer(text):
        rank_str_raw = m.group(1)
        block = m.group(2)
        rank_str = " ".join(rank_str_raw.split())  # normalize spacing
        nums = parse_rank_nums(rank_str)

        acc_auc = acc_auc_line_re.search(block)
        if acc_auc:
            acc = float(acc_auc.group(1))
            auc = float(acc_auc.group(2))
        else:
            fa = final_auc_bestacc_re.search(block)
            if not fa:
                # No metrics found; skip this block
                continue
            auc = float(fa.group(1))
            # pick whichever is present: Acc@best or Accuracy
            acc_str = fa.group(2) or fa.group(3)
            if acc_str is None:
                continue
            acc = float(acc_str)

        rows.append({
            "rank_str": rank_str,      # e.g., "65" or "64 64 16"
            "rank_nums": nums,         # e.g., (65,) or (64,64,16)
            "accuracy": acc,
            "auc": auc
        })
    return rows

def main():
    text = read_text()
    rows = parse_blocks(text)
    if not rows:
        print("No (rank, accuracy, auc) triples found.")
        sys.exit(2)

    # Output a simple CSV
    print("rank,accuracy,auc")
    for r in rows:
        print(f"{r['rank_str']},{r['accuracy']},{r['auc']}")

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
        f"Best accuracy -> rank {best_acc['rank_str']} "
        f"(accuracy={best_acc['accuracy']:.4f}, auc={best_acc['auc']:.6f})"
    )
    print(
        f"Best AUC      -> rank {best_auc['rank_str']} "
        f"(auc={best_auc['auc']:.6f}, accuracy={best_auc['accuracy']:.4f})"
    )

if __name__ == "__main__":
    main()

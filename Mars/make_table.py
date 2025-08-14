#!/usr/bin/env python3
"""
Build a wide table: one row per rank (r1,r2,r3) and one Accuracy column per Bottleneck.
- Preserves the order of ranks as they FIRST appear in the input text.
- Robust to extra lines between "Rank: ..." and "Accuracy: ..."(e.g., warnings).
- Writes a CSV with columns: rank, Bottleneck 8, Bottleneck 16, ...

Usage:
  python make_table.py -i data.txt -o table.csv
  # optionally fix the bottleneck column order (comma-separated)
  python make_table.py -i data.txt -o table.csv --bn-order 8,16,24,32,64
"""

import argparse
import re
import sys
from collections import OrderedDict
import csv

RANK_BLOCK_RE = re.compile(
    r"Rank:\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*Factor:\s*\d+\s*Bottleneck:\s*(\d+)",
    re.MULTILINE,
)
ACC_RE = re.compile(r"Accuracy:\s*([0-9.]+)")

def parse_runs(text: str):
    """Yield tuples (rank_tuple, bottleneck_int, accuracy_float) in encounter order."""
    matches = list(RANK_BLOCK_RE.finditer(text))
    for i, m in enumerate(matches):
        r1, r2, r3, b = m.groups()
        r = (int(r1), int(r2), int(r3))
        b = int(b)
        # search for the first Accuracy: inside the block until the next Rank:
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        acc_m = ACC_RE.search(chunk)
        if not acc_m:
            print(f"[warn] No Accuracy found for rank {r} bottleneck {b}", file=sys.stderr)
            continue
        acc = float(acc_m.group(1))
        yield r, b, acc

def build_table(rows, bn_order=None):
    """
    rows: iterable of (rank_tuple, bottleneck_int, accuracy_float)
    bn_order: list[int] to force bottleneck column order, else sorted ascending
    returns: (header: list[str], table_rows: list[list])
    """
    rank_map = OrderedDict()     # rank_str -> {bn: acc}
    seen_bns = OrderedDict()     # preserve first-seen order (fallback)

    for r, b, acc in rows:
        rank_str = f"({r[0]}, {r[1]}, {r[2]})"
        if rank_str not in rank_map:
            rank_map[rank_str] = {}
        # remember bottleneck and set accuracy
        seen_bns.setdefault(b, None)
        rank_map[rank_str][b] = acc

    # decide bottleneck column order
    if bn_order:
        bns = [int(x) for x in bn_order]
    else:
        # default to numeric ascending of all seen bns
        bns = sorted(seen_bns.keys())

    header = ["rank"] + [f"Bottleneck {b}" for b in bns]
    table_rows = []
    for rank_str, bn_to_acc in rank_map.items():
        row = [rank_str] + [bn_to_acc.get(b, "") for b in bns]
        table_rows.append(row)

    return header, table_rows

def main():
    ap = argparse.ArgumentParser(description="Make wide rank-by-bottleneck accuracy table.")
    ap.add_argument("-i", "--input", required=True, help="Path to the raw log text")
    ap.add_argument("-o", "--output", required=True, help="Path to write the CSV table")
    ap.add_argument(
        "--bn-order",
        help="Comma-separated bottleneck order, e.g. 8,16,24,32,64 (default: auto-detect & sort)",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    rows = list(parse_runs(text))
    if not rows:
        print("[error] No (Rank,Bottleneck,Accuracy) triples parsed.", file=sys.stderr)
        sys.exit(1)

    bn_order = [int(x.strip()) for x in args.bn_order.split(",")] if args.bn_order else None
    header, table_rows = build_table(rows, bn_order)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table_rows)

    print(f"[ok] Wrote {len(table_rows)} rows to {args.output}")

if __name__ == "__main__":
    main()

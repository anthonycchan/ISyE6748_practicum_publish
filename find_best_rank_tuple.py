#!/usr/bin/env python3
import csv
from typing import List, Tuple, Any

# ===== User variable =====
input_path = "output.csv"  # <== change this to your CSV file path
# =========================

def read_rows(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = []
        for row in rdr:
            clean = {}
            for k, v in row.items():
                v = v.strip() if isinstance(v, str) else v
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    clean[k] = v
            rows.append(clean)
        return rows

def lex_min(rows: List[dict], key_cols: List[str]) -> Tuple[dict, Tuple[float, ...]]:
    def key_fn(r: dict) -> Tuple[Any, ...]:
        return tuple(r[c] for c in key_cols)
    best_row = min(rows, key=key_fn)
    return best_row, key_fn(best_row)

def main():
    key_cols = ["fp", "p95", "mean"]
    rows = read_rows(input_path)
    best_row, best_tuple = lex_min(rows, key_cols)

    rank_val = best_row.get("rank", "<unknown>")
    print(f"Input file: {input_path}")
    print(f"Best rank: {rank_val}")
    print("Best tuple (fp, p95, mean):", best_tuple)

if __name__ == "__main__":
    main()

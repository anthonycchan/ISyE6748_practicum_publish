#!/usr/bin/env python3
# Parses Tucker rank-search logs where the "Rank: r1 r2 r3 Accuracy: X" line
# comes *after* the AUC/Acc@best lines. Input: data.txt. Output: plain text.
import re
import math

NUM = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)'

# "FINAL AUC=0.658 | Acc@best=0.630"
AUC_ACC_LINE_RE = re.compile(r'FINAL\s*AUC\s*=\s*' + NUM + r'\s*\|\s*Acc@best\s*=\s*' + NUM, re.IGNORECASE)
# Fallbacks if they ever appear on separate lines
AUC_ONLY_RE     = re.compile(r'FINAL\s*AUC\s*=\s*' + NUM, re.IGNORECASE)
ACC_BEST_ONLY_RE= re.compile(r'Acc@best\s*=\s*' + NUM, re.IGNORECASE)

# "Rank: 5 5 16 Accuracy: 0.61"  (trailing Accuracy is optional)
RANK_TUPLE_RE   = re.compile(r'(?m)^Rank:\s*(\d+)\s+(\d+)\s+(\d+)(?:\s+Accuracy:\s*' + NUM + r')?\s*$')

def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float('nan')

def fmt_num(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"

def parse_streaming(text: str):
    """
    Stream through lines, remember the most recent AUC/Acc@best seen.
    When we hit a Rank line, pair it with the last seen metrics.
    """
    last_auc = float('nan')
    last_acc = float('nan')
    rows = []

    for line in text.splitlines():
        line = line.strip()

        m_both = AUC_ACC_LINE_RE.search(line)
        if m_both:
            last_auc = _to_float(m_both.group(1))
            last_acc = _to_float(m_both.group(2))
            continue

        m_auc = AUC_ONLY_RE.search(line)
        if m_auc:
            last_auc = _to_float(m_auc.group(1))
            continue

        m_accbest = ACC_BEST_ONLY_RE.search(line)
        if m_accbest:
            last_acc = _to_float(m_accbest.group(1))
            continue

        m_rank = RANK_TUPLE_RE.search(line)
        if m_rank:
            r1, r2, r3 = map(int, m_rank.group(1,2,3))
            # If trailing "Accuracy: ..." is present on the same line, prefer it.
            acc_inline = m_rank.group(4)
            acc_val = _to_float(acc_inline) if acc_inline is not None else last_acc
            rows.append({"rank": (r1, r2, r3), "acc": acc_val, "auc": last_auc})

    return rows

def main():
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    rows = parse_streaming(text)

    # Print each parsed row
    for r in rows:
        r1, r2, r3 = r["rank"]
        print(f"Rank: {r1} {r2} {r3} | Accuracy: {fmt_num(r['acc'])} | AUC: {fmt_num(r['auc'])}")

    # Best by Accuracy (ignore NaNs)
    valid_acc = [r for r in rows if not math.isnan(r["acc"])]
    if valid_acc:
        best_acc = max(valid_acc, key=lambda x: x["acc"])
        r1, r2, r3 = best_acc["rank"]
        print(f"Rank with highest Accuracy: ({r1},{r2},{r3}) (Accuracy={fmt_num(best_acc['acc'])}, AUC={fmt_num(best_acc['auc'])})")
    else:
        print("Rank with highest Accuracy: N/A")

    # Best by AUC (ignore NaNs)
    valid_auc = [r for r in rows if not math.isnan(r["auc"])]
    if valid_auc:
        best_auc = max(valid_auc, key=lambda x: x["auc"])
        r1, r2, r3 = best_auc["rank"]
        print(f"Rank with highest AUC: ({r1},{r2},{r3}) (AUC={fmt_num(best_auc['auc'])}, Accuracy={fmt_num(best_auc['acc'])})")
    else:
        print("Rank with highest AUC: N/A")

if __name__ == "__main__":
    main()

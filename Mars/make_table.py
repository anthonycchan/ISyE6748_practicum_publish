#!/usr/bin/env python3
# Parses "Autoencoder (raw-pixel) — sweeping factors and bottlenecks" logs.
# The AE metrics lines come BEFORE the "Factor: X Bottleneck: Y Accuracy Z" line.
# Input: data.txt
# Output (per block):
#   Factor: <f> Bottleneck: <b> | ValAcc: <v> | FinalAcc: <fa> | AUC: <auc>
# Then summary lines for best FinalAcc and best AUC.

import re
import math

NUM = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)'

VALACC_RE    = re.compile(r'\[AE\(raw\)\]\s*Accuracy\s*@\s*VAL-derived\s*threshold\s*:\s*' + NUM, re.IGNORECASE)
AUC_RE       = re.compile(r'\[AE\(raw\)\]\s*FINAL\s*AUC\s*=\s*' + NUM, re.IGNORECASE)
FINALACC_RE  = re.compile(r'\[AE\(raw\)\]\s*Chosen\s+threshold.*?\|\s*Accuracy\s*=\s*' + NUM, re.IGNORECASE)
FACTOR_RE    = re.compile(r'(?m)^\s*Factor:\s*(\d+)\s+Bottleneck:\s*(\d+)\s+Accuracy\s+' + NUM + r'\s*$')

def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float('nan')

def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"

def parse_stream(text: str):
    """
    Walk the file once, remembering the most recent AE metrics.
    When a 'Factor: ... Bottleneck: ... Accuracy ...' line appears, pair it
    with the latest metrics observed above it.
    """
    last_val_acc   = float('nan')
    last_final_auc = float('nan')
    last_final_acc = float('nan')
    rows = []

    for raw in text.splitlines():
        line = raw.strip()

        m = VALACC_RE.search(line)
        if m:
            last_val_acc = _to_float(m.group(1))
            continue

        m = AUC_RE.search(line)
        if m:
            last_final_auc = _to_float(m.group(1))
            continue

        m = FINALACC_RE.search(line)
        if m:
            last_final_acc = _to_float(m.group(1))
            continue

        m = FACTOR_RE.search(line)
        if m:
            f  = int(m.group(1))
            bn = int(m.group(2))
            trailing = _to_float(m.group(3))  # accuracy on the Factor line
            rows.append({
                "factor": f,
                "bottleneck": bn,
                "val_acc": last_val_acc if not math.isnan(last_val_acc) else trailing,
                "final_acc": last_final_acc,
                "auc": last_final_auc,
                "factor_line_acc": trailing
            })
            # Do not reset metrics; logs typically repeat the metric trio before each Factor line

    return rows

def main():
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    rows = parse_stream(text)

    # Print each parsed row
    for r in rows:
        print(
            f"Factor: {r['factor']} Bottleneck: {r['bottleneck']} | "
            f"ValAcc: {_fmt(r['val_acc'])} | FinalAcc: {_fmt(r['final_acc'])} | AUC: {_fmt(r['auc'])}"
        )

    # Best by FinalAcc (ignore NaNs)
    valid_final = [r for r in rows if not math.isnan(r["final_acc"])]
    if valid_final:
        best_final = max(valid_final, key=lambda x: x["final_acc"])
        print(
            f"Best by FinalAcc: (Factor={best_final['factor']}, Bottleneck={best_final['bottleneck']}) "
            f"(FinalAcc={_fmt(best_final['final_acc'])}, AUC={_fmt(best_final['auc'])}, ValAcc={_fmt(best_final['val_acc'])})"
        )
    else:
        print("Best by FinalAcc: N/A")

    # Best by AUC (ignore NaNs)
    valid_auc = [r for r in rows if not math.isnan(r["auc"])]
    if valid_auc:
        best_auc = max(valid_auc, key=lambda x: x["auc"])
        print(
            f"Best by AUC: (Factor={best_auc['factor']}, Bottleneck={best_auc['bottleneck']}) "
            f"(AUC={_fmt(best_auc['auc'])}, FinalAcc={_fmt(best_auc['final_acc'])}, ValAcc={_fmt(best_auc['val_acc'])})"
        )
    else:
        print("Best by AUC: N/A")

if __name__ == "__main__":
    main()

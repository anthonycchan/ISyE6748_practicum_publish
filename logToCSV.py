#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from collections import OrderedDict
from typing import List, Dict, Any

EN_DASH = "\u2013"

# ---------- Rank header patterns ----------

RANK3_RE = re.compile(r"^Rank:\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
RANK1_RE = re.compile(r"^Rank:\s*(\d+)\s*$")

# ---------- Metrics patterns ----------

# Tucker metrics with tuple rank
TUCKER_METRICS_RE = re.compile(
    r"^\[(?P<method>[^\]]+)\]\s+R=\((?P<r1>\d+),\s*(?P<r2>\d+),\s*(?P<r3>\d+)\)\s+"
    r"peak_rss_mb=(?P<rss>[-\d.]+)\s+\|\s+peak_workingset_mb=(?P<ws>[-\d.]+)\s+\|\s+"
    r"peak_cuda_mb=(?P<cuda>[-\d.]+)\s+\|\s+cpu_time_sec=(?P<cpu>[-\d.]+)\s*$"
)

# Generic single-R metrics (CP / PCA / Tucker), supports R= / rank= / RANK=
LEGACY_METRICS_RE = re.compile(
    r"^\[(?P<method>[^\]]+)\]\s+(?:R|rank|RANK)=(?P<R>\d+)"
    r"\s+peak_rss_mb=(?P<rss>[-\d.]+)"
    r"\s+\|\s+peak_workingset_mb=(?P<ws>[-\d.]+)"
    r"\s+\|\s+peak_cuda_mb=(?P<cuda>[-\d.]+)"
    r"(?:\s+\|\s+cpu_time_sec=(?P<cpu>[-\d.]+))?\s*$"
)

# ---------- Best / search patterns ----------

# [X+OCSVM] (VAL one-class) chose {...} best_obj:(FP=..., P95=..., mean=...) Elapsed: ...
OCSVM_BEST_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+OCSVM)\]\s+\(VAL one-class\)\s+chose\s+(?P<param>\{.*?\})\s+"
    r"best_obj:\(FP=(?P<fp>[-\d.eE+]+),\s*P95=(?P<p95>[-\d.eE+]+),\s*mean=(?P<mean>[-\d.eE+]+)\)\s+"
    r"Elapsed:\s+(?P<elapsed>[\d.]+)\s*$"
)

# [X+AE] best_param=(...) best_obj=... Elapsed: ...
AE_BEST_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+AE)\]\s+best_param=(?P<param>\(.*?\))\s+"
    r"best_obj=(?P<best>[-\d.eE+]+)\s+Elapsed:\s+(?P<elapsed>[\d.]+)\s*$"
)

# [X+IF] best_obj=... best_param: {...} Elapsed: ... (or Elapsed (CPU s): ...)
IF_BEST_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+IF)\]\s+best_obj=(?P<best>[-\d.eE+]+)\s+"
    r"best_param:\s+(?P<param>\{.*?\})\s+Elapsed(?:\s*\(CPU s\))?:\s+(?P<elapsed>[\d.]+)\s*$"
)

# ---------- Final result patterns ----------

# Supports:
# - RANK=10
# - Rank=10
# - rank=10
# - RANK=(5, 5, 5)
# - rank=(5,5,5)
FINAL_RESULT_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+(?:OCSVM|AE|IF))\].*?Final result.*?"
    r"(?:\b(?:RANK|Rank|rank)="
    r"(?:(?P<rank>\d+)|\(\s*(?P<tr1>\d+)\s*,\s*(?P<tr2>\d+)\s*,\s*(?P<tr3>\d+)\s*\)))?"
    r".*?\bAUC=(?P<auc>[-\d.eE+]+)"
    r"\s*(?:\|\s*)?ACC=(?P<acc>[-\d.eE+]+)\b.*$"
)

# IF-only extras on same Final result line
IF_FINAL_EXTRAS_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+IF)\].*?Final result.*?"
    r"obj=(?P<obj>[-\d.eE+]+)\|.*?"
    r"thr=(?P<thr>[-\d.eE+]+)\s*\|\s*target_FP=(?P<tfp>[-\d.eE+]+)\b.*$"
)

# ---------- Bootstrap ----------

BOOT_RE = re.compile(
    r"^\[(?P<tag>(?:CP|Tucker|PCA)\+(?:OCSVM|AE|IF))\]\s+AUC=(?P<auc>[-\d.eE+]+)\s*Boot:\s*"
    r"mean:(?P<mean>[-\d.eE+]+)\s*std:(?P<std>[-\d.eE+]+),\s*CI\((?P<lo>[-\d.eE+]+)["
    + EN_DASH +
    r"-](?P<hi>[-\d.eE+]+)\)\s*\|\s*$"
)

# ---------- Generic best_obj fallback ----------

GENERIC_BESTOBJ_RE = re.compile(
    r"^\[(?P<tag>[^\]]+)\].*?\bbest_obj=(?P<val>.+?)(?:\s+\|\s+.*|\s+Elapsed.*|$)"
)

# ---------- Misc ----------

IGNORE_PREFIXES = (
    "WARNING:",
    "Console output is saving to:",
    "TensorLy backend:",
    "Rank search",
    "Split seed:",
    "[TRAIN]",
    "[VAL]",
    "[FINAL]",
)

def _clean_method_tag(tag: str) -> str:
    return tag.strip()

def _rank_key_3(a: str, b: str, c: str) -> str:
    return f"{a}x{b}x{c}"

def _rank_tuple_str(a: str, b: str, c: str) -> str:
    return f"({a}, {b}, {c})"

def _rank_sort_key(rank_key: str):
    if "x" in rank_key:
        a, b, c = (int(x) for x in rank_key.split("x"))
        return (a, b, c, 1)
    return (int(rank_key), 0, 0, 0)

METHOD_ORDER = {
    "CP only": 0,
    "Tucker only": 0,
    "PCA only": 0,
    "CP+OCSVM": 1,
    "Tucker+OCSVM": 1,
    "PCA+OCSVM": 1,
    "CP+AE": 2,
    "Tucker+AE": 2,
    "PCA+AE": 2,
    "CP+IF": 3,
    "Tucker+IF": 3,
    "PCA+IF": 3,
}

def _set_once(row: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return
    if not row.get(key):
        row[key] = value

# ---------- Core parsing ----------

def parse_file(path: str, group_mode: str = "method") -> List[Dict[str, Any]]:
    rows: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
    current_rank_key: str | None = None
    current_rank_display: str = ""
    pending_best: Dict[str, Dict[str, Any]] = {}

    def get_row(rank_key: str, method: str) -> Dict[str, Any]:
        key = (rank_key, method)
        if key not in rows:
            rows[key] = {
                "rank_key": rank_key,
                "rank_display": current_rank_display,
                "method": method,
                "R": "",
                "peak_rss_mb": "",
                "peak_workingset_mb": "",
                "peak_cuda_mb": "",
                "cpu_time_sec": "",
                "best_param": "",
                "best_obj": "",
                "best_obj_raw": "",
                "elapsed": "",
                "fp": "",
                "p95": "",
                "mean": "",
                "auc": "",
                "acc": "",
                "boot_mean": "",
                "boot_std": "",
                "ci_lo": "",
                "ci_hi": "",
                "thr": "",
                "target_fp": "",
                "obj": "",
            }
        return rows[key]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(IGNORE_PREFIXES):
                continue

            # ----- Rank headers -----
            m = RANK3_RE.match(line)
            if m:
                r1, r2, r3 = m.groups()
                current_rank_key = _rank_key_3(r1, r2, r3)
                current_rank_display = _rank_tuple_str(r1, r2, r3)
                pending_best.clear()
                continue

            m = RANK1_RE.match(line)
            if m:
                r = m.group(1)
                current_rank_key = r
                current_rank_display = r
                pending_best.clear()
                continue

            # ----- Tucker tuple-R metrics -----
            m = TUCKER_METRICS_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("method"))
                r1, r2, r3 = m.group("r1"), m.group("r2"), m.group("r3")
                rank_key = _rank_key_3(r1, r2, r3)
                current_rank_key = rank_key
                current_rank_display = _rank_tuple_str(r1, r2, r3)
                row = get_row(rank_key, method)
                _set_once(row, "R", current_rank_display)
                _set_once(row, "peak_rss_mb", m.group("rss"))
                _set_once(row, "peak_workingset_mb", m.group("ws"))
                _set_once(row, "peak_cuda_mb", m.group("cuda"))
                _set_once(row, "cpu_time_sec", m.group("cpu"))
                continue

            # ----- Single-R metrics (CP/PCA/Tucker) -----
            m = LEGACY_METRICS_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("method"))
                R = m.group("R")
                rank_key = R
                current_rank_key = rank_key
                current_rank_display = R
                row = get_row(rank_key, method)
                _set_once(row, "R", R)
                _set_once(row, "peak_rss_mb", m.group("rss"))
                _set_once(row, "peak_workingset_mb", m.group("ws"))
                _set_once(row, "peak_cuda_mb", m.group("cuda"))
                if m.group("cpu") is not None:
                    _set_once(row, "cpu_time_sec", m.group("cpu"))
                continue

            # ----- OCSVM best -----
            m = OCSVM_BEST_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("tag"))
                info = {
                    "best_param": m.group("param"),
                    "elapsed": m.group("elapsed"),
                    "fp": m.group("fp"),
                    "p95": m.group("p95"),
                    "mean": m.group("mean"),
                    "best_obj_raw": (
                        f"(FP={m.group('fp')}, P95={m.group('p95')}, mean={m.group('mean')})"
                    ),
                }
                if current_rank_key is not None:
                    row = get_row(current_rank_key, method)
                    for k, v in info.items():
                        _set_once(row, k, v)
                else:
                    pending_best.setdefault(method, {}).update(info)
                continue

            # ----- AE best -----
            m = AE_BEST_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("tag"))
                info = {
                    "best_param": m.group("param"),
                    "best_obj": m.group("best"),
                    "best_obj_raw": m.group("best"),
                    "elapsed": m.group("elapsed"),
                }
                if current_rank_key is not None:
                    row = get_row(current_rank_key, method)
                    for k, v in info.items():
                        _set_once(row, k, v)
                else:
                    pending_best.setdefault(method, {}).update(info)
                continue

            # ----- IF best -----
            m = IF_BEST_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("tag"))
                info = {
                    "best_param": m.group("param"),
                    "best_obj": m.group("best"),
                    "best_obj_raw": m.group("best"),
                    "elapsed": m.group("elapsed"),
                }
                if current_rank_key is not None:
                    row = get_row(current_rank_key, method)
                    for k, v in info.items():
                        _set_once(row, k, v)
                else:
                    pending_best.setdefault(method, {}).update(info)
                continue

            # ----- Generic best_obj fallback (non-final) -----
            if "Final result" not in line:
                m = GENERIC_BESTOBJ_RE.match(line)
                if m:
                    method = _clean_method_tag(m.group("tag"))
                    val = m.group("val")
                    if current_rank_key is not None:
                        row = get_row(current_rank_key, method)
                        _set_once(row, "best_obj_raw", val)
                    else:
                        pending_best.setdefault(method, {}).setdefault(
                            "best_obj_raw", val
                        )

            # ----- Final result (scalar or tuple rank) -----
            m = FINAL_RESULT_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("tag"))

                rank_scalar = m.group("rank")
                tr1, tr2, tr3 = m.group("tr1"), m.group("tr2"), m.group("tr3")

                if tr1 and tr2 and tr3:
                    # Tuple rank from line e.g. (5, 5, 5)
                    rank_key = _rank_key_3(tr1, tr2, tr3)
                    current_rank_display = _rank_tuple_str(tr1, tr2, tr3)
                elif rank_scalar:
                    # Scalar rank from line
                    rank_key = rank_scalar
                    current_rank_display = rank_scalar
                else:
                    # Fall back to current rank context
                    rank_key = current_rank_key

                if rank_key is None:
                    # No reliable rank; skip to avoid corrupt grouping
                    continue

                current_rank_key = rank_key
                row = get_row(rank_key, method)

                _set_once(row, "R", current_rank_display or row.get("R", ""))
                _set_once(row, "auc", m.group("auc"))
                _set_once(row, "acc", m.group("acc"))

                # IF extras (obj, thr, target_FP)
                m2 = IF_FINAL_EXTRAS_RE.match(line)
                if m2:
                    _set_once(row, "obj", m2.group("obj"))
                    _set_once(row, "thr", m2.group("thr"))
                    _set_once(row, "target_fp", m2.group("tfp"))

                # Attach any pending best info for this method
                if method in pending_best:
                    for k, v in pending_best[method].items():
                        _set_once(row, k, v)
                    pending_best.pop(method, None)

                continue

            # ----- Bootstrap line -----
            m = BOOT_RE.match(line)
            if m:
                method = _clean_method_tag(m.group("tag"))
                if current_rank_key is None:
                    continue
                row = get_row(current_rank_key, method)
                _set_once(row, "auc", row["auc"] or m.group("auc"))
                _set_once(row, "boot_mean", m.group("mean"))
                _set_once(row, "boot_std", m.group("std"))
                _set_once(row, "ci_lo", m.group("lo"))
                _set_once(row, "ci_hi", m.group("hi"))
                continue

            # Everything else: ignore

    # ---------- Sorting / grouping ----------

    def _key_method(r: Dict[str, Any]):
        return (
            METHOD_ORDER.get(r["method"], 99),
            _rank_sort_key(r["rank_key"]),
            r["method"],
        )

    def _key_rank(r: Dict[str, Any]):
        return (
            _rank_sort_key(r["rank_key"]),
            METHOD_ORDER.get(r["method"], 99),
            r["method"],
        )

    key_fn = _key_method if group_mode.lower() == "method" else _key_rank
    return sorted(rows.values(), key=key_fn)

# ---------- CSV writing ----------

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "rank_key",
        "rank_display",
        "method",
        "R",
        "peak_rss_mb",
        "peak_workingset_mb",
        "peak_cuda_mb",
        "cpu_time_sec",
        "best_param",
        "best_obj",
        "best_obj_raw",
        "elapsed",
        "fp",
        "p95",
        "mean",
        "auc",
        "acc",
        "boot_mean",
        "boot_std",
        "ci_lo",
        "ci_hi",
        "thr",
        "target_fp",
        "obj",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Parse experiment logs into CSV.")
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=r"C:/OMSA/ISyE6748_practicum_publish/Results/Train/Tucker/tucker_seed5_3000reduced.out",
        help="Path to input log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=r"C:/OMSA/ISyE6748_practicum_publish/Results/parsed.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "-g",
        "--group",
        choices=["method", "rank"],
        default="method",
        help="Primary grouping for CSV ordering",
    )
    args = parser.parse_args()

    rows = parse_file(args.input, group_mode=args.group)
    write_csv(rows, args.output)
    print(
        f"Parsed {len(rows)} row(s). CSV written to: {args.output} (grouped by {args.group})"
    )

if __name__ == "__main__":
    main()

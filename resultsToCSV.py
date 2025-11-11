import re
import csv
from collections import defaultdict
import sys
import os

# ------------------------------
# CONFIGURATION
# ------------------------------

# Option 1: Set manually here
INPUT_PATH = "C:/OMSA/ISyE6748_practicum_publish/Results/results.out"  # Change to your input log file path
OUTPUT_DIR = "Results"               # Folder to save the CSV files

# Option 2: Allow overrides via command line:
#   python parse_auc.py results.out output_folder
if len(sys.argv) > 1:
    INPUT_PATH = sys.argv[1]
if len(sys.argv) > 2:
    OUTPUT_DIR = sys.argv[2]

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# REGEX PATTERNS
# ------------------------------

seed_pattern = re.compile(r"Split seed:\s*(\d+)")
auc_patterns = {
    "ocsvm_pca":    re.compile(r"\[PCA\+OCSVM\].*?AUC=([0-9]*\.[0-9]+)"),
    "ocsvm_cp":     re.compile(r"\[CP\+OCSVM\].*?AUC=([0-9]*\.[0-9]+)"),
    "ocsvm_tucker": re.compile(r"\[Tucker\+OCSVM\].*?AUC=([0-9]*\.[0-9]+)"),

    "ae_pca":       re.compile(r"\[PCA\+AE\].*?AUC=([0-9]*\.[0-9]+)"),
    "ae_cp":        re.compile(r"\[CP\+AE\].*?AUC=([0-9]*\.[0-9]+)"),
    "ae_tucker":    re.compile(r"\[Tucker\+AE\].*?AUC=([0-9]*\.[0-9]+)"),

    "if_pca":       re.compile(r"\[PCA\+IF\].*?AUC=([0-9]*\.[0-9]+)"),
    "if_cp":        re.compile(r"\[CP\+IF\].*?AUC=([0-9]*\.[0-9]+)"),
    "if_tucker":    re.compile(r"\[Tucker\+IF\].*?AUC=([0-9]*\.[0-9]+)"),
}

# ------------------------------
# PARSING
# ------------------------------

def parse_aucs(path):
    results = defaultdict(dict)
    current_seed = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_seed = seed_pattern.search(line)
            if m_seed:
                current_seed = int(m_seed.group(1))
                continue

            if current_seed is None:
                continue

            for key, pat in auc_patterns.items():
                m_auc = pat.search(line)
                if m_auc:
                    auc = float(m_auc.group(1))
                    # Keep first AUC per (model, seed); ignore later duplicates
                    results[key].setdefault(current_seed, auc)

    return results

# ------------------------------
# HELPERS
# ------------------------------

def get_val(results, key, seed):
    return results.get(key, {}).get(seed, "")

def get_avg(results, key, seeds):
    vals = [
        results.get(key, {}).get(s)
        for s in seeds
        if s in results.get(key, {})
    ]
    if not vals:
        return ""
    return sum(vals) / len(vals)

def write_csv(filename, header, row, outdir):
    out_path = os.path.join(outdir, filename)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    print(f"  → {out_path}")

# ------------------------------
# CSV WRITER (WITH AVERAGES)
# ------------------------------

def write_csvs(results, outdir, seeds=(1, 3, 5, 8, 13)):
    # ----- OCSVM -----
    ocsvm_header = [
        # PCA
        "ocsvm_pca_seed1", "ocsvm_pca_seed3", "ocsvm_pca_seed5", "ocsvm_pca_seed8", "ocsvm_pca_seed13",
        "ocsvm_pca_avg",
        # CP
        "ocsvm_cp_seed1", "ocsvm_cp_seed3", "ocsvm_cp_seed5", "ocsvm_cp_seed8", "ocsvm_cp_seed13",
        "ocsvm_cp_avg",
        # Tucker
        "ocsvm_tucker_seed1", "ocsvm_tucker_seed3", "ocsvm_tucker_seed5", "ocsvm_tucker_seed8", "ocsvm_tucker_seed13",
        "ocsvm_tucker_avg",
    ]

    ocsvm_row = [
        # PCA seeds
        *[get_val(results, "ocsvm_pca", s) for s in seeds],
        get_avg(results, "ocsvm_pca", seeds),
        # CP seeds
        *[get_val(results, "ocsvm_cp", s) for s in seeds],
        get_avg(results, "ocsvm_cp", seeds),
        # Tucker seeds
        *[get_val(results, "ocsvm_tucker", s) for s in seeds],
        get_avg(results, "ocsvm_tucker", seeds),
    ]

    write_csv("ocsvm_auc.csv", ocsvm_header, ocsvm_row, outdir)

    # ----- AE -----
    ae_header = [
        # PCA
        "ae_pca_seed1", "ae_pca_seed3", "ae_pca_seed5", "ae_pca_seed8", "ae_pca_seed13",
        "ae_pca_avg",
        # CP
        "ae_cp_seed1", "ae_cp_seed3", "ae_cp_seed5", "ae_cp_seed8", "ae_cp_seed13",
        "ae_cp_avg",
        # Tucker
        "ae_tucker_seed1", "ae_tucker_seed3", "ae_tucker_seed5", "ae_tucker_seed8", "ae_tucker_seed13",
        "ae_tucker_avg",
    ]

    ae_row = [
        # PCA seeds
        *[get_val(results, "ae_pca", s) for s in seeds],
        get_avg(results, "ae_pca", seeds),
        # CP seeds
        *[get_val(results, "ae_cp", s) for s in seeds],
        get_avg(results, "ae_cp", seeds),
        # Tucker seeds
        *[get_val(results, "ae_tucker", s) for s in seeds],
        get_avg(results, "ae_tucker", seeds),
    ]

    write_csv("ae_auc.csv", ae_header, ae_row, outdir)

    # ----- IF -----
    if_header = [
        # PCA
        "if_pca_seed1", "if_pca_seed3", "if_pca_seed5", "if_pca_seed8", "if_pca_seed13",
        "if_pca_avg",
        # CP
        "if_cp_seed1", "if_cp_seed3", "if_cp_seed5", "if_cp_seed8", "if_cp_seed13",
        "if_cp_avg",
        # Tucker
        "if_tucker_seed1", "if_tucker_seed3", "if_tucker_seed5", "if_tucker_seed8", "if_tucker_seed13",
        "if_tucker_avg",
    ]

    if_row = [
        # PCA seeds
        *[get_val(results, "if_pca", s) for s in seeds],
        get_avg(results, "if_pca", seeds),
        # CP seeds
        *[get_val(results, "if_cp", s) for s in seeds],
        get_avg(results, "if_cp", seeds),
        # Tucker seeds
        *[get_val(results, "if_tucker", s) for s in seeds],
        get_avg(results, "if_tucker", seeds),
    ]

    write_csv("if_auc.csv", if_header, if_row, outdir)

# ------------------------------
# MAIN
# ------------------------------

if __name__ == "__main__":
    print(f"Parsing input file: {INPUT_PATH}")
    print(f"Writing output to directory: {OUTPUT_DIR}")
    res = parse_aucs(INPUT_PATH)
    write_csvs(res, OUTPUT_DIR)
    print("Done.")

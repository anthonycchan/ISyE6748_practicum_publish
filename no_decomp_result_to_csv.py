import re
import csv

# ----------------------------------------------------
# EDIT THESE TWO LINES ONLY
INPUT_FILE = "Results/results.out"
OUTPUT_FILE = "Results/no_decomp.csv"
# ----------------------------------------------------

# Regex patterns
seed_pattern = re.compile(r"Split seed:\s*(\d+)")
ocsvm_pattern = re.compile(r"OCSVM FINAL AUC=([0-9.]+)")
ae_pattern = re.compile(r"AE Final result .* AUC=([0-9.]+)")
if_pattern = re.compile(r"IF Final result AUC=([0-9.]+)")

def parse_file(path):
    results = {
        "ocsvm": {},
        "ae": {},
        "if": {}
    }

    current_seed = None

    with open(path, "r") as f:
        for line in f:
            # Detect seed
            m = seed_pattern.search(line)
            if m:
                current_seed = int(m.group(1))
                continue

            if current_seed is None:
                continue

            # Extract AUCs
            m = ocsvm_pattern.search(line)
            if m:
                results["ocsvm"][current_seed] = float(m.group(1))
                continue

            m = ae_pattern.search(line)
            if m:
                results["ae"][current_seed] = float(m.group(1))
                continue

            m = if_pattern.search(line)
            if m:
                results["if"][current_seed] = float(m.group(1))
                continue

    return results


def write_csv(results, out_path):
    seeds = [1, 3, 5]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed_1", "seed_3", "seed_5", "average"])

        for method in ["ocsvm", "ae", "if"]:
            vals = [results[method].get(s, "") for s in seeds]
            numeric_vals = [v for v in vals if v != ""]
            avg = sum(numeric_vals) / len(numeric_vals) if numeric_vals else ""

            writer.writerow([method] + vals + [avg])


if __name__ == "__main__":
    parsed = parse_file(INPUT_FILE)
    write_csv(parsed, OUTPUT_FILE)
    print("CSV created:", OUTPUT_FILE)

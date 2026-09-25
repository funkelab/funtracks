"""Compare two pytest-benchmark JSON files and produce a markdown table.

Usage:
    python benchmark_pr.py <old.json> <new.json> <output.md> [header]

Exits with code 1 if any benchmark regresses by more than REGRESSION_THRESHOLD.
"""

import json
import sys
import pandas as pd

REGRESSION_THRESHOLD = 50  # percent
# Benchmarks with a median below this floor are dominated by timer resolution and OS
# scheduling jitter rather than the code being measured, so a "regression" on them is
# noise, not signal: skip the gate (and still report the row) unless the benchmark is
# slow on *both* sides, which means it should have been caught by the floor already.
NOISE_FLOOR_SECONDS = 0.02

def load_stats(path):
    with open(path) as f:
        data = json.load(f)

    commit = data["commit_info"]["id"]

    rows = []
    for d in data["benchmarks"]:
        rows.append({"Benchmark": d["name"], "median": d["stats"]["median"]})

    return commit, pd.DataFrame(rows)


def make_report(old_path, new_path, out_file, header=None):
    old = load_stats(old_path)
    new = load_stats(new_path)

    # Merge on benchmark name. Outer join so benchmarks that exist on only one
    # side (added or removed by the PR) still show up in the report.
    df = old[-1].merge(new[-1], on="Benchmark", suffixes=("_old", "_new"), how="outer")

    pct_change = 100 * (df["median_new"] - df["median_old"]) / df["median_old"]
    df["Percent Change"] = pct_change.map("{:+.2f}".format).where(
        pct_change.notna(), "n/a"
    )

    # A benchmark below this floor on both sides is dominated by timer resolution and
    # OS scheduling jitter rather than the code being measured, so a "regression" on it
    # is noise, not signal: exclude it from the gate (still shown in the report). NaN
    # (benchmark missing on one side) treated as below the floor, not above.
    above_floor = (df["median_old"].fillna(0) >= NOISE_FLOOR_SECONDS) | (
        df["median_new"].fillna(0) >= NOISE_FLOOR_SECONDS
    )

    # Format runtimes
    for col in ("median_old", "median_new"):
        df[col] = df[col].map("{:.5f}".format).where(df[col].notna(), "-")

    # Change column names to commit ids
    df = df.rename(
        columns={
            "median_new": f"Median (s) HEAD {new[0]}",
            "median_old": f"Median (s) BASE {old[0]}",
        }
    )

    report = df.to_markdown(index=False)
    if header:
        report = f"## {header}\n\n{report}"

    with open(out_file, "w") as f:
        f.write(report)

    # Print report to logs
    print(report)  # noqa: T201

    # Fail if any benchmark above the noise floor regressed beyond threshold.
    if (above_floor & (pct_change > REGRESSION_THRESHOLD)).any():
        print(  # noqa: T201
            f"\nFAILED: Regression exceeds {REGRESSION_THRESHOLD}% threshold"
        )
        sys.exit(1)


if __name__ == "__main__":
    header = sys.argv[4] if len(sys.argv) > 4 else None
    make_report(sys.argv[1], sys.argv[2], sys.argv[3], header)

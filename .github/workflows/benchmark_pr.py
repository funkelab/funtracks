"""Compare two pytest-benchmark JSON files and produce a markdown table.

Usage:
    python benchmark_pr.py <old.json> <new.json> <output.md> [header]

Exits with code 1 if any benchmark regresses by more than REGRESSION_THRESHOLD.
"""

import json
import sys

import pandas as pd

REGRESSION_THRESHOLD = 50  # percent


def load_stats(path):
    with open(path) as f:
        data = json.load(f)

    commit = data["commit_info"]["id"]

    rows = []
    for d in data["benchmarks"]:
        rows.append({"Benchmark": d["name"], "mean": d["stats"]["mean"]})

    return commit, pd.DataFrame(rows)


def make_report(old_path, new_path, out_file, header=None):
    old = load_stats(old_path)
    new = load_stats(new_path)

    # Merge on benchmark name
    df = old[-1].merge(new[-1], on="Benchmark", suffixes=("_old", "_new"))

    pct_change = 100 * (df["mean_new"] - df["mean_old"]) / df["mean_old"]
    df["Percent Change"] = pct_change.map("{:+.2f}".format)

    # Format runtimes
    df["mean_old"] = df["mean_old"].map("{:.5f}".format)
    df["mean_new"] = df["mean_new"].map("{:.5f}".format)

    # Change column names to commit ids
    df = df.rename(
        columns={
            "mean_new": f"Mean (s) HEAD {new[0]}",
            "mean_old": f"Mean (s) BASE {old[0]}",
        }
    )

    report = df.to_markdown(index=False)
    if header:
        report = f"## {header}\n\n{report}"

    with open(out_file, "w") as f:
        f.write(report)

    # Print report to logs
    print(report)  # noqa: T201

    # Fail if any benchmark regressed beyond threshold
    if (pct_change > REGRESSION_THRESHOLD).any():
        print(  # noqa: T201
            f"\nFAILED: Regression exceeds {REGRESSION_THRESHOLD}% threshold"
        )
        sys.exit(1)


if __name__ == "__main__":
    header = sys.argv[4] if len(sys.argv) > 4 else None
    make_report(sys.argv[1], sys.argv[2], sys.argv[3], header)

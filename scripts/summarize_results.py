#!/usr/bin/env python3
"""Collect every finished configuration of the given datasets into a table / Excel file (CPU only).

    python scripts/summarize_results.py mmar iemocap
    python scripts/summarize_results.py mmar --excel
    python scripts/summarize_results.py mmar --csv results/mmar.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caga.results import collect, save_excel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--excel", action="store_true", help="write <dataset>_results.xlsx to RUNS_BASE")
    ap.add_argument("--csv", help="write all rows of all datasets to this CSV file")
    args = ap.parse_args()

    import pandas as pd
    frames = []
    for ds in args.datasets:
        df = save_excel(ds) if args.excel else collect(ds)
        print(f"\n=== {ds}: {len(df)} configurations ===")
        if len(df):
            print(df.drop(columns=["dataset", "estimator", "norm"]).to_string(index=False))
        frames.append(df)
    if args.csv:
        pd.concat(frames, ignore_index=True).to_csv(args.csv, index=False)
        print("saved:", args.csv)


if __name__ == "__main__":
    main()

"""Check total rows across all raw/enriched CSVs."""
import os, pandas as pd

total = 0
for d in ["data/raw", "data/raw_enriched"]:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(d, f))
            total += len(df)
            print(f"  {len(df):>5} | {d}/{f}")
        except Exception as e:
            print(f"  ERROR | {d}/{f}: {e}")

print(f"\nTotal rows across all CSVs: {total}")

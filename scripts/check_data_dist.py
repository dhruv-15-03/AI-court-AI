"""Quick check of data label distribution."""
import pandas as pd

df = pd.read_csv("data/processed/all_cases.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()

vals = df["judgement"].value_counts()
print("Top 30 judgement values:")
for v, c in vals.head(30).items():
    print(f"  {c:>5} | {str(v)[:90]}")

print(f"\nUnique judgement values: {df['judgement'].nunique()}")
print(f"\nCase type dist:")
print(df["case_type"].value_counts().head(20).to_string())

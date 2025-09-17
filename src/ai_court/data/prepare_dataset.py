import os
import pandas as pd
from typing import List

REQUIRED = ["case_data", "case_type", "judgement"]


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    # Heuristics to map into required fields
    mapping = {}
    # case_data
    for cand in ["case_data", "case_text", "text", "case_summary", "summary"]:
        if cand in cols:
            mapping["case_data"] = cand
            break
    # case_type
    for cand in ["case_type", "type", "category", "law"]:
        if cand in cols:
            mapping["case_type"] = cand
            break
    # judgement
    for cand in ["judgement", "judgment", "judgment_text", "outcome"]:
        if cand in cols:
            mapping["judgement"] = cand
            break
    if set(mapping.keys()) != set(REQUIRED):
        missing = set(REQUIRED) - set(mapping.keys())
        raise ValueError(f"Cannot map columns to required schema; missing: {missing}")
    out = df[[mapping[c] for c in REQUIRED]].copy()
    out.columns = REQUIRED
    return out


def build_processed_dataset(inputs: List[str], out_csv: str = os.path.join("data", "processed", "all_cases.csv")) -> str:
    frames = []
    for path in inputs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        try:
            frames.append(coerce_schema(df))
        except Exception:
            # Skip incompatible files
            continue
    if not frames:
        raise RuntimeError("No compatible input CSVs found")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    candidates = [
        os.path.join("data", "rape_murder_cases.csv"),
        os.path.join("data", "rape_murder_cases_summarized.csv"),
        "legal_cases.csv",
        "property_disputes.csv",
        "rape.csv",
        "child_labour.csv",
    ]
    path = build_processed_dataset(candidates)
    print(path)

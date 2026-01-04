import os
import re
import glob
import pandas as pd
from typing import List, Iterable

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
            # Fallback mapping for our harvested raw CSVs (id,title,url,case_summary,judgment)
            cols = set(df.columns)
            if {"case_summary", "judgment"}.issubset(cols):
                # Derive case_type from filename e.g., data/raw/kanoon_rape.csv -> "rape"
                base = os.path.basename(path)
                name = os.path.splitext(base)[0]
                # strip common prefixes
                if name.startswith("kanoon_"):
                    name = name[len("kanoon_"):]
                case_type_val = name.replace("_", " ")
                out = pd.DataFrame({
                    "case_data": df["case_summary"].astype(str),
                    "case_type": case_type_val,
                    "judgement": df["judgment"].astype(str),
                })
                frames.append(out)
            else:
                # Skip incompatible files
                continue
    if not frames:
        raise RuntimeError("No compatible input CSVs found")
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    return out_csv


def _iter_csvs_from_dirs(dirs: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for d in dirs:
        if not d:
            continue
        # Match *.csv recursively
        pattern = os.path.join(d, "**", "*.csv")
        paths.extend(glob.glob(pattern, recursive=True))
    # stable order
    return sorted(list({os.path.normpath(p) for p in paths}))


def _canonicalize(text: str) -> str:
    # Lowercase, strip non-alnum, collapse spaces
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_from_dirs(input_dirs: List[str], out_csv: str, min_text_len: int = 0, dedupe: bool = True) -> str:
    csvs = _iter_csvs_from_dirs(input_dirs)
    if not csvs:
        raise RuntimeError(f"No CSVs found in: {input_dirs}")
    tmp = build_processed_dataset(csvs, out_csv)
    # Post-filter: min length and dedupe
    df = pd.read_csv(tmp)
    if min_text_len and min_text_len > 0:
        df = df[df["case_data"].astype(str).str.len() >= int(min_text_len)]
    if dedupe:
        canon = df["case_data"].astype(str).map(_canonicalize)
        df = df.loc[~canon.duplicated()].reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    # Optional directory-mode via env: INPUT_DIRS=dir1;dir2, OUT_CSV, MIN_TEXT_LEN, DEDUPE=0/1
    input_dirs = os.environ.get("INPUT_DIRS", "").strip()
    out_csv = os.environ.get("OUT_CSV", os.path.join("data", "processed", "all_cases.csv"))
    min_len_env = os.environ.get("MIN_TEXT_LEN", "0").strip()
    dedupe_env = os.environ.get("DEDUP", "1").strip()
    try:
        min_len = int(min_len_env) if min_len_env else 0
    except ValueError:
        min_len = 0
    dedupe_flag = (dedupe_env != "0")
    if input_dirs:
        dirs = [d for d in input_dirs.split(";") if d]
        path = build_from_dirs(dirs, out_csv, min_text_len=min_len, dedupe=dedupe_flag)
        print(path)
    else:
        candidates = [
            os.path.join("data", "rape_murder_cases.csv"),
            os.path.join("data", "rape_murder_cases_summarized.csv"),
            "legal_cases.csv",
            "property_disputes.csv",
            "rape.csv",
            "child_labour.csv",
        ]
        path = build_processed_dataset(candidates, out_csv)
        print(path)

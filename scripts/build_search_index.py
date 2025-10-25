"""
Build a simple TF-IDF search index over scraped cases with URLs and outcomes.
Defaults to data/raw or data/raw_enriched and writes models/search_index.pkl, but
supports CLI flags to customize paths and vectorizer size.
"""
import os
import sys
import glob
import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ai_court.model.legal_case_classifier import LegalCaseClassifier  # noqa: E402


def pick_first(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first existing column as a Series, else None."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

RAW_DIR = os.path.join("data", "raw")
RAW_ENRICHED_DIR = os.path.join("data", "raw_enriched")
OUT_PATH = os.path.join("models", "search_index.pkl")


def load_cases(input_dir: Optional[str] = None) -> pd.DataFrame:
    """Load cases from CSVs under input_dir (if provided) or default directories.
    Prefers data/raw_enriched when present; otherwise falls back to data/raw; finally to processed/all_cases.csv.
    """
    # Prefer enriched data if present
    if input_dir:
        search_dir = input_dir
    else:
        search_dir = RAW_ENRICHED_DIR if os.path.isdir(RAW_ENRICHED_DIR) and glob.glob(os.path.join(RAW_ENRICHED_DIR, "*.csv")) else RAW_DIR
    paths = glob.glob(os.path.join(search_dir, "*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # Heuristics to extract fields
            title = pick_first(df, ["title", "case_title", "case_name", "Title"])
            url = pick_first(df, ["url", "link", "case_url"])
            text = pick_first(df, ["case_summary", "summary", "judgment", "judgement", "case_data", "text"])
            # Build minimal frame
            if text is not None:
                f = pd.DataFrame({
                    "title": title if title is not None else "",
                    "url": url if url is not None else "",
                    "text": text,
                })
                frames.append(f)
        except Exception:
            continue
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        # Fallback to processed dataset
        proc = os.path.join("data", "processed", "all_cases.csv")
        if not os.path.exists(proc):
            raise RuntimeError("No raw CSVs found under data/raw and no processed dataset at data/processed/all_cases.csv")
        pdf = pd.read_csv(proc)
        # Expect columns: case_data, case_type, judgement
        if not all(c in pdf.columns for c in ["case_data", "case_type", "judgement"]):
            raise RuntimeError("Processed dataset missing required columns: ['case_data','case_type','judgement']")
        df = pd.DataFrame({
            "title": pdf["case_type"].astype(str),
            "url": "",
            "text": pdf["case_data"].astype(str),
        })
    # Minimal clean
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip().astype(bool)]
    # Derive missing titles from text
    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("")
    empty_title = df["title"].astype(str).str.strip() == ""
    df.loc[empty_title, "title"] = df.loc[empty_title, "text"].astype(str).str.split().str[:10].str.join(" ")
    return df


def build_index(df: pd.DataFrame, out_path: str = OUT_PATH, max_features: int = 50000, ngram_max: int = 2) -> str:
    clf = LegalCaseClassifier()
    preprocess = clf.preprocess_text
    processed = df["text"].astype(str).apply(preprocess).tolist()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), sublinear_tf=True)
    matrix = vectorizer.fit_transform(processed)

    # Outcome estimation from judgment text (optional best-effort using normalizer)
    outcomes = df["text"].astype(str).apply(clf.normalize_outcome).tolist()

    # Snippet: first 300 chars of text
    snippets = df["text"].astype(str).str.slice(0, 300).tolist()

    meta: List[Dict] = []
    for title, url, outcome, snippet in zip(df.get("title", [""]*len(df)), df.get("url", [""]*len(df)), outcomes, snippets):
        meta.append({
            "title": str(title) if pd.notna(title) else "",
            "url": str(url) if pd.notna(url) else "",
            "outcome": outcome,
            "snippet": snippet,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        dill.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "meta": meta,
        }, f)
    print(f"[index] Saved {len(meta)} cases to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TF-IDF search index for AI Court.")
    parser.add_argument("--input", dest="input_dir", default=None, help="Directory with *.csv files (defaults to data/raw_enriched or data/raw)")
    parser.add_argument("--out", dest="out_path", default=OUT_PATH, help="Output path for index pickle (default: models/search_index.pkl)")
    parser.add_argument("--max-features", dest="max_features", type=int, default=50000, help="Max features for TF-IDF (default: 50000)")
    parser.add_argument("--ngram-max", dest="ngram_max", type=int, default=2, choices=[1,2,3], help="Use up to this n-gram (default: 2)")
    args = parser.parse_args()

    df = load_cases(args.input_dir)
    build_index(df, out_path=args.out_path, max_features=args.max_features, ngram_max=args.ngram_max)

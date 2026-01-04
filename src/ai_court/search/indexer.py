import os
import glob
import dill
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import logging
from ai_court.model.legal_case_classifier import LegalCaseClassifier

logger = logging.getLogger(__name__)

def pick_first(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first existing column as a Series, else None."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

def load_cases(raw_dir: str, raw_enriched_dir: str, processed_path: str, input_dir: Optional[str] = None) -> pd.DataFrame:
    """Load cases from CSVs under input_dir (if provided) or default directories.
    Prefers data/raw_enriched when present; otherwise falls back to data/raw; finally to processed/all_cases.csv.
    """
    # Prefer enriched data if present
    if input_dir:
        search_dir = input_dir
    else:
        search_dir = raw_enriched_dir if os.path.isdir(raw_enriched_dir) and glob.glob(os.path.join(raw_enriched_dir, "*.csv")) else raw_dir
    
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
        if not os.path.exists(processed_path):
            raise RuntimeError(f"No raw CSVs found under {raw_dir} and no processed dataset at {processed_path}")
        pdf = pd.read_csv(processed_path)
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

def build_index(df: pd.DataFrame, out_path: str, max_features: int = 50000, ngram_max: int = 2) -> str:
    clf = LegalCaseClassifier()
    preprocess = clf.preprocess_text
    processed = df["text"].astype(str).apply(preprocess).tolist()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), sublinear_tf=True)
    matrix = vectorizer.fit_transform(processed)

    # Outcome estimation from judgment text (optional best-effort using normalizer)
    outcomes = df["text"].astype(str).apply(clf.normalize_outcome).tolist()
    
    # Metadata
    meta = []
    titles = df["title"].tolist()
    urls = df["url"].tolist()
    texts = df["text"].tolist()
    
    for i in range(len(df)):
        meta.append({
            "title": titles[i],
            "url": urls[i],
            "outcome": outcomes[i],
            "snippet": texts[i][:200] + "..."
        })
        
    index_data = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "meta": meta
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        dill.dump(index_data, f)
        
    return out_path

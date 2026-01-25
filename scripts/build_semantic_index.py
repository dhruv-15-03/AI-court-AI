"""Build a semantic (dense) vector search index using a SentenceTransformer model.

Writes models/semantic_index.pkl with keys:
  - model_name
  - embeddings (numpy array shape [N, D])
  - meta (list of {title,url,outcome,snippet})
  - dim

CLI:
  python scripts/build_semantic_index.py \
      --model all-MiniLM-L6-v2 \
      --batch-size 64 \
      --device auto

If sentence-transformers not available, exits with a clear message.
"""
from __future__ import annotations

import argparse
import os
import sys
import dill
import pandas as pd
from typing import List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.build_search_index import load_cases  # reuse existing loader # noqa: E402
from src.ai_court.model.legal_case_classifier import LegalCaseClassifier  # type: ignore # noqa: E402


def encode_texts(texts: List[str], model_name: str, batch_size: int, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit("sentence-transformers package required. Install via pip install sentence-transformers") from e
    model = SentenceTransformer(model_name, device=None if device == 'cpu' else device)
    # Model auto-selects device if not specified
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vectors, model


def build_semantic_index(df: pd.DataFrame, model_name: str, batch_size: int, device: str, out_path: str) -> str:
    clf = LegalCaseClassifier()
    preprocess = clf.preprocess_text
    processed = df['text'].astype(str).apply(preprocess).tolist()
    vectors, model = encode_texts(processed, model_name=model_name, batch_size=batch_size, device=device)
    # Minimal meta consistent with TF-IDF index
    outcomes = df['text'].astype(str).apply(clf.normalize_outcome).tolist()
    snippets = df['text'].astype(str).str.slice(0, 300).tolist()
    meta: List[Dict] = []
    for title, url, outcome, snippet in zip(df.get('title', ['']*len(df)), df.get('url', ['']*len(df)), outcomes, snippets):
        meta.append({
            'title': str(title) if title is not None else '',
            'url': str(url) if url is not None else '',
            'outcome': outcome,
            'snippet': snippet,
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        dill.dump({
            'model_name': model_name,
            'embeddings': vectors,
            'meta': meta,
            'dim': int(vectors.shape[1]),
        }, f)
    print(f"[semantic-index] Saved {vectors.shape[0]} embeddings of dim {vectors.shape[1]} to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Build semantic embedding index for AI Court search.')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformer model name (default: all-MiniLM-L6-v2)')
    parser.add_argument('--input', default=None, help='Optional input directory of CSVs (defaults to raw/enriched heuristic)')
    parser.add_argument('--batch-size', type=int, default=64, help='Encoding batch size (default: 64)')
    parser.add_argument('--device', default='auto', help='Device: auto|cpu|cuda (default: auto)')
    parser.add_argument('--out', default=os.path.join('models','semantic_index.pkl'), help='Output pickle path')
    args = parser.parse_args()
    df = load_cases(args.input)
    build_semantic_index(df, model_name=args.model, batch_size=args.batch_size, device=args.device, out_path=args.out)


if __name__ == '__main__':
    main()

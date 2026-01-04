"""
Build a simple TF-IDF search index over scraped cases with URLs and outcomes.
Defaults to data/raw or data/raw_enriched and writes models/search_index.pkl, but
supports CLI flags to customize paths and vectorizer size.
"""
import os
import sys
import argparse
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_court.search.indexer import load_cases, build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_search_index")

RAW_DIR = os.path.join("data", "raw")
RAW_ENRICHED_DIR = os.path.join("data", "raw_enriched")
PROCESSED_PATH = os.path.join("data", "processed", "all_cases.csv")
OUT_PATH = os.path.join("models", "search_index.pkl")

def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF search index")
    parser.add_argument("--input-dir", type=str, help="Directory containing raw CSVs")
    parser.add_argument("--out", type=str, default=OUT_PATH, help="Output path for pickle")
    parser.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF features")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max ngram size")
    args = parser.parse_args()

    try:
        logger.info("Loading cases...")
        df = load_cases(RAW_DIR, RAW_ENRICHED_DIR, PROCESSED_PATH, args.input_dir)
        logger.info(f"Loaded {len(df)} cases. Building index...")
        
        out_path = build_index(df, args.out, args.max_features, args.ngram_max)
        logger.info(f"Index saved to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

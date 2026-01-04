"""Harvest cases from Indian Kanoon for multiple queries.

Enhancements:
    - Supports query catalog JSON (data/query_catalog.json) if present.
    - Writes/updates a harvest manifest (data/harvest_manifest.json) with per-query stats.
    - Graceful resume semantics: if output CSV already exists and --resume set, skip re-harvest.
    - Aggregates into processed/all_cases.csv (best-effort) via existing prepare_dataset.

Limitations:
    - True page-level incremental resume depends on underlying scraper; current approach skips
        re-fetch if file exists (avoid heavy duplicate network calls). For deeper historical backfill
        run with increasing KANOON_PAGES after removing a specific per-query CSV.
"""

import csv
import os
import sys
import time
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("harvest.log")
    ]
)
logger = logging.getLogger("harvest")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ai_court.scraper.kanoon import create_dataset  # noqa: E402

DEFAULT_QUERIES = [
    # Criminal
    "rape",
    "murder",
    "attempt to murder",
    "narcotics NDPS",
    "bail granted",
    "bail denied",
    # Civil / property / contracts
    "property dispute",
    "contract dispute",
    "negligence damages",
    # Family
    "divorce custody",
    # Labor
    "labour wages termination",
]

OUT_DIR = os.path.join("data", "raw")
AGG_CSV = os.path.join("data", "processed", "all_cases.csv")
CATALOG_JSON = os.path.join("data", "query_catalog.json")
MANIFEST_PATH = os.path.join("data", "harvest_manifest.json")


def load_query_catalog() -> list[str]:
    if os.path.exists(CATALOG_JSON):
        try:
            with open(CATALOG_JSON,'r',encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'queries' in data:
                q = data['queries']
            else:
                q = data
            q = [x for x in q if isinstance(x,str) and x.strip()]
            if q:
                logger.info(f"[catalog] Loaded {len(q)} queries from query_catalog.json")
                return q
        except Exception as ce:
            logger.error(f"[catalog] Failed to load catalog: {ce}")
    return DEFAULT_QUERIES


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH,'r',encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}


def save_manifest(m: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp = {k: v for k,v in m.items()}
    with open(MANIFEST_PATH,'w',encoding='utf-8') as f:
        json.dump(tmp, f, indent=2)


def harvest_queries(queries, pages_per_query=3, resume=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_manifest()
    run_ts = datetime.utcnow().isoformat()
    harvested = []
    for q in queries:
        safe = "_".join(q.split())[:40]
        out = os.path.join(OUT_DIR, f"kanoon_{safe}.csv")
        if resume and os.path.exists(out):
            logger.info(f"[harvest] SKIP (resume) Query='{q}' existing={out}")
            harvested.append(out)
            continue
        logger.info(f"[harvest] Query='{q}' pages={pages_per_query} -> {out}")
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                n = create_dataset(q, pages_per_query, out)
                if n > 0:
                    harvested.append(out)
                    manifest[q] = {
                        'file': out,
                        'pages_requested': pages_per_query,
                        'rows_written': n,
                        'last_run_utc': run_ts,
                        'status': 'success'
                    }
                    save_manifest(manifest)
                    break
                else:
                    logger.warning(f"Attempt {attempt+1}: Output file empty or missing for query '{q}'")
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for query '{q}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    manifest[q] = {
                        "last_run": run_ts,
                        "status": "failed",
                        "error": str(e)
                    }
                    save_manifest(manifest)
        
        time.sleep(1.5)  # Polite delay between queries

    if harvested:
        # Update high-level summary
        manifest['_summary'] = manifest.get('_summary', {})
        summary = manifest['_summary']
        summary['last_run_utc'] = run_ts
        summary['num_queries'] = len([k for k in manifest.keys() if k not in {'_summary'}])
        save_manifest(manifest)
    return harvested


if __name__ == "__main__":
    pages = int(os.environ.get("KANOON_PAGES", "3"))
    qfile = os.environ.get("KANOON_QUERIES_FILE")
    queries = DEFAULT_QUERIES
    if qfile and os.path.exists(qfile):
        with open(qfile, newline="", encoding="utf-8") as f:
            queries = [row[0] for row in csv.reader(f) if row]
            logger.info(f"[harvest] Loaded {len(queries)} queries from {qfile}")

    resume = os.environ.get('KANOON_RESUME','0') == '1'
    queries = load_query_catalog() if not qfile else queries
    harvested = harvest_queries(queries, pages_per_query=pages, resume=resume)
    logger.info(f"[harvest] Completed: {len(harvested)} CSVs")

    # Optionally aggregate using prepare_dataset (if their schema matches heuristics)
    try:
        from src.ai_court.data.prepare_dataset import build_processed_dataset

        agg_path = build_processed_dataset(harvested, out_csv=AGG_CSV)
        logger.info(f"[harvest] Aggregated dataset -> {agg_path}")
    except Exception as e:
        logger.error(f"[harvest] Aggregation skipped/failed: {e}")

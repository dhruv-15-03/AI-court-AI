"""
Enrich harvested case summaries using Hugging Face summarization.

Inputs: data/raw/kanoon_*.csv with columns [id,title,url,case_summary,judgment]
Outputs: data/raw_enriched/kanoon_*.csv with same columns but improved case_summary

Features:
- Loads .env for HUGGINGFACE_API_TOKEN and optional HUGGINGFACE_API_URL
- Resumable: if output exists, skips rows already processed (matched by URL)
- Safe rate limiting via small sleeps; configurable via HF_RATE_LIMIT_SLEEP env (seconds)
- Atomic writes and periodic checkpoints to avoid data loss on interruption
"""
from __future__ import annotations

import os
import glob
import time
import logging
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv


RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "raw_enriched")
LOG_DIR = os.path.join("logs")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "enrich_hf.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_env():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    url = os.getenv(
        "HUGGINGFACE_API_URL",
        "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
    )
    sleep_s = float(os.getenv("HF_RATE_LIMIT_SLEEP", "0.3"))
    return token, url, sleep_s


def hf_summarize(text: str, token: str, url: str, max_chars: int = 1500) -> str | None:
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    max_input_length = 1024
    t = (text or "")[:max_input_length]
    payload = {
        "inputs": t,
        "parameters": {"max_length": 220, "min_length": 60, "do_sample": False},
        "options": {"wait_for_model": True},
    }
    for attempt in range(2):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    return None
                if isinstance(j, list) and j:
                    if isinstance(j[0], dict) and "summary_text" in j[0]:
                        return j[0]["summary_text"]
                    return str(j[0])
                if isinstance(j, dict) and "summary_text" in j:
                    return j["summary_text"]
                return str(j)
            else:
                logger.warning(f"HF non-200 {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"HF error attempt {attempt+1}: {e}")
        time.sleep(1.0)
    return None


def enrich_file(path: str, out_path: str, token: str, url: str, sleep_s: float) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(path)
    required = ["id", "title", "url", "case_summary", "judgment"]
    if not all(c in df.columns for c in required):
        logger.warning(f"Skipping {os.path.basename(path)}: missing required columns")
        return 0

    existing_urls: set[str] = set()
    rows: List[Dict] = []
    if os.path.exists(out_path):
        try:
            prev = pd.read_csv(out_path)
            if all(c in prev.columns for c in required):
                rows = prev[required].to_dict(orient='records')
                existing_urls = set(prev['url'].dropna().astype(str))
                logger.info(f"Resuming {os.path.basename(path)}: {len(rows)} rows already enriched")
        except Exception as e:
            logger.warning(f"Could not resume {out_path}: {e}")

    processed = len(rows)
    total = len(df)
    try:
        for _, rec in df.iterrows():
            url_val = str(rec.get('url', ''))
            if url_val in existing_urls:
                continue
            original_summary = str(rec.get('case_summary', '') or '')
            judgment = str(rec.get('judgment', '') or '')
            # Prefer summarizing judgment; fallback to existing summary
            source_text = judgment if len(judgment.strip()) >= 200 else original_summary
            enriched = hf_summarize(source_text, token, url)
            new_summary = enriched if enriched and len(enriched.strip()) > 0 else original_summary

            rows.append({
                'id': rec.get('id'),
                'title': rec.get('title'),
                'url': url_val,
                'case_summary': new_summary,
                'judgment': judgment,
            })
            existing_urls.add(url_val)
            processed += 1

            # Periodic checkpoint every 20 rows
            if processed % 20 == 0 or processed == total:
                out_df = pd.DataFrame(rows, columns=required)
                tmp = out_path + ".tmp"
                out_df.to_csv(tmp, index=False)
                os.replace(tmp, out_path)
                logger.info(f"Checkpoint saved {processed}/{total} -> {out_path}")

            time.sleep(sleep_s)
    except KeyboardInterrupt:
        # Gracefully save current progress before bubbling up
        try:
            out_df = pd.DataFrame(rows, columns=required)
            tmp = out_path + ".tmp"
            out_df.to_csv(tmp, index=False)
            os.replace(tmp, out_path)
            logger.info(f"Interrupted. Final checkpoint saved {processed}/{total} -> {out_path}")
        except Exception as e:
            logger.warning(f"Interrupted and failed to save checkpoint for {out_path}: {e}")
        raise
    except Exception:
        # Save partial progress on unexpected errors and re-raise for caller logging
        try:
            out_df = pd.DataFrame(rows, columns=required)
            tmp = out_path + ".tmp"
            out_df.to_csv(tmp, index=False)
            os.replace(tmp, out_path)
            logger.info(f"Error encountered. Partial checkpoint saved {processed}/{total} -> {out_path}")
        except Exception as e2:
            logger.warning(f"Failed to save partial checkpoint for {out_path}: {e2}")
        raise

    # Final write (ensure order and columns)
    out_df = pd.DataFrame(rows, columns=required)
    tmp = out_path + ".tmp"
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    logger.info(f"Enriched saved {len(rows)} rows -> {out_path}")
    return len(rows)


def main():
    token, url, sleep_s = load_env()
    logger.info(f"Using HF model: {url}")
    paths = glob.glob(os.path.join(RAW_DIR, "kanoon_*.csv"))
    if not paths:
        logger.error("No input CSVs found under data/raw")
        return 1
    total_rows = 0
    for p in sorted(paths):
        base = os.path.basename(p)
        out_p = os.path.join(OUT_DIR, base)
        logger.info(f"Enriching {base} -> {out_p}")
        try:
            n = enrich_file(p, out_p, token, url, sleep_s)
            total_rows += n
        except Exception as e:
            logger.error(f"Error enriching {base}: {e}")
        time.sleep(0.2)
    logger.info(f"Completed enrichment. Total rows: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

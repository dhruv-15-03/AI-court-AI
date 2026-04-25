"""Refresh the AI corpus with newly-published case law.

Incrementally harvests new Kanoon case pages for a set of queries, skips any
URL already present in the existing vector store, then embeds + appends the
new documents to the semantic index.

Usage:
    python scripts/refresh_corpus.py \\
        --queries "latest Supreme Court 2025" "bail application 2025" \\
        --pages 2 --index data/embeddings/cases.npz

If --queries is omitted, falls back to ``data/queries.example.csv``.
Designed to be run on a schedule (cron / Task Scheduler / GitHub Action).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_court.retrieval.vector_store import VectorStore  # noqa: E402
from ai_court.scraper.kanoon import get_case_links, get_case_content  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("refresh_corpus")


def load_queries(queries: List[str] | None, csv_path: Path) -> List[str]:
    if queries:
        return queries
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    return ["latest Supreme Court 2025", "bail application 2025", "High Court 2025"]


def existing_urls(store: VectorStore) -> set[str]:
    urls: set[str] = set()
    for m in store.metadata:
        url = m.get("url") or m.get("source_url")
        if url:
            urls.add(url)
    return urls


def refresh(index_path: Path, queries: List[str], pages: int) -> dict:
    if index_path.exists():
        store = VectorStore.load(str(index_path))
        logger.info("Loaded existing store: %d docs", len(store))
    else:
        logger.warning("No existing index at %s — building from scratch", index_path)
        store = VectorStore.build([""], [{}])  # tiny placeholder, will be replaced
        store._texts.clear()
        store._metadata.clear()
        import numpy as np
        store._embeddings = store._embeddings[:0]
        store._faiss_index = None

    seen = existing_urls(store)
    logger.info("Known URLs: %d", len(seen))

    new_texts: list[str] = []
    new_meta: list[dict] = []
    for q in queries:
        logger.info("Harvesting query=%r pages=%d", q, pages)
        try:
            links = get_case_links(q, pages=pages) or []
        except Exception as exc:
            logger.warning("Search failed for %r: %s", q, exc)
            continue
        for link in links:
            url = link if isinstance(link, str) else link.get("url")
            if not url or url in seen:
                continue
            try:
                case = get_case_content(url)
            except Exception as exc:
                logger.warning("Fetch failed %s: %s", url, exc)
                continue
            if not case or not case.get("text"):
                continue
            text = case["text"].strip()
            if len(text) < 400:  # skip stubs
                continue
            new_texts.append(text[:20000])  # cap doc size
            new_meta.append({
                "url": url,
                "title": case.get("title", ""),
                "query": q,
                "source": "kanoon",
            })
            seen.add(url)

    added = 0
    if new_texts:
        added = store.add(new_texts, new_meta)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(str(index_path))
        logger.info("Saved updated index (%d total docs)", len(store))
    else:
        logger.info("No new documents to add.")

    return {
        "added": added,
        "total_docs": len(store),
        "queries_run": len(queries),
        "index_path": str(index_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", nargs="*", default=None)
    p.add_argument("--queries-csv", default=str(ROOT / "data" / "queries.example.csv"))
    p.add_argument("--pages", type=int, default=2)
    p.add_argument("--index", default=str(ROOT / "data" / "embeddings" / "cases.npz"))
    args = p.parse_args()

    queries = load_queries(args.queries, Path(args.queries_csv))
    result = refresh(Path(args.index), queries, args.pages)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

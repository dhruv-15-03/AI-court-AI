import os
import logging
from typing import Optional

import pandas as pd

# Import scraper functions from src package
from ..scraper.kanoon import (
    get_case_links,
    get_case_content,
    extract_judgment_section,
    get_case_summary,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def scrape_kanoon(query: str, pages: int = 2, out_csv: Optional[str] = None) -> str:
    """Scrape Indian Kanoon search results and save raw dataset.

    Returns path to the saved CSV under data/raw.
    """
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = out_csv or os.path.join(raw_dir, f"kanoon_{query.replace(' ', '_')}_{pages}p.csv")

    links = get_case_links(query, pages)
    if not links:
        raise RuntimeError("No links found from Indian Kanoon")

    rows = []
    for i, link in enumerate(links, start=1):
        info = get_case_content(link["url"]) if isinstance(link, dict) else None
        if not info:
            continue
        text = info.get("text", "")
        judgment = extract_judgment_section(text)
        summary = get_case_summary(text)
        rows.append({
            "id": i,
            "title": info.get("title", "Unknown"),
            "url": info.get("url", link.get("url") if isinstance(link, dict) else ""),
            "case_summary": summary,
            "judgement": judgment,
            "case_data": text,
            "case_type": "Unknown"
        })
        if i % 5 == 0:
            logger.info("Scraped %d/%d", i, len(links))

    if not rows:
        raise RuntimeError("No rows scraped from Indian Kanoon")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved raw scraped cases: %s", out_csv)
    return out_csv


if __name__ == "__main__":
    # Simple CLI usage
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="cases on rape and murder")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    path = scrape_kanoon(args.query, args.pages, args.out)
    print(path)

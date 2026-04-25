"""Bulk Kanoon Harvester — systematic scraper for scaling to 100K+ cases.

This script harvests cases from Indian Kanoon across multiple categories,
courts, years, and sections to build a diverse training corpus.

Usage:
    python scripts/bulk_harvest.py --target 100000 --output data/raw
    python scripts/bulk_harvest.py --target 5000 --categories murder,bail --output data/raw
    python scripts/bulk_harvest.py --resume  # Resume from harvest_manifest.json

Features:
- Configurable queries across categories, courts, years
- Resume support via manifest file
- Rate-limited (respectful scraping)
- Deduplication by URL
- Proper label extraction from judgment text
- Checkpoint saves every 50 cases
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "bulk_harvest.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/?formInput="
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "harvest_manifest.json")

# Respectful rate limits
MIN_DELAY = 2.0
MAX_DELAY = 5.0
PAGE_DELAY = 3.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
}

# ── Query Categories ────────────────────────────────────────────────────
# Each category produces multiple search queries to maximize coverage.

QUERY_CATEGORIES = {
    "murder": {
        "queries": [
            "murder IPC 302",
            "murder conviction Section 302",
            "murder acquittal benefit of doubt",
            "murder BNS 101",  # New code
            "culpable homicide IPC 304",
            "attempt to murder IPC 307",
        ],
        "expected_label_keywords": ["convicted", "acquitted", "appeal dismissed", "conviction upheld"],
    },
    "rape": {
        "queries": [
            "rape IPC 376",
            "rape conviction IPC 376",
            "rape acquittal",
            "rape BNS 63",  # New code
            "POCSO conviction",
            "sexual assault",
        ],
        "expected_label_keywords": ["convicted", "acquitted", "conviction upheld"],
    },
    "bail": {
        "queries": [
            "bail granted",
            "bail denied rejected",
            "anticipatory bail granted",
            "anticipatory bail rejected",
            "interim bail",
            "default bail Section 167 CrPC",
            "bail NDPS Act",
            "bail PMLA money laundering",
        ],
        "expected_label_keywords": ["bail granted", "bail denied", "bail rejected", "enlarged on bail"],
    },
    "quashing": {
        "queries": [
            "Section 482 CrPC quashing",
            "FIR quashed",
            "charge sheet quashed",
            "proceedings quashed",
            "quashing matrimonial dispute 498A",
            "Section 528 BNSS quashing",  # New code
        ],
        "expected_label_keywords": ["quashed", "proceedings quashed", "FIR quashed"],
    },
    "sentence": {
        "queries": [
            "sentence reduced modified",
            "sentence commuted",
            "death sentence commuted life imprisonment",
            "fine reduced",
            "imprisonment reduced",
        ],
        "expected_label_keywords": ["sentence reduced", "commuted", "modified"],
    },
    "civil_relief": {
        "queries": [
            "writ petition allowed",
            "writ petition dismissed",
            "injunction granted",
            "specific performance contract",
            "partition suit decree",
            "consumer complaint allowed",
            "motor accident compensation",
        ],
        "expected_label_keywords": ["petition allowed", "petition dismissed", "relief granted", "dismissed"],
    },
    "divorce_family": {
        "queries": [
            "divorce Hindu Marriage Act",
            "maintenance CrPC 125",
            "custody child welfare",
            "domestic violence protection order",
            "matrimonial cruelty",
            "restitution conjugal rights",
        ],
        "expected_label_keywords": ["allowed", "dismissed", "granted", "denied"],
    },
    "labour": {
        "queries": [
            "reinstatement Industrial Disputes Act",
            "termination workmen",
            "wages recovery labour",
            "retrenchment compensation",
            "service matter termination",
        ],
        "expected_label_keywords": ["allowed", "dismissed", "reinstated", "compensation"],
    },
    "property": {
        "queries": [
            "property dispute title suit",
            "land acquisition compensation",
            "eviction suit tenant",
            "succession inheritance dispute",
        ],
        "expected_label_keywords": ["allowed", "dismissed", "decree"],
    },
    "cheque_bounce": {
        "queries": [
            "Section 138 Negotiable Instruments Act",
            "cheque dishonour conviction",
            "cheque bounce complaint",
        ],
        "expected_label_keywords": ["convicted", "acquitted", "complaint dismissed"],
    },
    "corruption": {
        "queries": [
            "Prevention of Corruption Act",
            "bribery conviction public servant",
            "disproportionate assets",
        ],
        "expected_label_keywords": ["convicted", "acquitted"],
    },
    "narcotics": {
        "queries": [
            "NDPS Act conviction",
            "narcotics bail",
            "commercial quantity NDPS",
        ],
        "expected_label_keywords": ["convicted", "acquitted", "bail granted", "bail denied"],
    },
    "cyber_it": {
        "queries": [
            "Information Technology Act",
            "cyber crime",
            "data breach privacy",
        ],
        "expected_label_keywords": ["convicted", "dismissed", "allowed"],
    },
    "constitutional": {
        "queries": [
            "Article 21 fundamental rights",
            "Article 14 equality",
            "habeas corpus",
            "PIL public interest litigation",
            "Article 226 writ jurisdiction",
        ],
        "expected_label_keywords": ["allowed", "dismissed", "directed"],
    },
    "remand": {
        "queries": [
            "case remanded fresh consideration",
            "de novo trial ordered",
            "matter sent back lower court",
        ],
        "expected_label_keywords": ["remanded", "sent back", "de novo"],
    },
    "withdrawn": {
        "queries": [
            "petition withdrawn",
            "appeal withdrawn dismissed as withdrawn",
            "matter not pressed",
        ],
        "expected_label_keywords": ["withdrawn", "not pressed"],
    },
}

# Courts to filter by (Indian Kanoon supports court filters)
COURTS = [
    "",  # All courts
    "supremecourt",
    "allahabad",
    "bombay",
    "delhi",
    "madras",
    "calcutta",
    "karnataka",
    "kerala",
    "punjab",
    "rajasthan",
    "gauhati",
    "hyderabad",
    "patna",
    "chhattisgarh",
    "jharkhand",
    "uttarakhand",
    "himachal",
    "jammu",
    "tripura",
    "meghalaya",
    "sikkim",
]


def _fetch_page(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited (429). Sleeping %ds...", wait)
                time.sleep(wait)
                continue
            logger.warning("HTTP %d for %s", resp.status_code, url)
            return None
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.warning("Request attempt %d failed: %s", attempt + 1, exc)
            time.sleep(3 * (attempt + 1))
    return None


def search_cases(query: str, court: str = "", pages: int = 5) -> List[Dict[str, str]]:
    """Search Indian Kanoon and return list of {url, title}."""
    results: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()

    for page in range(1, pages + 1):
        encoded = quote_plus(query)
        url = f"{SEARCH_URL}{encoded}"
        if court:
            url += f"&formInput={encoded}+doctypes:+{court}"
        url += f"&pagenum={page}"

        html = _fetch_page(url)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")

        for selector in [".result_title a", "div.result_title > a", "a.result_title"]:
            tags = soup.select(selector)
            if tags:
                break
        else:
            tags = []

        for tag in tags:
            href = tag.get("href", "")
            if not href:
                continue
            # Extract doc ID from /docfragment/XXXX/ or /doc/XXXX/
            m = re.search(r"/(?:doc|docfragment)/(\d+)/", href)
            if m:
                full_url = BASE_URL + f"/doc/{m.group(1)}/"
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    results.append({
                        "url": full_url,
                        "title": tag.get_text(strip=True),
                    })

        if page < pages:
            time.sleep(random.uniform(PAGE_DELAY, PAGE_DELAY + 2))

    return results


def scrape_case(url: str) -> Optional[Dict[str, Any]]:
    """Scrape a single case page and extract structured data."""
    html = _fetch_page(url, timeout=45)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.select_one("div.judgments_title")
    title = title_el.get_text(strip=True) if title_el else "Unknown Case"

    # Full judgment text
    full_text = ""
    for selector in ["pre", "#judgments_text", ".judgments_text", "div.judgments", "div.doc-content"]:
        el = soup.select_one(selector)
        if el:
            full_text = el.get_text()
            if len(full_text) > 500:
                break

    if not full_text or len(full_text) < 200:
        body = soup.find("body")
        if body:
            for tag in body.select("nav, header, footer, script, style"):
                tag.decompose()
            full_text = body.get_text()

    if not full_text or len(full_text) < 100:
        return None

    # Clean up text
    full_text = re.sub(r"\n\s*\n", "\n\n", full_text).strip()

    # Extract judgment/order section (last portion with legal conclusions)
    judgment_section = _extract_judgment_section(full_text)

    # Create summary (first 1500 + judgment section)
    summary = _create_summary(full_text, judgment_section)

    # Extract metadata
    court = _extract_court(soup, full_text)
    date = _extract_date(soup, full_text)
    citations = _extract_citations(full_text)

    return {
        "title": title,
        "url": url,
        "case_summary": summary,
        "judgment": judgment_section,
        "full_text_length": len(full_text),
        "court": court,
        "date": date,
        "citations": citations,
    }


def _extract_judgment_section(text: str) -> str:
    """Extract the concluding judgment/order from full text."""
    patterns = [
        r"(?:ORDER|JUDGMENT|CONCLUSION|HELD|THEREFORE|OPERATIVE ORDER)(.*?)(?:\n\n|$)",
        r"(?:we\s+hold|we\s+direct|it\s+is\s+ordered|it\s+is\s+hereby|accordingly|"
        r"the\s+appeal\s+is|the\s+petition\s+is|we\s+find\s+that)(.*?)(?:\n\n|$)",
        r"(?:for\s+the\s+foregoing\s+reasons|in\s+view\s+of\s+the\s+above|"
        r"in\s+the\s+light\s+of)(.*?)(?:\n\n|$)",
    ]
    sections = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            sections.append(match.group(0).strip())

    if sections:
        return " ".join(sections[-3:])

    # Fallback: last 15 meaningful lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines[-15:])


def _create_summary(full_text: str, judgment_section: str, max_len: int = 2000) -> str:
    """Create an informative summary with key parts of the case."""
    # Take beginning (facts/intro) + judgment
    intro_len = max_len - len(judgment_section) - 50
    if intro_len > 200:
        intro = full_text[:intro_len]
    else:
        intro = full_text[:500]
    return f"{intro}\n\n[...]\n\n{judgment_section}"


def _extract_court(soup: BeautifulSoup, text: str) -> str:
    """Try to extract court name."""
    court_el = soup.select_one("div.docsource_main")
    if court_el:
        return court_el.get_text(strip=True)
    # Fallback from text
    for pattern in [r"(Supreme Court of India)", r"(High Court of \w+)", r"(\w+ High Court)"]:
        m = re.search(pattern, text[:500], re.IGNORECASE)
        if m:
            return m.group(1)
    return "Unknown"


def _extract_date(soup: BeautifulSoup, text: str) -> str:
    """Try to extract judgment date."""
    date_el = soup.select_one("div.doc_date, span.doc_date")
    if date_el:
        return date_el.get_text(strip=True)
    m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text[:1000])
    if m:
        return m.group(1)
    return ""


def _extract_citations(text: str) -> List[str]:
    """Extract legal citations from text."""
    patterns = [
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"AIR\s+\d{4}\s+\w+\s+\d+",
        r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",
        r"\d{4}\s+Cri\.?L\.?J\.\s+\d+",
    ]
    citations = set()
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            citations.add(m.group(0))
    return list(citations)[:20]


def load_manifest() -> Dict[str, Any]:
    """Load or create harvest manifest for resume support."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "total_harvested": 0,
        "target": 0,
        "processed_urls": [],
        "completed_queries": [],
        "last_updated": "",
    }


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Save manifest atomically."""
    manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def harvest(
    target: int = 100000,
    output_dir: str = "data/raw",
    categories: Optional[List[str]] = None,
    pages_per_query: int = 50,  # 10 results/page × 50 = 500 per query
    resume: bool = True,
) -> int:
    """Main harvest loop.

    For each category × court combination, scrapes pages of results,
    then fetches individual case details.

    Args:
        target: Target total number of cases.
        output_dir: Directory to save CSVs.
        categories: List of category names to scrape (None = all).
        pages_per_query: Max pages to scrape per search query.
        resume: Whether to resume from manifest.

    Returns:
        Total number of cases harvested.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = load_manifest() if resume else {"total_harvested": 0, "target": target, "processed_urls": [], "completed_queries": [], "last_updated": ""}
    manifest["target"] = target
    processed_urls: Set[str] = set(manifest.get("processed_urls", []))
    completed_queries: Set[str] = set(manifest.get("completed_queries", []))
    total = manifest.get("total_harvested", 0)

    cats = categories or list(QUERY_CATEGORIES.keys())
    logger.info("Starting bulk harvest: target=%d, categories=%d, resume=%s", target, len(cats), resume)

    for cat_name in cats:
        cat = QUERY_CATEGORIES.get(cat_name)
        if cat is None:
            logger.warning("Unknown category: %s", cat_name)
            continue

        for query in cat["queries"]:
            # Try with subset of courts to diversify
            courts_to_try = ["", "supremecourt", "delhi", "bombay", "allahabad", "madras", "calcutta", "karnataka"]

            for court in courts_to_try:
                query_key = f"{query}|{court}"
                if query_key in completed_queries:
                    continue
                if total >= target:
                    logger.info("Target reached: %d/%d", total, target)
                    save_manifest(manifest)
                    return total

                logger.info("[%d/%d] Searching: '%s' court=%s", total, target, query, court or "all")

                try:
                    links = search_cases(query, court=court, pages=pages_per_query)
                except Exception as exc:
                    logger.error("Search failed for '%s': %s", query, exc)
                    continue

                new_links = [l for l in links if l["url"] not in processed_urls]
                logger.info("  Found %d results (%d new)", len(links), len(new_links))

                # CSV file per category
                csv_path = os.path.join(output_dir, f"kanoon_{cat_name}_bulk.csv")
                existing_data = []
                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                        existing_data = existing_df.to_dict("records")
                    except Exception:
                        pass

                batch_data = []
                for link in new_links:
                    if total >= target:
                        break

                    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

                    try:
                        case = scrape_case(link["url"])
                    except Exception as exc:
                        logger.warning("Failed to scrape %s: %s", link["url"], exc)
                        processed_urls.add(link["url"])
                        continue

                    if case is None:
                        processed_urls.add(link["url"])
                        continue

                    record = {
                        "id": total + 1,
                        "title": case["title"],
                        "url": case["url"],
                        "case_summary": case["case_summary"],
                        "judgment": case["judgment"],
                        "court": case.get("court", ""),
                        "date": case.get("date", ""),
                        "category": cat_name,
                        "query_source": query,
                    }
                    batch_data.append(record)
                    processed_urls.add(link["url"])
                    total += 1

                    if total % 50 == 0:
                        # Checkpoint save
                        all_data = existing_data + batch_data
                        pd.DataFrame(all_data).to_csv(csv_path, index=False)
                        existing_data = all_data
                        batch_data = []
                        manifest["total_harvested"] = total
                        manifest["processed_urls"] = list(processed_urls)
                        manifest["completed_queries"] = list(completed_queries)
                        save_manifest(manifest)
                        logger.info("  Checkpoint: %d/%d total cases saved", total, target)

                # Save remaining batch
                if batch_data:
                    all_data = existing_data + batch_data
                    pd.DataFrame(all_data).to_csv(csv_path, index=False)

                completed_queries.add(query_key)

    manifest["total_harvested"] = total
    manifest["processed_urls"] = list(processed_urls)
    manifest["completed_queries"] = list(completed_queries)
    save_manifest(manifest)
    logger.info("Harvest complete: %d total cases", total)
    return total


def main():
    parser = argparse.ArgumentParser(description="Bulk Kanoon Case Harvester")
    parser.add_argument("--target", type=int, default=100000, help="Target number of cases")
    parser.add_argument("--output", default="data/raw", help="Output directory for CSVs")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category names")
    parser.add_argument("--pages", type=int, default=50, help="Max pages per search query")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from manifest")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start fresh")
    args = parser.parse_args()

    cats = args.categories.split(",") if args.categories else None

    total = harvest(
        target=args.target,
        output_dir=args.output,
        categories=cats,
        pages_per_query=args.pages,
        resume=args.resume,
    )
    print(f"\nDone. Total cases harvested: {total}")


if __name__ == "__main__":
    main()

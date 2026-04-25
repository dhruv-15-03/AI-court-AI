"""
Grand Harvest Orchestrator — Multi-source pipeline to reach 1,00,000 cases
==========================================================================

Strategy:
  Source 1: Indian Kanoon web scraping (96 queries × courts × pages)  → ~60K
  Source 2: ILDC  (Indian Legal Document Corpus, ~35K, CC BY-NC)      → ~35K
  Source 3: HLDC  (Hindi Legal Document Corpus, bail, ~5K)            → ~5K
                                                       Total target  → 100K+

Usage:
    python scripts/harvest_100k.py                      # Full pipeline (all sources)
    python scripts/harvest_100k.py --kanoon-only        # Only web scraping
    python scripts/harvest_100k.py --external-only      # Only ILDC+HLDC
    python scripts/harvest_100k.py --status             # Check progress
    python scripts/harvest_100k.py --resume             # Resume interrupted run
    python scripts/harvest_100k.py --workers 4          # Parallel court scraping
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "harvest_100k.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("harvest_100k")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ENRICHED_DIR = PROJECT_ROOT / "data" / "raw_enriched"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "harvest_manifest.json"
QUERY_CATALOG = PROJECT_ROOT / "data" / "query_catalog.json"
MASTER_CSV = PROCESSED_DIR / "all_cases_master.csv"

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/?formInput="

MIN_DELAY = 2.0
MAX_DELAY = 4.0
PAGE_DELAY = 2.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# Courts for diversification
COURTS = [
    "",            # all courts
    "supremecourt",
    "delhi",
    "bombay",
    "allahabad",
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

# Query → category mapping for labeling
QUERY_CATEGORY_MAP = {
    "murder": "criminal_murder",
    "IPC 302": "criminal_murder",
    "IPC 304": "criminal_homicide",
    "IPC 307": "criminal_attempt_murder",
    "rape": "criminal_rape",
    "IPC 376": "criminal_rape",
    "POCSO": "criminal_pocso",
    "kidnapping": "criminal_kidnapping",
    "robbery": "criminal_robbery",
    "dacoity": "criminal_dacoity",
    "cheating": "criminal_cheating",
    "IPC 420": "criminal_cheating",
    "forgery": "criminal_forgery",
    "dowry": "criminal_dowry",
    "IPC 498A": "criminal_cruelty",
    "IPC 406": "criminal_breach_trust",
    "IPC 325": "criminal_assault",
    "NDPS": "narcotics",
    "narcotics": "narcotics",
    "Arms Act": "criminal_arms",
    "Corruption Act": "corruption",
    "PMLA": "money_laundering",
    "bail": "bail",
    "anticipatory bail": "bail",
    "acquittal": "criminal_acquittal",
    "conviction upheld": "criminal_conviction",
    "conviction reversed": "criminal_acquittal",
    "sentence reduced": "sentencing",
    "sentence enhanced": "sentencing",
    "FIR quashed": "quashing",
    "quashed": "quashing",
    "482 CrPC": "quashing",
    "specific performance": "civil_contract",
    "injunction": "civil_injunction",
    "negligence": "civil_negligence",
    "defamation": "civil_defamation",
    "recovery of money": "civil_recovery",
    "partition suit": "civil_property",
    "declaratory decree": "civil_property",
    "eviction": "civil_property",
    "easement": "civil_property",
    "divorce": "family_divorce",
    "custody": "family_custody",
    "maintenance": "family_maintenance",
    "guardianship": "family_custody",
    "domestic violence": "family_dv",
    "DV Act": "family_dv",
    "talaq": "family_divorce",
    "termination": "labour",
    "retrenchment": "labour",
    "dismissal": "labour",
    "reinstatement": "labour",
    "wages": "labour",
    "contract labour": "labour",
    "service matter": "service",
    "land acquisition": "property",
    "adverse possession": "property",
    "mortgage": "property",
    "benami": "property",
    "habeas corpus": "constitutional",
    "mandamus": "constitutional",
    "certiorari": "constitutional",
    "PIL": "constitutional",
    "fundamental rights": "constitutional",
    "Article 21": "constitutional",
    "election petition": "election",
    "RTI": "administrative",
    "arbitration": "commercial",
    "winding up": "commercial",
    "NCLT": "commercial",
    "IBC": "commercial",
    "consumer": "consumer",
}


def _categorize_query(query: str) -> str:
    """Map a search query to a category."""
    for keyword, category in QUERY_CATEGORY_MAP.items():
        if keyword.lower() in query.lower():
            return category
    return "general"


def _text_hash(text: str) -> str:
    """Quick dedup hash on first 500 chars."""
    normalized = re.sub(r"\s+", " ", text[:500].lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
            # Ensure all keys exist
            data.setdefault("total_harvested", 0)
            data.setdefault("processed_urls", [])
            data.setdefault("completed_queries", [])
            data.setdefault("dedup_hashes", [])
            data.setdefault("source_counts", {})
            data.setdefault("category_counts", {})
            return data
    return {
        "total_harvested": 0,
        "target": 100000,
        "processed_urls": [],
        "completed_queries": [],
        "dedup_hashes": [],
        "source_counts": {"kanoon_scrape": 0, "ildc": 0, "hldc": 0, "existing": 0},
        "category_counts": {},
        "last_updated": "",
    }


def save_manifest(manifest: Dict[str, Any]):
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
    tmp = str(MANIFEST_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, str(MANIFEST_PATH))


# ── Web Scraping ──────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 30) -> Optional[str]:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                wait = 15 * (attempt + 1) + random.uniform(0, 5)
                logger.warning("Rate-limited (429). Sleeping %.0fs...", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                logger.warning("Blocked (403). Rotating delay...")
                time.sleep(30 + random.uniform(0, 15))
                continue
            logger.warning("HTTP %d: %s", resp.status_code, url[:100])
            return None
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, str(e)[:100])
            time.sleep(5 * (attempt + 1))
    return None


def search_kanoon(query: str, court: str = "", page: int = 1) -> List[Dict[str, str]]:
    """Search one page of results on Indian Kanoon."""
    encoded = quote_plus(query)
    url = f"{SEARCH_URL}{encoded}"
    if court:
        url += f"+doctypes:+{court}"
    url += f"&pagenum={page}"

    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

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
        # Extract doc ID from /docfragment/XXXX/ or /doc/XXXX/ URLs
        m = re.search(r"/(?:doc|docfragment)/(\d+)/", href)
        if m:
            doc_url = f"{BASE_URL}/doc/{m.group(1)}/"
            results.append({
                "url": doc_url,
                "title": tag.get_text(strip=True),
            })

    return results


def scrape_case_detail(url: str) -> Optional[Dict[str, Any]]:
    """Scrape a full case page."""
    html = _fetch(url, timeout=45)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.select_one("div.judgments_title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Court
    court_el = soup.select_one("div.docsource_main")
    court = court_el.get_text(strip=True) if court_el else ""

    # Date
    date_el = soup.select_one("div.doc_date, span.doc_date")
    date_str = date_el.get_text(strip=True) if date_el else ""

    # Full text
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

    full_text = re.sub(r"\n\s*\n", "\n\n", full_text).strip()

    # Extract judgment section (concluding part)
    judgment = _extract_judgment(full_text)

    # Create rich summary: intro + key sections + conclusion
    summary = _create_rich_summary(full_text, judgment)

    # Extract citations for metadata
    citations = _extract_citations(full_text)

    # Extract sections mentioned
    sections_mentioned = _extract_sections(full_text)

    return {
        "title": title or "Unknown",
        "url": url,
        "case_summary": summary,
        "judgment": judgment,
        "court": court,
        "date": date_str,
        "citations": json.dumps(citations),
        "sections_mentioned": json.dumps(sections_mentioned),
        "full_text_length": len(full_text),
    }


def _extract_judgment(text: str) -> str:
    """Extract the conclusive judgment/order portion."""
    # Try to find explicit judgment markers
    markers = [
        r"(?:ORDER|JUDGMENT|CONCLUSION|HELD|OPERATIVE\s+ORDER)\s*\n([\s\S]{100,3000}?)(?:\n\n\n|\Z)",
        r"(?:we\s+(?:hold|direct|order)|it\s+is\s+(?:ordered|directed|held)|the\s+(?:appeal|petition|writ)\s+is)\s+([\s\S]{50,2000}?)(?:\n\n|\Z)",
        r"(?:for\s+the\s+(?:foregoing|above)\s+reasons|in\s+(?:view|light)\s+of)[,\s]+([\s\S]{50,2000}?)(?:\n\n|\Z)",
    ]
    sections = []
    for pattern in markers:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            sections.append(match.group(0).strip())

    if sections:
        return " ".join(sections[-3:])[:3000]

    # Fallback: last 20 meaningful lines
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 30]
    return " ".join(lines[-20:])[:3000]


def _create_rich_summary(full_text: str, judgment: str, max_len: int = 3000) -> str:
    """Create a richer summary: intro (facts) + middle (arguments) + conclusion.

    This captures ~3000 chars which is much more context than the 200-500 word
    summaries we had before, giving the ML model and LLM more signal.
    """
    lines = full_text.split("\n")
    clean_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]

    if len(clean_lines) < 10:
        return full_text[:max_len]

    total = len(clean_lines)
    # Take intro (first 30%), mid-section sample (around 50%), and end
    intro_end = max(5, total // 3)
    mid_start = total // 3
    mid_end = 2 * total // 3
    mid_sample_size = min(10, mid_end - mid_start)

    intro = " ".join(clean_lines[:intro_end])[:1200]

    # Sample from middle to capture key arguments
    if mid_sample_size > 0:
        step = max(1, (mid_end - mid_start) // mid_sample_size)
        mid_lines = clean_lines[mid_start:mid_end:step][:mid_sample_size]
        middle = " ".join(mid_lines)[:800]
    else:
        middle = ""

    # Judgment/conclusion
    conclusion = judgment[:1000]

    parts = [intro]
    if middle:
        parts.append("[...arguments...]")
        parts.append(middle)
    parts.append("[...conclusion...]")
    parts.append(conclusion)

    return "\n\n".join(parts)[:max_len]


def _extract_citations(text: str) -> List[str]:
    patterns = [
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"AIR\s+\d{4}\s+\w+\s+\d+",
        r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",
        r"\d{4}\s+Cri\.?L\.?J\.?\s+\d+",
    ]
    citations = set()
    for p in patterns:
        for m in re.finditer(p, text):
            citations.add(m.group(0))
    return sorted(citations)[:30]


def _extract_sections(text: str) -> List[str]:
    """Extract IPC/BNS/CrPC/CPC etc. section references."""
    patterns = [
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:IPC|Indian\s+Penal\s+Code)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:CrPC|Cr\.P\.C\.|Code\s+of\s+Criminal)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:CPC|C\.P\.C\.|Code\s+of\s+Civil)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:BNS|Bharatiya\s+Nyaya)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:BNSS|Bharatiya\s+Nagarik)",
        r"Section\s+(\d+)\s+(?:of\s+)?(?:the\s+)?(?:Evidence\s+Act|BSA)",
        r"Article\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?Constitution",
        r"Section\s+138\s+(?:of\s+)?(?:the\s+)?(?:NI|Negotiable)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:NDPS|Narcotic)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:POCSO)",
        r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:IT\s+Act|Information\s+Technology)",
    ]
    sections = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            sections.add(m.group(0).strip())
    return sorted(sections)[:30]


# ── Kanoon Harvest Engine ─────────────────────────────────────────────────────

def harvest_kanoon(
    target: int,
    manifest: Dict[str, Any],
    pages_per_query: int = 30,
    courts_per_query: int = 5,
    workers: int = 1,
) -> int:
    """Harvest cases from Indian Kanoon using the full query catalog."""

    # Load query catalog
    if QUERY_CATALOG.exists():
        with open(QUERY_CATALOG, encoding="utf-8") as f:
            catalog = json.load(f)
        queries = catalog.get("queries", [])
    else:
        queries = []

    # Also add the bulk_harvest categories' queries
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from bulk_harvest import QUERY_CATEGORIES
        for cat in QUERY_CATEGORIES.values():
            queries.extend(cat.get("queries", []))
    except ImportError:
        pass

    # Deduplicate
    seen = set()
    unique_queries = []
    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    logger.info("Total unique queries: %d", len(unique_queries))

    processed_urls: Set[str] = set(manifest.get("processed_urls", []))
    completed_keys: Set[str] = set(manifest.get("completed_queries", []))
    dedup_hashes: Set[str] = set(manifest.get("dedup_hashes", []))
    total = manifest.get("total_harvested", 0)

    # Select courts — cycle through to diversify
    court_rotation = ["", "supremecourt", "delhi", "bombay", "allahabad",
                       "madras", "calcutta", "karnataka", "kerala", "punjab",
                       "rajasthan", "gauhati", "hyderabad", "patna"]

    def _process_query_court(query: str, court: str) -> List[Dict[str, str]]:
        """Process a single query+court combo: search + scrape."""
        results = []
        for page in range(1, pages_per_query + 1):
            links = search_kanoon(query, court=court, page=page)
            if not links:
                break  # No more results

            new_links = [l for l in links if l["url"] not in processed_urls]
            if not new_links:
                continue

            for link in new_links:
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                try:
                    case = scrape_case_detail(link["url"])
                except Exception as e:
                    logger.debug("Scrape error %s: %s", link["url"][:60], e)
                    continue

                if case:
                    h = _text_hash(case["case_summary"])
                    if h not in dedup_hashes:
                        case["category"] = _categorize_query(query)
                        case["query_source"] = query
                        results.append(case)
                        dedup_hashes.add(h)

                processed_urls.add(link["url"])

            time.sleep(random.uniform(PAGE_DELAY, PAGE_DELAY + 1.5))

        return results

    # Shuffle to avoid hitting same court repeatedly
    random.shuffle(unique_queries)

    for qi, query in enumerate(unique_queries):
        courts_to_try = court_rotation[:courts_per_query]

        for court in courts_to_try:
            key = f"{query}|{court}"
            if key in completed_keys:
                continue
            if total >= target:
                save_manifest(manifest)
                return total

            logger.info(
                "[%d/%d] Q%d/%d: '%s' court=%s",
                total, target, qi + 1, len(unique_queries), query[:50], court or "all"
            )

            try:
                if workers > 1:
                    # For future: parallel page fetching
                    batch = _process_query_court(query, court)
                else:
                    batch = _process_query_court(query, court)
            except Exception as e:
                logger.error("Query failed '%s': %s", query[:50], e)
                batch = []

            if batch:
                # Save to per-category CSV
                category = _categorize_query(query)
                csv_path = RAW_DIR / f"kanoon_{category}_bulk.csv"
                _append_to_csv(csv_path, batch)
                total += len(batch)
                manifest["source_counts"]["kanoon_scrape"] = manifest["source_counts"].get("kanoon_scrape", 0) + len(batch)
                manifest["category_counts"][category] = manifest["category_counts"].get(category, 0) + len(batch)
                logger.info("  +%d cases (total: %d/%d)", len(batch), total, target)

            completed_keys.add(key)

            # Checkpoint every query
            manifest["total_harvested"] = total
            manifest["processed_urls"] = list(processed_urls)
            manifest["completed_queries"] = list(completed_keys)
            manifest["dedup_hashes"] = list(dedup_hashes)
            save_manifest(manifest)

    return total


def _append_to_csv(path: Path, records: List[Dict[str, Any]]):
    """Append records to CSV, creating if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    fieldnames = ["title", "url", "case_summary", "judgment", "court", "date",
                   "category", "query_source", "citations", "sections_mentioned",
                   "full_text_length"]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


# ── External Dataset Loaders ──────────────────────────────────────────────────

# ILDC label mapping — expanded to 11 classes where possible
ILDC_LABEL_MAP = {
    0: "Relief Granted/Convicted",
    1: "Relief Denied/Dismissed",
    "accepted": "Relief Granted/Convicted",
    "rejected": "Relief Denied/Dismissed",
    "allowed": "Relief Granted/Convicted",
    "dismissed": "Relief Denied/Dismissed",
    "partly allowed": "Partially Allowed",
    "partly": "Partially Allowed",
}

HLDC_LABEL_MAP = {
    1: "Bail Granted",
    0: "Bail Denied",
    "bail_granted": "Bail Granted",
    "bail_denied": "Bail Denied",
    "granted": "Bail Granted",
    "denied": "Bail Denied",
}


def _load_json_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "train", "dev", "test", "cases"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def load_ildc(manifest: Dict[str, Any], target: int) -> int:
    """Load ILDC dataset if available."""
    ildc_dir = EXTERNAL_DIR / "ildc"
    if not ildc_dir.exists():
        logger.info("ILDC not found at %s — skipping. Download from: https://github.com/Exploration-Lab/CAIL", ildc_dir)
        return manifest.get("total_harvested", 0)

    total = manifest.get("total_harvested", 0)
    dedup_hashes: Set[str] = set(manifest.get("dedup_hashes", []))
    records = []

    for split in ("train.json", "dev.json", "test.json"):
        rows = _load_json_dataset(ildc_dir / split)
        logger.info("ILDC %s: %d records", split, len(rows))

        for rec in rows:
            if total >= target:
                break

            text = rec.get("doc") or rec.get("text") or rec.get("case_text") or ""
            if len(text.strip()) < 50:
                continue

            raw_label = rec.get("label")
            if raw_label is None:
                raw_label = rec.get("outcome") or rec.get("decision")
            label = ILDC_LABEL_MAP.get(raw_label) or ILDC_LABEL_MAP.get(str(raw_label).lower().strip())
            if not label:
                continue

            h = _text_hash(text)
            if h in dedup_hashes:
                continue
            dedup_hashes.add(h)

            # Rich summary from full text
            judgment = _extract_judgment(text)
            summary = _create_rich_summary(text, judgment)

            records.append({
                "title": str(rec.get("title", rec.get("case_name", "ILDC Case")))[:300],
                "url": rec.get("url", ""),
                "case_summary": summary,
                "judgment": judgment,
                "court": rec.get("court", ""),
                "date": rec.get("date", ""),
                "category": rec.get("case_type", "constitutional"),
                "query_source": "ILDC",
                "citations": "[]",
                "sections_mentioned": "[]",
                "full_text_length": len(text),
            })
            total += 1

    if records:
        csv_path = RAW_DIR / "external_ildc.csv"
        _append_to_csv(csv_path, records)
        manifest["source_counts"]["ildc"] = len(records)
        manifest["total_harvested"] = total
        manifest["dedup_hashes"] = list(dedup_hashes)
        save_manifest(manifest)
        logger.info("ILDC: loaded %d cases (total: %d/%d)", len(records), total, target)

    return total


def load_hldc(manifest: Dict[str, Any], target: int) -> int:
    """Load HLDC (bail prediction) dataset if available."""
    hldc_dir = EXTERNAL_DIR / "hldc"
    if not hldc_dir.exists():
        logger.info("HLDC not found at %s — skipping.", hldc_dir)
        return manifest.get("total_harvested", 0)

    total = manifest.get("total_harvested", 0)
    dedup_hashes: Set[str] = set(manifest.get("dedup_hashes", []))
    records = []

    for split in ("train.json", "dev.json", "test.json"):
        rows = _load_json_dataset(hldc_dir / split)
        logger.info("HLDC %s: %d records", split, len(rows))

        for rec in rows:
            if total >= target:
                break

            text = rec.get("text") or rec.get("doc") or ""
            if len(text.strip()) < 50:
                continue

            raw_label = rec.get("label")
            label = HLDC_LABEL_MAP.get(raw_label) or HLDC_LABEL_MAP.get(str(raw_label).lower().strip())
            if not label:
                continue

            h = _text_hash(text)
            if h in dedup_hashes:
                continue
            dedup_hashes.add(h)

            judgment = _extract_judgment(text)
            summary = _create_rich_summary(text, judgment)

            records.append({
                "title": str(rec.get("title", f"HLDC Bail Case"))[:300],
                "url": rec.get("url", ""),
                "case_summary": summary,
                "judgment": judgment,
                "court": "",
                "date": "",
                "category": "bail",
                "query_source": "HLDC",
                "citations": "[]",
                "sections_mentioned": "[]",
                "full_text_length": len(text),
            })
            total += 1

    if records:
        csv_path = RAW_DIR / "external_hldc.csv"
        _append_to_csv(csv_path, records)
        manifest["source_counts"]["hldc"] = len(records)
        manifest["total_harvested"] = total
        manifest["dedup_hashes"] = list(dedup_hashes)
        save_manifest(manifest)
        logger.info("HLDC: loaded %d cases (total: %d/%d)", len(records), total, target)

    return total


# ── Count existing data ───────────────────────────────────────────────────────

def count_existing() -> Tuple[int, Dict[str, int]]:
    """Count rows in all existing raw CSVs."""
    total = 0
    by_file: Dict[str, int] = {}

    for d in [RAW_DIR, ENRICHED_DIR]:
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, usecols=[0], nrows=None)
                count = len(df)
                by_file[csv_file.name] = count
                total += count
            except Exception:
                pass

    return total, by_file


# ── Master Builder ────────────────────────────────────────────────────────────

def build_master_csv():
    """Consolidate all CSVs into one master dataset with dedup."""
    frames = []

    for d in [RAW_DIR, ENRICHED_DIR]:
        if not d.exists():
            continue
        for csv_file in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, dtype=str)
                # Ensure required columns
                if "case_summary" not in df.columns and "case_data" in df.columns:
                    df["case_summary"] = df["case_data"]
                if "case_summary" not in df.columns:
                    continue

                if "judgment" not in df.columns and "judgement" in df.columns:
                    df["judgment"] = df["judgement"]

                df["_source_file"] = csv_file.name
                frames.append(df)
            except Exception as e:
                logger.warning("Failed to read %s: %s", csv_file.name, e)

    if not frames:
        logger.error("No data files found!")
        return

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined: %d rows from %d files", len(combined), len(frames))

    # Dedup on first 300 chars of case_summary
    combined["_dedup"] = combined["case_summary"].astype(str).str[:300].str.strip().str.lower()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_dedup"])
    combined = combined.drop(columns=["_dedup"])
    logger.info("After dedup: %d rows (removed %d duplicates)", len(combined), before - len(combined))

    # Filter out very short entries
    combined = combined[combined["case_summary"].astype(str).str.len() >= 50]
    logger.info("After length filter: %d rows", len(combined))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(MASTER_CSV, index=False)
    logger.info("Master CSV saved: %s (%d rows)", MASTER_CSV, len(combined))

    # Print distribution
    if "category" in combined.columns:
        dist = combined["category"].value_counts()
        logger.info("\nCategory distribution:")
        for cat, count in dist.head(30).items():
            logger.info("  %-30s %6d", cat, count)

    return len(combined)


# ── Status ────────────────────────────────────────────────────────────────────

def print_status():
    existing_total, by_file = count_existing()
    manifest = load_manifest()

    print("\n" + "=" * 60)
    print("  HARVEST STATUS")
    print("=" * 60)
    print(f"\n  Existing cases in data/raw + data/raw_enriched:")
    print(f"    Total: {existing_total:,} rows across {len(by_file)} files")
    print(f"\n  Manifest:")
    print(f"    Total harvested (tracked): {manifest.get('total_harvested', 0):,}")
    print(f"    Target: {manifest.get('target', 100000):,}")
    print(f"    Processed URLs: {len(manifest.get('processed_urls', [])):,}")
    print(f"    Completed queries: {len(manifest.get('completed_queries', [])):,}")
    print(f"    Dedup hashes: {len(manifest.get('dedup_hashes', [])):,}")

    sc = manifest.get("source_counts", {})
    if sc:
        print(f"\n  By source:")
        for src, cnt in sorted(sc.items()):
            print(f"    {src}: {cnt:,}")

    cc = manifest.get("category_counts", {})
    if cc:
        print(f"\n  By category (top 15):")
        for cat, cnt in sorted(cc.items(), key=lambda x: -x[1])[:15]:
            print(f"    {cat}: {cnt:,}")

    if by_file:
        print(f"\n  File breakdown:")
        for fname, count in sorted(by_file.items(), key=lambda x: -x[1])[:20]:
            print(f"    {fname}: {count:,}")

    print(f"\n  Gap to 100K: {max(0, 100000 - existing_total):,} cases needed")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grand Harvest Orchestrator — reach 1,00,000 cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/harvest_100k.py --status
    python scripts/harvest_100k.py --kanoon-only --pages 30 --courts 5
    python scripts/harvest_100k.py --external-only
    python scripts/harvest_100k.py --target 100000
    python scripts/harvest_100k.py --build-master
        """,
    )
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--pages", type=int, default=30, help="Max pages per query+court combo")
    parser.add_argument("--courts", type=int, default=5, help="Courts per query")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (future)")
    parser.add_argument("--kanoon-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--build-master", action="store_true", help="Only consolidate existing CSVs")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.build_master:
        build_master_csv()
        return

    # Ensure directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest() if args.resume else {
        "total_harvested": 0, "target": args.target,
        "processed_urls": [], "completed_queries": [], "dedup_hashes": [],
        "source_counts": {}, "category_counts": {}, "last_updated": "",
    }

    # Count existing as baseline
    existing_total, _ = count_existing()
    if manifest.get("total_harvested", 0) < existing_total:
        manifest["total_harvested"] = existing_total
        manifest["source_counts"]["existing"] = existing_total
    total = manifest["total_harvested"]

    logger.info("Starting harvest: current=%d, target=%d, gap=%d", total, args.target, args.target - total)

    # Phase 1: External datasets (fast, no rate limits)
    if not args.kanoon_only:
        logger.info("\n=== PHASE 1: External Datasets ===")
        total = load_ildc(manifest, args.target)
        if total < args.target:
            total = load_hldc(manifest, args.target)
        logger.info("After external: %d/%d", total, args.target)

    # Phase 2: Indian Kanoon scraping (slow, rate-limited)
    if not args.external_only and total < args.target:
        logger.info("\n=== PHASE 2: Indian Kanoon Scraping ===")
        total = harvest_kanoon(
            target=args.target,
            manifest=manifest,
            pages_per_query=args.pages,
            courts_per_query=args.courts,
            workers=args.workers,
        )
        logger.info("After Kanoon: %d/%d", total, args.target)

    # Phase 3: Build master CSV
    logger.info("\n=== PHASE 3: Building Master CSV ===")
    final_count = build_master_csv()

    # Final status
    print("\n" + "=" * 60)
    print(f"  HARVEST COMPLETE")
    print(f"  Final count: {final_count or total:,} / {args.target:,}")
    print(f"  Master CSV: {MASTER_CSV}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python scripts/retrain_model.py --rebuild-dataset --rebuild-search")
    print("  2. python scripts/build_semantic_index.py")


if __name__ == "__main__":
    main()

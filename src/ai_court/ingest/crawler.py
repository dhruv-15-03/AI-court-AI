"""Ingestion Phase (Phase 2) - Scrapy-based crawler scaffolding.

This module defines a lightweight interface around a (future) Scrapy project.
We avoid importing Scrapy at module import time to keep runtime dependencies optional
and tests fast. Actual crawling will live in a standalone Scrapy project folder
(`crawl/`) or integrated later.

Key intended features:
 - Rotating user agents & polite throttling
 - Incremental delta sync (hash-based; store last seen IDs)
 - De-duplication via content hash (sha256 canonical text)
 - Pluggable persistence (local disk, object storage, or DB)

Current status: placeholder structures & contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any, Optional
import hashlib, os, json, time

@dataclass
class RawJudgment:
    url: str
    title: str
    court: str | None
    date: str | None
    body_html: str
    scraped_at: float

    def content_hash(self) -> str:
        canon = (self.body_html or '').encode('utf-8')
        return hashlib.sha256(canon).hexdigest()


class JudgmentStore:
    """Simple file-based storage (can be swapped by DB/Blob later)."""
    def __init__(self, root: str = 'data/raw_html_store') -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def has(self, h: str) -> bool:
        return os.path.exists(os.path.join(self.root, f"{h}.json"))

    def write(self, j: RawJudgment) -> bool:
        path = os.path.join(self.root, f"{j.content_hash()}.json")
        if os.path.exists(path):
            return False
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(j.__dict__, f, ensure_ascii=False)
        return True


class CrawlerClient:
    """Placeholder orchestrator (would internally dispatch Scrapy requests)."""
    def __init__(self, store: Optional[JudgmentStore] = None) -> None:
        self.store = store or JudgmentStore()

    def crawl_seed(self, seed_urls: Iterable[str]) -> Dict[str, Any]:
        # Placeholder: pretend to fetch and store skeleton docs
        new, skipped = 0, 0
        for url in seed_urls:
            fake = RawJudgment(url=url, title=url.split('/')[-1], court=None, date=None, body_html=f"<p>{url}</p>", scraped_at=time.time())
            if self.store.write(fake):
                new += 1
            else:
                skipped += 1
        return {"new": new, "skipped": skipped}


__all__ = [
    'RawJudgment','JudgmentStore','CrawlerClient'
]

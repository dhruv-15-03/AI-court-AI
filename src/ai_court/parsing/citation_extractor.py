"""Citation extraction utilities.

Extracts case citations and builds a lightweight normalized representation.
Patterns covered (initial heuristic):
  - SCC style: (2024) 3 SCC 123
  - AIR style: AIR 2023 SC 455
  - Neutral style: 2023 SCC OnLine SC 123
  - Generic pattern: (YYYY) <vol> <REPORTER> <page>

Returns list of dicts: { 'raw': str, 'normalized': str|None, 'reporter': str|None }
Future: Link to internal case graph resolver.
"""
from __future__ import annotations
import re
from typing import List, Dict

SCC_RE = re.compile(r"\((20\d{2})\)\s+(\d+)\s+SCC\s+(\d+)")
AIR_RE = re.compile(r"AIR\s+(20\d{2})\s+SC\s+(\d+)")
NEUTRAL_RE = re.compile(r"20\d{2}\s+SCC\s+OnLine\s+SC\s+\d+")
GENERIC_RE = re.compile(r"\((20\d{2})\)\s+(\d+)\s+([A-Z]{2,10})\s+(\d+)")


def extract_citations(text: str) -> List[Dict]:
    if not text:
        return []
    found: List[Dict] = []
    # Order matters; more specific first
    for m in SCC_RE.finditer(text):
        year, vol, page = m.groups()
        raw = m.group(0)
        norm = f"SCC:{year}:{vol}:{page}"
        found.append({'raw': raw, 'normalized': norm, 'reporter': 'SCC'})
    for m in AIR_RE.finditer(text):
        year, page = m.groups()
        raw = m.group(0)
        norm = f"AIR:SC:{year}:{page}"
        found.append({'raw': raw, 'normalized': norm, 'reporter': 'AIR'})
    for m in NEUTRAL_RE.finditer(text):
        raw = m.group(0)
        found.append({'raw': raw, 'normalized': raw.replace(' ', ':'), 'reporter': 'NEUTRAL'})
    for m in GENERIC_RE.finditer(text):
        year, vol, rep, page = m.groups()
        raw = m.group(0)
        if rep in {'SCC','AIR'}:  # already captured
            continue
        norm = f"{rep}:{year}:{vol}:{page}"
        found.append({'raw': raw, 'normalized': norm, 'reporter': rep})
    # Deduplicate by raw span order preserving
    seen = set()
    dedup = []
    for c in found:
        key = c['raw']
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup

if __name__ == '__main__':
    sample = "This was reported in (2024) 3 SCC 123 and AIR 2023 SC 455 as well as 2023 SCC OnLine SC 789."
    print(extract_citations(sample))

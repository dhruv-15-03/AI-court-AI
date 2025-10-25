"""Statute / section reference extraction.

Heuristics for Indian statute references such as:
  - Section 482 CrPC
  - Sec. 438 of the Code of Criminal Procedure, 1973
  - S. 138 NI Act
  - Section 420 IPC

Returns list of dicts: { 'raw': str, 'act_id': str|None, 'section': str|None }
Future: canonical mapping via act dictionary.
"""
from __future__ import annotations
import re
from typing import List, Dict

SECTION_RE = re.compile(r"\b(?:section|sec\.|s\.)\s*(\d+[A-Za-z]*)(?:\s*[-/]?\s*\(?\w+\)?)?\s*(?:of\s+the\s+)?(?:(?:code\s+of\s+criminal\s+procedure|crpc|code\s+of\s+criminal\s+procedure,?\s*1973)|ipc|indian\s+penal\s+code|ni\s+act|negotiable\s+instruments\s+act)\b", re.IGNORECASE)

ACT_ALIASES = {
    'crpc': 'CrPC',
    'code of criminal procedure': 'CrPC',
    'code of criminal procedure, 1973': 'CrPC',
    'ipc': 'IPC',
    'indian penal code': 'IPC',
    'ni act': 'NI Act',
    'negotiable instruments act': 'NI Act'
}

ACT_CAPTURE = re.compile(r"(crpc|code of criminal procedure(?:,?\s*1973)?|ipc|indian penal code|ni act|negotiable instruments act)", re.IGNORECASE)


def extract_statutes(text: str) -> List[Dict]:
    if not text:
        return []
    refs: List[Dict] = []
    for m in SECTION_RE.finditer(text):
        raw = m.group(0)
        section = m.group(1)
        act_match = ACT_CAPTURE.search(raw)
        act_id = None
        if act_match:
            norm = act_match.group(1).lower()
            act_id = ACT_ALIASES.get(norm, act_match.group(1))
        refs.append({'raw': raw, 'act_id': act_id, 'section': section})
    # Dedup by (act_id, section)
    seen = set()
    dedup = []
    for r in refs:
        key = (r.get('act_id'), r.get('section'))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup

if __name__ == '__main__':
    sample = "Applications under Section 482 CrPC and Section 138 NI Act with reference also to Sec. 420 IPC."
    print(extract_statutes(sample))

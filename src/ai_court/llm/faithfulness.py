"""Citation faithfulness verification for RAG answers.

Legal AI's single biggest liability is a fabricated case citation. After the LLM
produces an answer grounded in retrieved documents, :func:`verify_citations`
extracts the case names the model actually cited and checks each one against the
titles of the documents that were retrieved. Citations that share no significant
party tokens with any retrieved document are surfaced as ``unverified`` so the
caller can flag or strip them.

This is a heuristic guard, deliberately lenient (it favours *not* flagging a real
citation over aggressively flagging a grounded one), but it reliably catches the
common failure mode of the model inventing a plausible-sounding case that was
never retrieved.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

# Matches "<Party> v./vs./versus <Party>" with each party up to ~7 tokens. Party
# tokens allow letters, digits, dots, ampersands, apostrophes and hyphens so a
# name like "M/s. A.B. & Co." survives, while commas/parentheses terminate a party.
_CASE_RE = re.compile(
    r"([A-Z][A-Za-z0-9.&'\-]*(?:\s+[A-Za-z0-9.&'\-]+){0,6}?)"
    r"\s+(?:v\.?|vs\.?|versus)\s+"
    r"([A-Z][A-Za-z0-9.&'\-]*(?:\s+[A-Za-z0-9.&'\-]+){0,6})"
)

# Tokens too generic to prove a citation came from a specific retrieved document.
_STOPWORDS = {
    "the", "of", "and", "union", "state", "ltd", "limited", "co", "company",
    "vs", "versus", "others", "anr", "ors", "india", "government",
}


def _normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()


def _significant_tokens(norm: str) -> Set[str]:
    return {t for t in norm.split() if len(t) >= 4 and t not in _STOPWORDS}


def extract_case_citations(text: str) -> List[str]:
    """Return de-duplicated "X v. Y" style case citations found in *text*."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in _CASE_RE.finditer(text or ""):
        cite = re.sub(r"\s+", " ", m.group(0)).strip().rstrip(".,;:")
        key = _normalize(cite)
        if key and key not in seen:
            seen.add(key)
            out.append(cite)
    return out


def _matches_any(cite_norm: str, doc_norms: Sequence[str]) -> bool:
    ctoks = _significant_tokens(cite_norm)
    if not ctoks:
        # Nothing distinctive to test against -> don't flag as hallucinated.
        return True
    for dt in doc_norms:
        if cite_norm and cite_norm in dt:
            return True
        if ctoks & _significant_tokens(dt):
            return True
    return False


def verify_citations(
    answer: str, documents: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Cross-check case citations in *answer* against retrieved *documents*.

    Returns a dict with ``cited_cases``, ``verified_citations``,
    ``unverified_citations`` and a ``grounded`` boolean (True when every cited
    case maps to a retrieved document, or when no case was cited).
    """
    cited = extract_case_citations(answer)
    doc_norms = [
        _normalize(d.get("title"))
        for d in (documents or [])
        if isinstance(d, dict) and d.get("title")
    ]
    verified: List[str] = []
    unverified: List[str] = []
    for cite in cited:
        if _matches_any(_normalize(cite), doc_norms):
            verified.append(cite)
        else:
            unverified.append(cite)
    return {
        "cited_cases": cited,
        "verified_citations": verified,
        "unverified_citations": unverified,
        "grounded": len(unverified) == 0,
    }


__all__ = ["verify_citations", "extract_case_citations"]

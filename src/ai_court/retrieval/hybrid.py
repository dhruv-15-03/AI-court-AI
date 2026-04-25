"""Hybrid Retrieval - Lexical + Vector fusion utilities.

Provides reusable reciprocal-rank fusion independent of API layer,
with support for weighted fusion and outcome filtering.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Sequence


def reciprocal_rank_fusion(
    semantic: Sequence[Dict[str, Any]],
    lexical: Sequence[Dict[str, Any]],
    k: int = 10,
    K: int = 60,
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
    outcome_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fuse semantic and lexical search results using reciprocal-rank fusion.

    Args:
        semantic: Ranked results from dense/semantic search.
        lexical: Ranked results from TF-IDF/lexical search.
        k: Maximum number of results to return.
        K: RRF constant (higher = smoother blending).
        semantic_weight: Multiplier for the semantic rank contribution.
        lexical_weight: Multiplier for the lexical rank contribution.
        outcome_filter: If provided, only keep results whose ``outcome``
            field contains this substring (case-insensitive).

    Returns:
        List of fused result dicts sorted by ``fusion_score``, truncated to *k*.
    """
    s_index = {(r.get('url'), r.get('title')): i for i, r in enumerate(semantic)}
    l_index = {(r.get('url'), r.get('title')): i for i, r in enumerate(lexical)}
    all_keys = set(s_index) | set(l_index)
    fused: List[Dict[str, Any]] = []
    for key in all_keys:
        sr = s_index.get(key, 10**6)
        lr = l_index.get(key, 10**6)
        score = semantic_weight / (K + sr) + lexical_weight / (K + lr)
        # pick representative doc (prefer semantic)
        base = None
        if key in s_index:
            base = semantic[s_index[key]]
        elif key in l_index:
            base = lexical[l_index[key]]
        if base:
            fused.append({**base, 'fusion_score': score})

    # Outcome filtering
    if outcome_filter:
        needle = outcome_filter.lower()
        fused = [r for r in fused if needle in (r.get('outcome') or '').lower()]

    fused.sort(key=lambda x: -x['fusion_score'])
    return fused[:k]


__all__ = ['reciprocal_rank_fusion']

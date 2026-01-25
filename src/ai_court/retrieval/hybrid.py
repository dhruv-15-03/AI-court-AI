"""Hybrid Retrieval (Phase 2 & 4) - Lexical + Vector fusion utilities.

Provides reusable reciprocal-rank fusion independent of API layer.
Future Enhancements:
 - Weighted z-score fusion
 - Reranker (cross-encoder) integration
 - Caching layer
"""
from __future__ import annotations
from typing import List, Dict, Any, Sequence


def reciprocal_rank_fusion(semantic: Sequence[Dict[str, Any]], lexical: Sequence[Dict[str, Any]], k: int = 10, K: int = 60) -> List[Dict[str, Any]]:
    s_index = {(r.get('url'), r.get('title')): i for i, r in enumerate(semantic)}
    l_index = {(r.get('url'), r.get('title')): i for i, r in enumerate(lexical)}
    all_keys = set(s_index) | set(l_index)
    fused = []
    for key in all_keys:
        sr = s_index.get(key, 10**6)
        lr = l_index.get(key, 10**6)
        score = 1.0 / (K + sr) + 1.0 / (K + lr)
        # pick representative doc (prefer semantic)
        base = None
        if key in s_index:
            base = semantic[s_index[key]]
        elif key in l_index:
            base = lexical[l_index[key]]
        if base:
            fused.append({**base, 'fusion_score': score})
    fused.sort(key=lambda x: -x['fusion_score'])
    return fused[:k]


__all__ = ['reciprocal_rank_fusion']

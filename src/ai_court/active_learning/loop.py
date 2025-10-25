"""Active Learning Loop Scaffold (Phase 5).

Implements minimal queueing + selection strategies placeholders.
Future features:
 - Uncertainty sampling (max entropy / margin)
 - Diversity sampling (clustering-based)
 - Human labeling API integration
 - LLM label suggestion with provenance trace
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import heapq, time


@dataclass(order=True)
class QueueItem:
    priority: float
    doc_id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    queued_at: float = field(compare=False, default_factory=lambda: time.time())


class ActiveLearningQueue:
    def __init__(self):
        self._heap: List[QueueItem] = []

    def push(self, doc_id: str, uncertainty: float, payload: Dict[str, Any]):
        heapq.heappush(self._heap, QueueItem(-uncertainty, doc_id, payload))  # negative for max-heap

    def pop_batch(self, n: int = 10) -> List[QueueItem]:
        out = []
        for _ in range(min(n, len(self._heap))):
            out.append(heapq.heappop(self._heap))
        return out

    def __len__(self):  # pragma: no cover
        return len(self._heap)


__all__ = ['ActiveLearningQueue','QueueItem']

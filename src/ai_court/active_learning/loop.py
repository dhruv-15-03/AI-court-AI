"""Active Learning Loop — queuing, selection strategies, and retraining.

Features:
 - Priority queue with uncertainty sampling (entropy / margin)
 - Diversity sampling via term-frequency clustering
 - Labeled item ingestion into training data
 - Incremental retraining trigger
 - LLM label suggestion (optional)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import heapq
import json
import logging
import math
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


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


# ── Uncertainty Scoring ───────────────────────────────────────────────

def entropy_uncertainty(probabilities: Dict[str, float]) -> float:
    """Shannon entropy over predicted class probabilities (higher = more uncertain)."""
    probs = np.array(list(probabilities.values()), dtype=float)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def margin_uncertainty(probabilities: Dict[str, float]) -> float:
    """1 minus the margin between top-2 class probabilities (higher = more uncertain)."""
    probs = sorted(probabilities.values(), reverse=True)
    if len(probs) < 2:
        return 0.0
    return float(1.0 - (probs[0] - probs[1]))


def compute_uncertainty(
    probabilities: Dict[str, float],
    method: str = "entropy",
) -> float:
    """Compute uncertainty score from class probabilities.

    Args:
        probabilities: Dict of class_label -> probability
        method: 'entropy', 'margin', or 'confidence' (1 - max)

    Returns:
        Float uncertainty score (higher = more uncertain)
    """
    if not probabilities:
        return 1.0
    if method == "entropy":
        return entropy_uncertainty(probabilities)
    elif method == "margin":
        return margin_uncertainty(probabilities)
    else:  # confidence
        return 1.0 - max(probabilities.values())


# ── Labeled Data Store ────────────────────────────────────────────────

class LabeledDataStore:
    """Stores human-labeled corrections for retraining."""

    def __init__(self, store_path: str = "data/active_learning_labels.jsonl"):
        self.store_path = store_path
        os.makedirs(os.path.dirname(store_path) if os.path.dirname(store_path) else ".", exist_ok=True)

    def add_label(
        self,
        text: str,
        label: str,
        case_type: str = "",
        source: str = "human",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a labeled example to the store."""
        record = {
            "text": text,
            "label": label,
            "case_type": case_type,
            "source": source,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Stored label: %s -> %s (source=%s)", text[:50], label, source)

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all labeled examples."""
        if not os.path.exists(self.store_path):
            return []
        records = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def count(self) -> int:
        if not os.path.exists(self.store_path):
            return 0
        count = 0
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


# ── Retraining Engine ─────────────────────────────────────────────────

class RetrainingEngine:
    """Manages incremental model retraining when enough new labels accumulate."""

    def __init__(
        self,
        label_store: LabeledDataStore,
        retrain_threshold: int = 50,
        model_dir: str = "models",
        data_dirs: Optional[List[str]] = None,
    ):
        self.label_store = label_store
        self.retrain_threshold = retrain_threshold
        self.model_dir = model_dir
        self.data_dirs = data_dirs or ["data/raw", "data/raw_enriched"]
        self._labels_at_last_train = 0

    def should_retrain(self) -> bool:
        """Check if enough new labels have accumulated to justify retraining."""
        current = self.label_store.count()
        new_labels = current - self._labels_at_last_train
        return new_labels >= self.retrain_threshold

    def retrain(self) -> Dict[str, Any]:
        """Retrain the model with original data + new human labels.

        Returns metadata about the training run.
        """
        import pandas as pd
        from ai_court.model.legal_case_classifier import LegalCaseClassifier
        from ai_court.data.prepare_dataset import build_from_dirs

        logger.info("Starting retraining cycle...")

        # 1. Rebuild base dataset from raw CSVs
        base_csv = os.path.join("data", "processed", "all_cases.csv")
        build_from_dirs(self.data_dirs, base_csv, min_text_len=20, dedupe=True)
        base_df = pd.read_csv(base_csv)

        # 2. Append human-labeled data
        labels = self.label_store.load_all()
        if labels:
            new_rows = []
            for rec in labels:
                new_rows.append({
                    "case_data": rec["text"],
                    "case_type": rec.get("case_type", "Unknown"),
                    "judgement": rec["label"],
                })
            new_df = pd.DataFrame(new_rows)
            combined = pd.concat([base_df, new_df], ignore_index=True)
            logger.info(
                "Combined dataset: %d base + %d new labels = %d total",
                len(base_df), len(new_df), len(combined),
            )
        else:
            combined = base_df

        # 3. Train new model
        clf = LegalCaseClassifier()
        X_train, X_test, y_train, y_test = clf.prepare_data(combined)
        pipeline, f1 = clf.train_model(X_train, y_train)
        metrics = clf.evaluate(pipeline, X_test, y_test)

        # 4. Save new model
        new_model_path = os.path.join(self.model_dir, "legal_case_classifier.pkl")
        clf.save_model(new_model_path)
        logger.info("Retrained model saved to %s (F1=%.3f)", new_model_path, f1)

        self._labels_at_last_train = self.label_store.count()

        return {
            "status": "retrained",
            "total_samples": len(combined),
            "new_labels_used": len(labels),
            "f1_score": f1,
            "metrics": metrics,
            "model_path": new_model_path,
            "timestamp": time.time(),
        }


# ── LLM Label Suggestion ─────────────────────────────────────────────

def suggest_label_with_llm(
    text: str,
    llm_client: Any,
    valid_labels: List[str],
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to suggest a label for an uncertain prediction.

    Returns dict with 'suggested_label', 'reasoning', 'confidence'.
    """
    if llm_client is None:
        return None

    labels_str = ", ".join(valid_labels)
    prompt = (
        f"You are an expert Indian legal analyst. Given the following case text, "
        f"classify it into exactly ONE of these outcome categories:\n"
        f"[{labels_str}]\n\n"
        f"CASE TEXT:\n{text[:2000]}\n\n"
        f"Respond in JSON format:\n"
        f'{{"suggested_label": "...", "reasoning": "brief explanation", "confidence": 0.0-1.0}}'
    )
    try:
        raw = llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        result = json.loads(raw.strip())
        if result.get("suggested_label") in valid_labels:
            return result
        return None
    except Exception as exc:
        logger.warning("LLM label suggestion failed: %s", exc)
        return None


__all__ = [
    'ActiveLearningQueue',
    'QueueItem',
    'compute_uncertainty',
    'entropy_uncertainty',
    'margin_uncertainty',
    'LabeledDataStore',
    'RetrainingEngine',
    'suggest_label_with_llm',
]

"""In-memory searchable corpus of Indian statutory provisions."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_court.corpus.schemas import ActInfo, StatuteSection

logger = logging.getLogger(__name__)


class StatuteCorpus:
    """Load and search Indian statutory provisions from JSON files."""

    def __init__(self, corpus_dir: Optional[str] = None):
        self.corpus_dir = corpus_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "statutes"
        )
        self._sections: list[StatuteSection] = []
        self._acts: dict[str, ActInfo] = {}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._section_lookup: dict[tuple[str, str], StatuteSection] = {}
        # Optional semantic index (lazy-loaded)
        self._vector_store: Any = None

    def attach_vector_store(self, vector_store: Any) -> None:
        """Attach a pre-built :class:`VectorStore` for semantic search."""
        self._vector_store = vector_store
        logger.info("[corpus] vector store attached (n=%d)", len(vector_store))

    @property
    def loaded(self) -> bool:
        return len(self._sections) > 0

    def load(self) -> None:
        """Load all statute JSON files from corpus_dir."""
        if not os.path.isdir(self.corpus_dir):
            logger.warning("Statute corpus dir not found: %s", self.corpus_dir)
            return

        count = 0
        for fname in sorted(os.listdir(self.corpus_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.corpus_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                act_id = data["act_id"]
                act_name = data["full_name"]

                self._acts[act_id] = ActInfo(
                    act_id=act_id,
                    full_name=act_name,
                    short_name=data.get("short_name", act_id),
                    year=data.get("year", 0),
                    total_sections=len(data.get("sections", [])),
                    chapters=data.get("chapters", []),
                )

                for sec in data.get("sections", []):
                    section = StatuteSection(
                        act_id=act_id,
                        act_name=act_name,
                        section_number=sec["section_number"],
                        heading=sec.get("heading", ""),
                        body_text=sec.get("body_text", ""),
                        chapter=sec.get("chapter"),
                    )
                    self._sections.append(section)
                    self._section_lookup[(act_id, sec["section_number"])] = section
                    count += 1
            except Exception as e:
                logger.warning("Failed to load statute file %s: %s", fname, e)

        if self._sections:
            self._build_index()

        logger.info(
            "Loaded %d statute sections from %d acts", count, len(self._acts)
        )

    def _build_index(self) -> None:
        """Build TF-IDF index over section text for search."""
        texts = [
            f"{s.act_id} Section {s.section_number} {s.heading} {s.body_text}"
            for s in self._sections
        ]
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search_sections(
        self,
        query: str,
        act_filter: Optional[str] = None,
        *,
        use_semantic: Optional[bool] = None,
        alpha: float = 0.5,
    ) -> list[StatuteSection]:
        """Search for relevant statute sections by query text.

        Args:
            query: Free-text query.
            act_filter: Restrict results to a single act (e.g. ``"IPC"``).
            k: Number of results to return.
            use_semantic: If True, blend semantic search with TF-IDF.
                Defaults to True when a vector store is attached.
            alpha: Weight for TF-IDF in hybrid score (0..1). Semantic gets ``1-alpha``.
        """
        if not self._vectorizer or self._matrix is None:
            return []

        use_semantic = (
            self._vector_store is not None if use_semantic is None else bool(use_semantic)
        )

        qv = self._vectorizer.transform([query.lower()])
        tfidf_scores = (self._matrix @ qv.T).toarray().ravel()

        if use_semantic and self._vector_store is not None:
            try:
                # Fetch more than k so we can fuse meaningfully.
                semantic_hits = self._vector_store.search(query, k=max(k * 3, 15))
                sem_scores = np.zeros_like(tfidf_scores)
                for hit in semantic_hits:
                    if hit.index < len(sem_scores):
                        sem_scores[hit.index] = hit.score
                # Normalize TF-IDF to 0..1 for fair blending
                max_tf = float(tfidf_scores.max()) if tfidf_scores.size else 0.0
                if max_tf > 0:
                    tfidf_norm = tfidf_scores / max_tf
                else:
                    tfidf_norm = tfidf_scores
                scores = alpha * tfidf_norm + (1.0 - alpha) * sem_scores
            except Exception as exc:
                logger.warning("[corpus] semantic search failed, using TF-IDF only: %s", exc)
                scores = tfidf_scores
        else:
            scores = tfidf_scores
        scores = (self._matrix @ qv.T).toarray().ravel()

        # Apply act filter
        if act_filter:
            act_filter_upper = act_filter.upper()
            for i, sec in enumerate(self._sections):
                if sec.act_id.upper() != act_filter_upper:
                    scores[i] = 0.0

        top_idx = np.argsort(-scores)[:k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.01:
                results.append(self._sections[idx])
        return results

    def get_section(
        self, act_id: str, section_number: str
    ) -> Optional[StatuteSection]:
        """Direct lookup of a specific section."""
        return self._section_lookup.get((act_id.upper(), section_number)) or \
               self._section_lookup.get((act_id, section_number))

    def get_act_overview(self, act_id: str) -> Optional[ActInfo]:
        """Get act metadata."""
        return self._acts.get(act_id) or self._acts.get(act_id.upper())

    def get_sections_by_act(self, act_id: str) -> list[StatuteSection]:
        """Get all sections of a given act."""
        aid = act_id.upper()
        return [s for s in self._sections if s.act_id.upper() == aid]

    def format_for_context(
        self, sections: list[StatuteSection], max_length: int = 4000
    ) -> str:
        """Format sections as context string for LLM prompt."""
        if not sections:
            return "No applicable statutory provisions found."

        parts = []
        total = 0
        for s in sections:
            entry = (
                f"**{s.act_name} - Section {s.section_number}: {s.heading}**\n"
                f"{s.body_text}\n"
            )
            if total + len(entry) > max_length:
                break
            parts.append(entry)
            total += len(entry)
        return "\n".join(parts)

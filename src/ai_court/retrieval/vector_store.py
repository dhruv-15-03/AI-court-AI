"""Semantic vector store using sentence-transformers + FAISS (with numpy fallback).

This module adds a semantic retrieval layer on top of the existing TF-IDF search,
for when keyword match is insufficient (e.g. paraphrased queries, conceptual search).

Design goals:
    * **Optional** — never required at runtime. Skips gracefully on the free tier.
    * **Cheap** — uses ``all-MiniLM-L6-v2`` (~22MB, 384-dim, CPU-friendly).
    * **Portable** — FAISS is preferred; falls back to numpy brute-force if FAISS is missing.
    * **Persistable** — save/load to disk so we don't re-embed on cold start.

Typical usage:

    >>> from ai_court.retrieval.vector_store import VectorStore
    >>> vs = VectorStore.build(["doc 1 text", "doc 2 text"], metadata=[{"id": 1}, {"id": 2}])
    >>> vs.save("models/statute_vectors.npz")
    >>> hits = vs.search("my query", k=5)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class VectorHit:
    """A single retrieval result."""
    index: int
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""


class VectorStore:
    """Semantic retrieval using dense embeddings + cosine similarity.

    Lazy-loads the embedding model; never imported at module import time.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadata: Optional[Sequence[dict]] = None,
        model_name: str = DEFAULT_EMBED_MODEL,
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"texts ({len(texts)}) and embeddings ({embeddings.shape[0]}) size mismatch"
            )
        # Pre-normalize for cosine via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = (embeddings / norms).astype(np.float32)
        self._texts = list(texts)
        self._metadata = list(metadata) if metadata is not None else [{} for _ in texts]
        self.model_name = model_name
        self.dim = self._embeddings.shape[1]
        self._faiss_index: Any = None
        self._encoder: Any = None
        self._try_build_faiss()

    # ------------------------------------------------------------------
    # Factory: build from raw texts (embeds them)
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        metadata: Optional[Sequence[dict]] = None,
        model_name: str = DEFAULT_EMBED_MODEL,
        batch_size: int = 32,
    ) -> "VectorStore":
        """Embed ``texts`` and construct a VectorStore."""
        encoder = _load_encoder(model_name)
        logger.info("[vector_store] embedding %d texts with %s", len(texts), model_name)
        embeddings = encoder.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves
        )
        vs = cls(np.asarray(embeddings, dtype=np.float32), texts, metadata, model_name)
        vs._encoder = encoder
        return vs

    # ------------------------------------------------------------------
    def _try_build_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            index = faiss.IndexFlatIP(self.dim)
            index.add(self._embeddings)
            self._faiss_index = index
            logger.info("[vector_store] FAISS IndexFlatIP built (n=%d)", len(self._texts))
        except ImportError:
            logger.info("[vector_store] faiss not installed — using numpy brute-force (still fast <100k docs)")
            self._faiss_index = None
        except Exception as exc:
            logger.warning("[vector_store] FAISS build failed, using numpy: %s", exc)
            self._faiss_index = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = _load_encoder(self.model_name)
        return self._encoder

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[VectorHit]:
        """Return the top-k most semantically similar entries to ``query``."""
        if not query or not query.strip():
            return []
        encoder = self._get_encoder()
        qv = encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(qv, k)
            scores = scores[0]
            idx = idx[0]
        else:
            sims = (self._embeddings @ qv.T).ravel()
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]

        results: List[VectorHit] = []
        for i, s in zip(idx, scores):
            if i < 0 or s < min_score:
                continue
            results.append(
                VectorHit(
                    index=int(i),
                    score=float(s),
                    metadata=dict(self._metadata[int(i)]),
                    text=self._texts[int(i)],
                )
            )
        return results

    # ------------------------------------------------------------------
    def add(
        self,
        texts: Sequence[str],
        metadata: Optional[Sequence[dict]] = None,
        batch_size: int = 32,
    ) -> int:
        """Incrementally embed and append ``texts`` to this store.

        Returns the number of new items added.
        """
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return 0
        encoder = self._get_encoder()
        vecs = encoder.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        meta = list(metadata) if metadata is not None else [{} for _ in texts]

        self._embeddings = np.vstack([self._embeddings, vecs])
        self._texts.extend(texts)
        self._metadata.extend(meta)
        # Keep FAISS index in sync — cheapest is to append to existing index
        if self._faiss_index is not None:
            try:
                self._faiss_index.add(vecs)
            except Exception:  # rebuild fallback
                self._try_build_faiss()
        logger.info("[vector_store] +%d vectors (total=%d)", len(texts), len(self._texts))
        return len(texts)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist embeddings + texts + metadata to a ``.npz`` file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self._embeddings,
            texts=np.asarray(self._texts, dtype=object),
            metadata=np.asarray(self._metadata, dtype=object),
            model_name=np.asarray([self.model_name]),
        )
        logger.info("[vector_store] saved %d vectors → %s", len(self._texts), path)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        data = np.load(path, allow_pickle=True)
        model_name = str(data["model_name"][0])
        vs = cls(
            embeddings=data["embeddings"],
            texts=list(data["texts"]),
            metadata=list(data["metadata"]),
            model_name=model_name,
        )
        logger.info("[vector_store] loaded %d vectors from %s", len(vs._texts), path)
        return vs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._texts)

    @property
    def texts(self) -> List[str]:
        return list(self._texts)

    @property
    def metadata(self) -> List[dict]:
        return list(self._metadata)


# ---------------------------------------------------------------------- helpers


def _load_encoder(model_name: str) -> Any:
    """Lazy import sentence-transformers; raise a clear error if missing."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for VectorStore. "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


def hybrid_score(
    tfidf_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    """Linear-combine TF-IDF and semantic similarity scores."""
    return alpha * tfidf_score + (1.0 - alpha) * semantic_score

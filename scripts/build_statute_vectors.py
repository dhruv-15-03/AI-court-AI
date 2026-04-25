"""Script: build semantic FAISS vector store for the statute corpus.

Usage:
    python -m scripts.build_statute_vectors
    python scripts/build_statute_vectors.py --out models/statute_vectors.npz

This is a *one-shot* build step. The produced ``.npz`` can be loaded at app
startup via ``VectorStore.load`` for semantic section retrieval.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai_court.corpus.statutes import StatuteCorpus  # noqa: E402
from ai_court.retrieval.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_statute_vectors")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic index over statute corpus")
    parser.add_argument("--out", default="models/statute_vectors.npz", help="Output .npz path")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Override statute JSON dir (default: data/statutes)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Sentence-transformers model name",
    )
    args = parser.parse_args()

    corpus = StatuteCorpus(corpus_dir=args.corpus_dir)
    corpus.load()
    if not corpus.loaded:
        logger.error("Statute corpus is empty. Ensure data/statutes/ has JSON files.")
        return 1

    sections = corpus._sections  # noqa: SLF001 — intentional internal access
    texts = [
        f"{s.act_name} Section {s.section_number}: {s.heading}. {s.body_text}"
        for s in sections
    ]
    metadata = [
        {
            "act_id": s.act_id,
            "act_name": s.act_name,
            "section_number": s.section_number,
            "heading": s.heading,
            "chapter": s.chapter,
        }
        for s in sections
    ]

    logger.info("Embedding %d statute sections with %s …", len(texts), args.model)
    vs = VectorStore.build(texts, metadata=metadata, model_name=args.model)
    vs.save(args.out)
    logger.info("Done. Saved %d vectors (dim=%d) → %s", len(vs), vs.dim, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

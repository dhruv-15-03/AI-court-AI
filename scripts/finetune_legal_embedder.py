"""Fine-tune a sentence-transformer on the Indian legal corpus with TSDAE.

TSDAE (Transformer-based Sequential Denoising AutoEncoder) is an unsupervised
domain-adaptation technique: we delete random words from each sentence and
train the encoder-decoder to reconstruct it. This shifts the embedding space
toward legal vocabulary without any labeled pairs.

Reference: Wang et al. 2021 — "TSDAE: Using Transformer-based Sequential Denoising
Auto-Encoder for Unsupervised Sentence Embedding Learning".

Usage:
    python scripts/finetune_legal_embedder.py \\
        --corpus data/processed/all_cases.csv \\
        --out models/legal_miniLM --epochs 1 --batch 8

After training, set ``EMBED_MODEL`` to the output path to use the adapted
model everywhere the VectorStore is instantiated:
    export EMBED_MODEL=models/legal_miniLM

Requires: sentence-transformers >= 2.2, torch, nltk (for word tokenization).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("finetune_legal_embedder")


def load_corpus(
    csv_path: Path | None,
    extra_dirs: Iterable[Path],
    min_chars: int = 200,
    max_sentences: int | None = None,
) -> List[str]:
    """Aggregate training sentences from CSVs and plain-text folders."""
    sentences: List[str] = []

    if csv_path and csv_path.exists():
        with csv_path.open(encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("case_data") or row.get("text") or
                        row.get("description") or "").strip()
                if len(text) >= min_chars:
                    # Split long judgments into paragraph-ish chunks
                    for chunk in text.split("\n\n"):
                        chunk = chunk.strip()
                        if len(chunk) >= min_chars:
                            sentences.append(chunk[:1500])

    for d in extra_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() not in {".txt", ".md", ".json"}:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for chunk in txt.split("\n\n"):
                chunk = chunk.strip()
                if len(chunk) >= min_chars:
                    sentences.append(chunk[:1500])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if max_sentences and len(unique) >= max_sentences:
            break

    logger.info("Training corpus: %d unique passages", len(unique))
    return unique


def train(
    sentences: List[str],
    base_model: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset
    from torch.utils.data import DataLoader

    logger.info("Base model: %s", base_model)
    word_emb = models.Transformer(base_model)
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(),
        pooling_mode="cls",  # TSDAE recommends CLS
    )
    model = SentenceTransformer(modules=[word_emb, pooling])

    dataset = DenoisingAutoEncoderDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=base_model, tie_encoder_decoder=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training TSDAE for %d epoch(s), batch=%d, lr=%.2e → %s",
                epochs, batch_size, lr, out_dir)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        weight_decay=0.0,
        scheduler="constantlr",
        optimizer_params={"lr": lr},
        show_progress_bar=True,
        output_path=str(out_dir),
    )
    logger.info("Saved adapted model → %s", out_dir)
    logger.info("Point EMBED_MODEL at this path to use it in the running service.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(ROOT / "data" / "processed" / "all_cases.csv"))
    p.add_argument("--extra-dirs", nargs="*",
                   default=[str(ROOT / "data" / "statutes")])
    p.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default=str(ROOT / "models" / "legal_miniLM"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-sentences", type=int, default=100_000)
    args = p.parse_args()

    sentences = load_corpus(
        Path(args.corpus),
        [Path(d) for d in args.extra_dirs],
        max_sentences=args.max_sentences,
    )
    if len(sentences) < 100:
        logger.error("Too few sentences (%d). Build the corpus first.", len(sentences))
        sys.exit(1)
    train(sentences, args.base_model, Path(args.out), args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()

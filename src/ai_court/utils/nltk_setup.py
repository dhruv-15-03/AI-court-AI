"""NLTK resource management.

This module MUST be imported and called BEFORE any `from nltk.corpus import ...`
statements.  NLTK corpus readers snapshot their search paths at import time, so
the download directory must already be on `nltk.data.path` when the corpus
objects are first created.
"""

import os
import logging

import nltk          # safe — only core package, no corpus objects
import nltk.data     # ensure path list exists

logger = logging.getLogger(__name__)

# Canonical data directory — used across the whole project
NLTK_DATA_DIR: str = os.path.join(os.path.expanduser("~"), "nltk_data")


def ensure_nltk_resources() -> None:
    """Download required NLTK data packages if missing.

    This function:
    1. Creates a dedicated download directory.
    2. Registers it on ``nltk.data.path`` **before** any corpus reader
       snapshots its search paths.
    3. Sets the ``NLTK_DATA`` environment variable (overwriting any
       stale or empty value).
    4. Downloads missing packages.
    """
    # 1 — ensure directory exists
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    # 2 — register on nltk's search path *immediately*
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

    # 3 — force env var (setdefault won't override empty strings)
    os.environ["NLTK_DATA"] = NLTK_DATA_DIR

    # 4 — download any missing packages
    resources = [
        ("punkt",     "tokenizers"),
        ("punkt_tab", "tokenizers"),
        ("stopwords", "corpora"),
        ("wordnet",   "corpora"),
        ("omw-1.4",   "corpora"),
    ]

    for pkg, kind in resources:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            try:
                logger.info("Downloading NLTK resource: %s", pkg)
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception as e:
                logger.warning("Failed to download NLTK resource %s: %s", pkg, e)

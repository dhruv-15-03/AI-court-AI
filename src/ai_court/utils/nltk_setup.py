import os
import nltk
import logging

logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    NLTK_DATA_DIR = os.path.join(os.path.expanduser("~"), ".nltk_data")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", NLTK_DATA_DIR)

    resources = [
        ("punkt", "tokenizers"),
        ("punkt_tab", "tokenizers"),
        ("stopwords", "corpora"),
        ("wordnet", "corpora"),
        ("omw-1.4", "corpora")
    ]

    for pkg, kind in resources:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {pkg}")
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {pkg}: {e}")

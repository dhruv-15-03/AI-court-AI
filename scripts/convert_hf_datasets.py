"""
Convert downloaded HuggingFace datasets into our CSV training format.

Handles multiple dataset structures and maps labels to 11-class ontology.
Run after datasets have been downloaded.

Usage:
    python scripts/convert_hf_datasets.py
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_hf")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

from ai_court.model.preprocessor import TextPreprocessor
tp = TextPreprocessor()


def extract_judgment_section(text: str) -> str:
    """Extract the operative order from full judgment text."""
    markers = [
        r"(?:ORDER|JUDGMENT|CONCLUSION|HELD)\s*[:\n]([\s\S]{50,3000}?)(?:\n\n\n|\Z)",
        r"(?:the\s+appeal\s+is|the\s+petition\s+is|we\s+order|it\s+is\s+ordered)([\s\S]{30,2000}?)(?:\n\n|\Z)",
    ]
    for pattern in markers:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            return m.group(0).strip()[:3000]
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 30]
    return " ".join(lines[-15:])[:3000]


def create_summary(text: str, max_len: int = 3000) -> str:
    """Build a rich summary: intro + middle sample + conclusion."""
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 20]
    if len(lines) < 5:
        return text[:max_len]
    total = len(lines)
    intro = " ".join(lines[:max(5, total // 4)])[:1200]
    mid = " ".join(lines[total//3:2*total//3:max(1, total//20)])[:800]
    concl = extract_judgment_section(text)[:1000]
    return f"{intro}\n\n[...]\n\n{mid}\n\n[...]\n\n{concl}"[:max_len]


def smart_label_from_text(text: str) -> str:
    """Try to extract a label from judgment text using our normalizer."""
    judgment = extract_judgment_section(text)
    label = tp.normalize_outcome(judgment)
    return label


def convert_dataset(name: str):
    """Load a HF dataset and convert to our CSV format."""
    from datasets import load_dataset

    logger.info(f"Loading dataset: {name}")
    try:
        ds = load_dataset(name)
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        return 0

    records = []
    safe_name = name.replace("/", "_").replace("-", "_")

    for split_name, split_data in ds.items():
        logger.info(f"  Processing {split_name}: {len(split_data)} rows")
        cols = split_data.column_names
        logger.info(f"    Columns: {cols}")

        for i, item in enumerate(split_data):
            # Find text field
            text = ""
            for key in ["text", "doc", "document", "case_text", "judgment_text",
                         "sentence1", "content", "body", "facts"]:
                if key in item and item[key] and len(str(item[key])) > 50:
                    text = str(item[key])
                    break

            if not text:
                # For datasets with separate columns, concatenate available text
                text_parts = []
                for key in cols:
                    val = item.get(key)
                    if isinstance(val, str) and len(val) > 50:
                        text_parts.append(val)
                text = "\n\n".join(text_parts)

            if len(text.strip()) < 100:
                continue

            # Label
            label = smart_label_from_text(text)
            # If still Other, check if dataset has a label column
            if label == "Other":
                for key in ["label", "decision", "outcome", "prediction"]:
                    if key in item:
                        raw = item[key]
                        if isinstance(raw, int):
                            label = "Relief Granted/Convicted" if raw == 0 else "Relief Denied/Dismissed"
                        elif isinstance(raw, str):
                            label = tp.normalize_outcome(raw)
                        break

            # Title
            title = ""
            for key in ["title", "case_name", "name", "case_id"]:
                if key in item and item[key]:
                    title = str(item[key])[:300]
                    break
            if not title:
                title = f"{safe_name}_{split_name}_{i}"

            summary = create_summary(text)
            judgment = extract_judgment_section(text)

            records.append({
                "title": title,
                "url": item.get("url", ""),
                "case_summary": summary,
                "judgment": judgment,
                "court": item.get("court", "Supreme Court of India"),
                "date": str(item.get("date", "")),
                "category": "mixed_legal",
                "query_source": f"HF_{name}_{split_name}",
            })

            if (i + 1) % 2000 == 0:
                logger.info(f"    Processed {i+1}/{len(split_data)}")

    if not records:
        logger.warning(f"No usable records from {name}")
        return 0

    df = pd.DataFrame(records)
    output = RAW_DIR / f"hf_{safe_name}.csv"
    df.to_csv(output, index=False)
    logger.info(f"  Saved {len(df)} cases to {output}")
    return len(df)


def main():
    # List of datasets that we know exist and are accessible
    datasets = [
        "ninadn/indian-legal",
        "anuragiiser/ILDC_expert",
        "sujantkumarkv/indian_legal_corpus",
        "AnuragB/Indian-legal",
        "Mayank-02/indian_legal_text",
        "Yashaswat/Indian-Legal-Text-ABS",
    ]

    total = 0
    for name in datasets:
        try:
            count = convert_dataset(name)
            total += count
        except Exception as e:
            logger.error(f"Error with {name}: {e}")

    logger.info(f"\nTotal cases from HuggingFace: {total}")
    logger.info("Run: python scripts/check_distribution.py to see updated distribution")


if __name__ == "__main__":
    main()

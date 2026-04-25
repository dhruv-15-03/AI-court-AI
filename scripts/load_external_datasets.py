"""
Track B Dataset Loader — ILDC and HLDC Integration
====================================================
Integrates two publicly available Indian legal datasets into the
ai_court training pipeline by mapping their labels to the 3-class schema:

  Relief Granted/Convicted  |  Relief Denied/Dismissed  |  Acquittal/Conviction Overturned

License: Both datasets are CC BY-NC 4.0 (IIT Kanpur / Exploration Lab).
         Safe for research / non-commercial use only.

Datasets to download manually before running:
  1. ILDC (multi): https://github.com/Exploration-Lab/CAIL
     Expected path: data/external/ildc/
     Files: train.json, dev.json, test.json

  2. HLDC (bail): https://github.com/Exploration-Lab/IL-TUR (BAIL task)
     Expected path: data/external/hldc/
     Files: train.json, dev.json, test.json

Usage (from project root):
    python scripts/load_external_datasets.py
    python scripts/load_external_datasets.py --ildc-only
    python scripts/load_external_datasets.py --hldc-only
    python scripts/load_external_datasets.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("external_loader")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "external_cases.csv"

# ── Label Mappings ────────────────────────────────────────────────────────────
# ILDC: 0 = petition accepted (court grants petition) → Relief Granted
#       1 = petition rejected (court dismisses petition) → Relief Denied
ILDC_LABEL_MAP = {
    0: "Relief Granted/Convicted",
    1: "Relief Denied/Dismissed",
    "accepted": "Relief Granted/Convicted",
    "rejected": "Relief Denied/Dismissed",
    "allowed": "Relief Granted/Convicted",
    "dismissed": "Relief Denied/Dismissed",
}

# HLDC: bail prediction — binary outcome
HLDC_LABEL_MAP = {
    "bail_granted": "Relief Granted/Convicted",
    "bail_denied": "Relief Denied/Dismissed",
    1: "Relief Granted/Convicted",
    0: "Relief Denied/Dismissed",
    "granted": "Relief Granted/Convicted",
    "denied": "Relief Denied/Dismissed",
}

REQUIRED_COLUMNS = ["title", "case_summary", "judgment", "case_type", "source"]


def _load_json_split(path: Path) -> list[dict]:
    """Load a JSON file that may be a list or dict-of-list."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # Common format: {"data": [...]} or {"train": [...]}
        for key in ("data", "train", "dev", "test", "cases"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError(f"Unexpected JSON structure in {path}")


def load_ildc(ildc_dir: Path, dry_run: bool = False) -> pd.DataFrame:
    """
    Load ILDC (Indian Legal Documents Corpus) splits and convert to ai_court schema.

    ILDC record example:
        {
          "doc": "IN THE SUPREME COURT OF INDIA ...",
          "label": 0,          # 0=accepted, 1=rejected
          "title": "State v. XYZ"  # optional
        }
    """
    records = []
    for split in ("train.json", "dev.json", "test.json"):
        rows = _load_json_split(ildc_dir / split)
        logger.info(f"ILDC {split}: {len(rows)} records")
        records.extend(rows)

    if not records:
        logger.warning("No ILDC records loaded — check data/external/ildc/ directory")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rows_out = []
    skipped = 0
    for rec in records:
        # Get text — try common field names
        text = rec.get("doc") or rec.get("text") or rec.get("case_text") or rec.get("judgment_text") or ""
        if not text or len(text.strip()) < 50:
            skipped += 1
            continue

        raw_label = rec.get("label")
        if raw_label is None:
            raw_label = rec.get("outcome") or rec.get("decision")

        label = ILDC_LABEL_MAP.get(raw_label) or ILDC_LABEL_MAP.get(str(raw_label).lower().strip())
        if label is None:
            skipped += 1
            continue

        title = rec.get("title") or rec.get("case_name") or "ILDC Case"
        rows_out.append({
            "title": str(title)[:300],
            "case_summary": text[:3000],      # cap to keep CSV manageable
            "judgment": label,
            "case_type": rec.get("case_type", "Constitutional"),
            "source": "ILDC",
            "url": rec.get("url", ""),
        })

    logger.info(f"ILDC: {len(rows_out)} usable rows, {skipped} skipped (no text or unknown label)")
    return pd.DataFrame(rows_out)


def load_hldc(hldc_dir: Path, dry_run: bool = False) -> pd.DataFrame:
    """
    Load HLDC (bail prediction) splits and convert to ai_court schema.

    HLDC record example:
        {
          "text": "This is a bail application ...",
          "label": "bail_granted"
        }
    """
    records = []
    for split in ("train.json", "dev.json", "test.json"):
        rows = _load_json_split(hldc_dir / split)
        logger.info(f"HLDC {split}: {len(rows)} records")
        records.extend(rows)

    if not records:
        logger.warning("No HLDC records loaded — check data/external/hldc/ directory")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rows_out = []
    skipped = 0
    for rec in records:
        text = rec.get("text") or rec.get("doc") or rec.get("bail_text") or ""
        if not text or len(text.strip()) < 50:
            skipped += 1
            continue

        raw_label = rec.get("label")
        label = HLDC_LABEL_MAP.get(raw_label) or HLDC_LABEL_MAP.get(str(raw_label).lower().strip())
        if label is None:
            skipped += 1
            continue

        rows_out.append({
            "title": rec.get("title", rec.get("case_name", "HLDC Bail Case"))[:300],
            "case_summary": text[:3000],
            "judgment": label,
            "case_type": "Criminal",      # bail is always criminal
            "source": "HLDC",
            "url": rec.get("url", ""),
        })

    logger.info(f"HLDC: {len(rows_out)} usable rows, {skipped} skipped")
    return pd.DataFrame(rows_out)


def merge_with_existing(new_df: pd.DataFrame, existing_csv: Path) -> pd.DataFrame:
    """Merge new rows with the main training CSV, deduplicating on case_summary."""
    if existing_csv.exists():
        existing = pd.read_csv(existing_csv, dtype=str)
        logger.info(f"Existing dataset: {len(existing)} rows")
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicate on first 200 chars of case_summary
    combined["_dedup_key"] = combined["case_summary"].str[:200].str.strip().str.lower()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    logger.info(f"Dedup: {before} → {len(combined)} rows (removed {before - len(combined)} duplicates)")
    return combined


def print_class_distribution(df: pd.DataFrame, label: str = "Dataset") -> None:
    dist = df["judgment"].value_counts()
    logger.info(f"\n{label} class distribution:")
    for cls, cnt in dist.items():
        pct = cnt / len(df) * 100
        logger.info(f"  {cls}: {cnt:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Load ILDC/HLDC external datasets")
    parser.add_argument("--ildc-dir", default="data/external/ildc", help="Path to ILDC data directory")
    parser.add_argument("--hldc-dir", default="data/external/hldc", help="Path to HLDC data directory")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--ildc-only", action="store_true")
    parser.add_argument("--hldc-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report counts without writing output")
    parser.add_argument("--merge", action="store_true", help="Merge with data/processed/all_cases_db.csv")
    args = parser.parse_args()

    ildc_dir = Path(args.ildc_dir)
    hldc_dir = Path(args.hldc_dir)
    frames = []

    if not args.hldc_only:
        ildc_df = load_ildc(ildc_dir, dry_run=args.dry_run)
        if not ildc_df.empty:
            frames.append(ildc_df)

    if not args.ildc_only:
        hldc_df = load_hldc(hldc_dir, dry_run=args.dry_run)
        if not hldc_df.empty:
            frames.append(hldc_df)

    if not frames:
        logger.error(
            "No data loaded from either dataset.\n"
            "Download steps:\n"
            "  ILDC: git clone https://github.com/Exploration-Lab/CAIL data/external/ildc\n"
            "  HLDC: Download BAIL task data from https://cse.iitk.ac.in/users/ashutoshm/IL-TUR/\n"
            "        and place train.json/dev.json/test.json in data/external/hldc/"
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"\nTotal external rows: {len(combined):,}")
    print_class_distribution(combined, "External data")

    if args.dry_run:
        logger.info("[dry-run] No files written.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge:
        master = PROJECT_ROOT / "data" / "processed" / "all_cases_db.csv"
        combined = merge_with_existing(combined, master)
        out_path = master.parent / "all_cases_merged.csv"
        logger.info(f"Merged output will be written to {out_path}")

    combined.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Written {len(combined):,} rows → {out_path}")
    print_class_distribution(combined, "Final output")


if __name__ == "__main__":
    main()

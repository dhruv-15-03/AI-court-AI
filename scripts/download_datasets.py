"""
Download External Legal Datasets for AI Court System
=====================================================

Downloads and prepares:
1. ILDC (Indian Legal Documents Corpus) - ~35K Supreme Court cases
   Source: https://github.com/Exploration-Lab/ILDC
   
2. HLDC (Hindi Legal Document Corpus / Bail Prediction) 
   Source: https://github.com/Exploration-Lab/IL-TUR

3. SCI dataset (Supreme Court of India judgments)
   Source: https://huggingface.co/datasets/opennyaiorg/InLegalBert_Supreme_Court_of_India

Usage:
    python scripts/download_datasets.py              # Download all
    python scripts/download_datasets.py --ildc       # Only ILDC
    python scripts/download_datasets.py --hldc       # Only HLDC  
    python scripts/download_datasets.py --hf         # Only HuggingFace datasets
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_datasets")


def check_git():
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_ildc():
    """Download ILDC from GitHub."""
    ildc_dir = EXTERNAL_DIR / "ildc"
    if ildc_dir.exists() and any(ildc_dir.glob("*.json")):
        logger.info("ILDC already downloaded at %s", ildc_dir)
        return True

    ildc_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ILDC dataset...")

    if check_git():
        # Clone the repo (sparse checkout for just the data)
        tmp_dir = EXTERNAL_DIR / "_ildc_tmp"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/Exploration-Lab/ILDC.git",
                 str(tmp_dir)],
                check=True, capture_output=True, text=True, timeout=300
            )
            # Move JSON files to ildc_dir
            for json_file in tmp_dir.rglob("*.json"):
                dest = ildc_dir / json_file.name
                if not dest.exists():
                    shutil.copy2(json_file, dest)
                    logger.info("  Copied: %s", json_file.name)

            # Also check for CSV files
            for csv_file in tmp_dir.rglob("*.csv"):
                dest = ildc_dir / csv_file.name
                if not dest.exists():
                    shutil.copy2(csv_file, dest)
                    logger.info("  Copied: %s", csv_file.name)

            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("ILDC downloaded to %s", ildc_dir)
            return True
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out. Try manually: git clone https://github.com/Exploration-Lab/ILDC.git %s", ildc_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except subprocess.CalledProcessError as e:
            logger.error("Git clone failed: %s", e.stderr[:200] if e.stderr else str(e))
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        logger.info(
            "Git not found. Please download ILDC manually:\n"
            "  1. Go to https://github.com/Exploration-Lab/ILDC\n"
            "  2. Download the repository\n"
            "  3. Place train.json, dev.json, test.json in %s", ildc_dir
        )

    return False


def download_hldc():
    """Download HLDC bail prediction dataset."""
    hldc_dir = EXTERNAL_DIR / "hldc"
    if hldc_dir.exists() and any(hldc_dir.glob("*.json")):
        logger.info("HLDC already downloaded at %s", hldc_dir)
        return True

    hldc_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HLDC dataset...")

    if check_git():
        tmp_dir = EXTERNAL_DIR / "_hldc_tmp"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/Exploration-Lab/IL-TUR.git",
                 str(tmp_dir)],
                check=True, capture_output=True, text=True, timeout=300
            )
            # Look for BAIL task data
            for json_file in tmp_dir.rglob("*.json"):
                if any(keyword in str(json_file).lower() for keyword in ["bail", "hldc", "train", "dev", "test"]):
                    dest = hldc_dir / json_file.name
                    if not dest.exists():
                        shutil.copy2(json_file, dest)
                        logger.info("  Copied: %s", json_file.name)

            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("HLDC downloaded to %s", hldc_dir)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error("Download failed: %s", str(e)[:200])
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        logger.info(
            "Git not found. Please download HLDC manually:\n"
            "  1. Go to https://github.com/Exploration-Lab/IL-TUR\n"
            "  2. Download BAIL task data\n"
            "  3. Place JSON files in %s", hldc_dir
        )

    return False


def download_hf_datasets():
    """Download datasets from HuggingFace using the datasets library."""
    hf_dir = EXTERNAL_DIR / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("Installing datasets library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True, capture_output=True)
        from datasets import load_dataset

    # Dataset 1: Indian Legal NER / judgment prediction
    datasets_to_try = [
        {
            "name": "opennyaiorg/indian_legal_documents",
            "output": hf_dir / "indian_legal_docs.json",
            "split": "train",
        },
        {
            "name": "Exploration-Lab/ILDC-multi",
            "output": hf_dir / "ildc_multi.json",
            "split": "train",
        },
    ]

    for ds_config in datasets_to_try:
        output = ds_config["output"]
        if output.exists():
            logger.info("Already downloaded: %s", output.name)
            continue

        try:
            logger.info("Downloading %s from HuggingFace...", ds_config["name"])
            dataset = load_dataset(ds_config["name"], split=ds_config.get("split", "train"))
            
            # Convert to list of dicts and save
            records = []
            for item in dataset:
                records.append(dict(item))

            with open(output, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)

            logger.info("Saved %d records to %s", len(records), output)
        except Exception as e:
            logger.warning("Could not download %s: %s", ds_config["name"], str(e)[:200])

    return True


def main():
    parser = argparse.ArgumentParser(description="Download external legal datasets")
    parser.add_argument("--ildc", action="store_true")
    parser.add_argument("--hldc", action="store_true")
    parser.add_argument("--hf", action="store_true", help="HuggingFace datasets")
    parser.add_argument("--all", action="store_true", default=True)
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    download_all = not (args.ildc or args.hldc or args.hf)

    results = {}
    if download_all or args.ildc:
        results["ILDC"] = download_ildc()
    if download_all or args.hldc:
        results["HLDC"] = download_hldc()
    if download_all or args.hf:
        results["HuggingFace"] = download_hf_datasets()

    print("\n" + "=" * 50)
    print("  Download Results:")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ Failed/Manual needed"
        print(f"    {name}: {status}")
    print("=" * 50)
    print(f"\nData directory: {EXTERNAL_DIR}")
    print(f"\nNext: python scripts/harvest_100k.py --external-only")


if __name__ == "__main__":
    main()

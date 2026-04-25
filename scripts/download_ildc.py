"""
Fast ILDC Download — Get ~35K Indian legal cases from HuggingFace in minutes.

The ILDC (Indian Legal Documents Corpus) contains Supreme Court of India judgments
with binary prediction labels (accepted/rejected).

Source: https://huggingface.co/datasets/Exploration-Lab/ILDC-multi

This script:
1. Downloads ILDC from HuggingFace Hub (no git clone needed)
2. Converts to our CSV format with proper label mapping
3. Extracts richer summaries from the full judgment text
4. Maps binary labels to our 11-class ontology using text analysis
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ildc_download")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Import our outcome normalizer for smart label extraction
from ai_court.model.preprocessor import TextPreprocessor
tp = TextPreprocessor()


def extract_judgment_section(text: str) -> str:
    """Extract the operative/concluding part of a judgment."""
    # Look for explicit markers
    markers = [
        r"(?:ORDER|JUDGMENT|CONCLUSION|HELD|OPERATIVE\s+ORDER)\s*[:\n]([\s\S]{50,3000}?)(?:\n\n\n|\Z)",
        r"(?:we\s+(?:hold|direct|order|allow|dismiss)|it\s+is\s+(?:ordered|directed|held)|the\s+(?:appeal|petition|writ)\s+is)\s+([\s\S]{50,2000}?)(?:\n\n|\Z)",
        r"(?:for\s+the\s+(?:foregoing|above)\s+reasons|in\s+(?:view|light)\s+of)[,\s]+([\s\S]{50,2000}?)(?:\n\n|\Z)",
    ]
    sections = []
    for pattern in markers:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            sections.append(m.group(0).strip())

    if sections:
        return " ".join(sections[-3:])[:3000]

    # Fallback: last 20 meaningful lines
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 30]
    return " ".join(lines[-20:])[:3000]


def smart_label(text: str, binary_label: int) -> str:
    """
    Map ILDC binary label + text analysis to our 11-class ontology.
    
    ILDC labels: 0 = petition accepted, 1 = petition rejected
    But we can do better using the actual text of the judgment.
    """
    # First try our text-based normalizer on the judgment section
    judgment = extract_judgment_section(text)
    text_label = tp.normalize_outcome(judgment)
    
    if text_label != "Other":
        return text_label
    
    # Fallback: use the binary label
    if binary_label == 0:
        return "Relief Granted/Convicted"
    else:
        return "Relief Denied/Dismissed"


def create_summary(text: str, max_len: int = 3000) -> str:
    """Create a rich summary from full judgment text."""
    lines = text.split("\n")
    clean = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]
    
    if len(clean) < 10:
        return text[:max_len]
    
    total = len(clean)
    intro_end = max(5, total // 4)
    intro = " ".join(clean[:intro_end])[:1200]
    
    mid_start = total // 3
    mid_end = 2 * total // 3
    step = max(1, (mid_end - mid_start) // 8)
    middle = " ".join(clean[mid_start:mid_end:step])[:800]
    
    judgment = extract_judgment_section(text)[:1000]
    
    return f"{intro}\n\n[...]\n\n{middle}\n\n[...conclusion...]\n\n{judgment}"[:max_len]


def main():
    logger.info("Loading HuggingFace datasets library...")
    from datasets import load_dataset
    
    # Try multiple dataset names — HF naming varies
    dataset_names = [
        "Exploration-Lab/ILDC-multi",
        "Exploration-Lab/ILDC",
        "saibo/Indian_Legal_Documents_Corpus",
    ]
    
    ds = None
    for name in dataset_names:
        try:
            logger.info(f"Trying to download: {name}")
            ds = load_dataset(name, trust_remote_code=True)
            logger.info(f"Success! Downloaded {name}")
            break
        except Exception as e:
            logger.warning(f"  Failed: {str(e)[:100]}")
    
    if ds is None:
        logger.error("Could not download ILDC from any source. Trying alternative approach...")
        # Try the NyAI dataset which is also Indian Supreme Court cases
        try:
            logger.info("Trying: kiddothe2b/indian_legal_judgements")
            ds = load_dataset("kiddothe2b/indian_legal_judgements", trust_remote_code=True)
            logger.info("Success!")
        except Exception as e:
            logger.error(f"All download attempts failed: {e}")
            logger.info("\nManual alternative:")
            logger.info("  1. Go to https://huggingface.co/datasets/Exploration-Lab/ILDC-multi")
            logger.info("  2. Download the files")
            logger.info("  3. Place in data/external/ildc/")
            logger.info("  4. Run: python scripts/harvest_100k.py --external-only")
            return
    
    # Process all splits
    logger.info(f"Dataset splits: {list(ds.keys())}")
    
    import pandas as pd
    all_records = []
    
    for split_name, split_data in ds.items():
        logger.info(f"Processing split '{split_name}': {len(split_data)} records")
        
        for i, item in enumerate(split_data):
            # Find the text field
            text = ""
            for key in ["text", "doc", "document", "case_text", "judgment_text", "sentence1"]:
                if key in item and item[key]:
                    text = str(item[key])
                    break
            
            if not text or len(text.strip()) < 100:
                continue
            
            # Find label
            label_val = None
            for key in ["label", "decision", "outcome", "prediction"]:
                if key in item:
                    label_val = item[key]
                    break
            
            if label_val is None:
                label_val = 0  # Default to accepted
            
            # Smart label mapping
            outcome = smart_label(text, int(label_val) if isinstance(label_val, (int, float)) else 0)
            
            # Create summary
            summary = create_summary(text)
            judgment = extract_judgment_section(text)
            
            # Title
            title = ""
            for key in ["title", "case_name", "name"]:
                if key in item and item[key]:
                    title = str(item[key])[:300]
                    break
            if not title:
                title = f"ILDC_{split_name}_{i}"
            
            all_records.append({
                "title": title,
                "url": item.get("url", ""),
                "case_summary": summary,
                "judgment": judgment,
                "court": "Supreme Court of India",
                "date": item.get("date", ""),
                "category": "constitutional",
                "query_source": f"ILDC_{split_name}",
            })
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1}/{len(split_data)}")
    
    if not all_records:
        logger.error("No usable records found in dataset!")
        return
    
    # Save as CSV
    df = pd.DataFrame(all_records)
    output_path = RAW_DIR / "external_ildc.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} cases to {output_path}")
    
    # Show label distribution
    labels = df["judgment"].apply(tp.normalize_outcome)
    dist = labels.value_counts()
    logger.info("\nLabel distribution from text analysis:")
    for label, count in dist.head(15).items():
        logger.info(f"  {label:45s} {count:6d}")
    
    logger.info(f"\nDone! {len(df)} cases added. Run check_distribution.py to see full stats.")


if __name__ == "__main__":
    main()

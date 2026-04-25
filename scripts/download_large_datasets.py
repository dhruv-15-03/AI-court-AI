"""
Download large Indian court datasets from HuggingFace.
Focus on the biggest datasets to reach 100K quickly.
"""
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
logger = logging.getLogger("dl_large")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

from ai_court.model.preprocessor import TextPreprocessor
tp = TextPreprocessor()


def extract_judgment(text: str) -> str:
    markers = [
        r"(?:ORDER|JUDGMENT|CONCLUSION|HELD)\s*[:\n]([\s\S]{50,3000}?)(?:\n\n\n|\Z)",
        r"(?:the\s+appeal\s+is|the\s+petition\s+is|we\s+order)([\s\S]{30,2000}?)(?:\n\n|\Z)",
    ]
    for p in markers:
        for m in re.finditer(p, text, re.IGNORECASE):
            return m.group(0).strip()[:3000]
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 30]
    return " ".join(lines[-15:])[:3000]


def create_summary(text: str, max_len: int = 3000) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 20]
    if len(lines) < 5:
        return text[:max_len]
    total = len(lines)
    intro = " ".join(lines[:max(5, total // 4)])[:1200]
    mid = " ".join(lines[total//3:2*total//3:max(1, total//20)])[:800]
    concl = extract_judgment(text)[:1000]
    return f"{intro}\n\n[...]\n\n{mid}\n\n[...]\n\n{concl}"[:max_len]


def process_dataset(name: str, output_name: str):
    from datasets import load_dataset
    
    output_path = RAW_DIR / f"hf_{output_name}.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path)
        logger.info(f"Already have {len(existing)} rows in {output_path.name}, skipping")
        return len(existing)
    
    logger.info(f"Loading: {name}")
    try:
        ds = load_dataset(name)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 0
    
    records = []
    for split_name, split_data in ds.items():
        cols = split_data.column_names
        logger.info(f"  {split_name}: {len(split_data)} rows, cols: {cols[:10]}")
        
        for i, item in enumerate(split_data):
            # Find text
            text = ""
            for key in ["text", "doc", "document", "Text", "case_text", "judgment_text",
                         "content", "body", "sentence1", "judgment", "judgement", 
                         "facts", "case", "description"]:
                if key in item and item[key] and len(str(item[key])) > 100:
                    text = str(item[key])
                    break
            
            if not text:
                # Concatenate all string fields > 50 chars
                parts = []
                for key in cols:
                    val = item.get(key)
                    if isinstance(val, str) and len(val) > 50:
                        parts.append(val)
                text = "\n\n".join(parts)
            
            if len(text.strip()) < 100:
                continue
            
            # Label from text
            label = tp.normalize_outcome(extract_judgment(text))
            if label == "Other":
                # Check for explicit label column
                for key in ["label", "decision", "outcome", "prediction"]:
                    if key in item and item[key] is not None:
                        raw = item[key]
                        if isinstance(raw, int):
                            label = "Relief Granted/Convicted" if raw == 0 else "Relief Denied/Dismissed"
                        elif isinstance(raw, str) and raw.strip():
                            mapped = tp.normalize_outcome(raw)
                            if mapped != "Other":
                                label = mapped
                        break
            
            # Title
            title = ""
            for key in ["title", "case_name", "name", "case_id", "Case ID",
                         "case_title", "heading"]:
                if key in item and item[key]:
                    title = str(item[key])[:300]
                    break
            if not title:
                title = f"{output_name}_{split_name}_{i}"
            
            summary = create_summary(text)
            judgment = extract_judgment(text)
            
            records.append({
                "title": title,
                "url": str(item.get("url", item.get("URL", ""))),
                "case_summary": summary,
                "judgment": judgment,
                "court": str(item.get("court", item.get("Court", ""))),
                "date": str(item.get("date", item.get("Date", ""))),
                "category": "mixed_legal",
                "query_source": f"HF_{name}_{split_name}",
            })
            
            if (i + 1) % 5000 == 0:
                logger.info(f"    {i+1}/{len(split_data)} processed")
    
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"  Saved {len(df)} cases to {output_path.name}")
        return len(df)
    return 0


def main():
    # Priority: biggest datasets first
    datasets = [
        ("santoshtyss/indian_courts_cases", "santoshtyss_court_cases"),
        ("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", "vihaannnn_sc_chunked"),
        ("labofsahil/Indian-Supreme-Court-Judgments", "labofsahil_sc"),
        ("Rahul1872/Indian-Supreme-Court-Judgments", "rahul_sc"),
        ("Immanuel30303/Indian-High-Court-Judgments-all", "immanuel_hc"),
        ("maheshCoder/indian_court_cases", "mahesh_court"),
        ("debkanchan/supreme-court-of-india-judgements", "debkanchan_sc"),
        ("rishiai/indian-court-judgements-and-its-summaries", "rishiai_court"),
        ("engineersaloni159/INS-indian-legal-dataset", "saloni_legal"),
        ("Prarabdha/indian-legal-data", "prarabdha_legal"),
    ]
    
    total_new = 0
    for name, output in datasets:
        try:
            count = process_dataset(name, output)
            total_new += count
            logger.info(f"  Running total from HF: {total_new}")
        except Exception as e:
            logger.error(f"Error with {name}: {e}")
    
    logger.info(f"\nTotal new cases from HuggingFace: {total_new}")
    
    # Count grand total
    grand = 0
    for csv in RAW_DIR.glob("*.csv"):
        try:
            grand += len(pd.read_csv(csv))
        except:
            pass
    from pathlib import Path
    enriched_dir = PROJECT_ROOT / "data" / "raw_enriched"
    if enriched_dir.exists():
        for csv in enriched_dir.glob("*.csv"):
            try:
                grand += len(pd.read_csv(csv))
            except:
                pass
    logger.info(f"GRAND TOTAL across all CSVs: {grand}")


if __name__ == "__main__":
    main()

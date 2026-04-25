"""Check current class distribution across all data."""
import sys, os, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from ai_court.model.preprocessor import TextPreprocessor
from ai_court.ontology import map_coarse_label
import pandas as pd

tp = TextPreprocessor()

# Filename -> label inference (mirrors DataLoader._infer_label_from_filename)
FILENAME_LABEL_MAP = [
    (["bail_granted", "bail_allowed", "anticipatory_bail"], "Bail Granted"),
    (["bail_denied", "bail_rejected", "bail_refused"], "Bail Denied"),
    (["acquittal", "acquitted", "conviction_overturned"], "Acquittal/Conviction Overturned"),
    (["conviction_upheld", "conviction_confirmed", "appeal_dismissed"], "Conviction Upheld/Appeal Dismissed"),
    (["conviction", "convicted", "ipc_302_conviction", "ipc_376_conviction", "pocso_conviction"], "Relief Granted/Convicted"),
    (["quashed", "quashing", "charges_quashed", "fir_quashed", "section_482"], "Charges/Proceedings Quashed"),
    (["sentence_reduced", "sentence_modified", "commuted"], "Sentence Reduced/Modified"),
    (["remanded", "sent_back", "remand"], "Case Remanded/Sent Back"),
    (["withdrawn", "not_pressed"], "Petition Withdrawn/Dismissed as Withdrawn"),
    (["allowed", "relief_granted", "writ_allowed", "petition_allowed"], "Relief Granted/Convicted"),
    (["dismissed", "relief_denied", "petition_dismissed"], "Relief Denied/Dismissed"),
]

def infer_from_name(basename):
    bn = basename.lower()
    for patterns, label in FILENAME_LABEL_MAP:
        for p in patterns:
            if p in bn:
                return label
    return "Other"

csvs = sorted(glob.glob("data/raw/*.csv") + glob.glob("data/raw_enriched/*.csv"))
print(f"Scanning {len(csvs)} CSV files...")

frames = []
for path in csvs:
    try:
        df = pd.read_csv(path)
        jcol = None
        for c in ["judgment", "judgement", "outcome", "refined_label"]:
            if c in df.columns:
                jcol = c
                break
        if not jcol:
            continue
        
        basename = os.path.splitext(os.path.basename(path))[0]
        labels = df[jcol].dropna().astype(str).apply(tp.normalize_outcome).apply(lambda x: map_coarse_label(x)[0])
        
        # Apply filename fallback for "Other" rows
        filename_label = infer_from_name(basename)
        if filename_label != "Other":
            labels = labels.where(labels != "Other", filename_label)
        
        frames.append(labels)
    except Exception as e:
        pass

all_labels = pd.concat(frames, ignore_index=True)
print(f"Total rows processed: {len(all_labels)}")

dist = all_labels.value_counts()
print()
for label, count in dist.items():
    pct = count / len(all_labels) * 100
    bar = "#" * int(pct / 2)
    print(f"  {label:45s} {count:6d}  ({pct:5.1f}%)  {bar}")

num_classes = all_labels.nunique()
other_count = dist.get("Other", 0)
non_other = len(all_labels) - other_count
print(f"\nTotal classes: {num_classes}")
print(f"Other: {other_count} ({other_count/len(all_labels)*100:.1f}%)")
print(f"Labeled (non-Other): {non_other} ({non_other/len(all_labels)*100:.1f}%)")

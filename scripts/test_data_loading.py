"""Test data loading and class distribution."""
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

os.environ["ENABLE_SMOTE"] = "1"
os.environ["MIN_CLASS_SAMPLES"] = "3"

from ai_court.model.legal_case_classifier import LegalCaseClassifier

clf = LegalCaseClassifier()
csv_files = []
for d in ["data/raw", "data/raw_enriched"]:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".csv"):
                csv_files.append(os.path.join(d, f))

print(f"CSV files: {len(csv_files)}")
df = clf.data_loader.load_data(csv_files)
print(f"Loaded: {len(df)} rows")
stats = clf.data_loader.analyze_dataset(df)
print(f"Classes: {stats['num_classes']}")
print("Distribution:")
for k, v in sorted(stats["judgement_distribution"].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

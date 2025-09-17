"""
Train the boosted RandomForest legal case outcome model and persist it to models/legal_case_classifier.pkl.

Usage (from project root):
  - Option A: Use existing auto-collector (fallback): python train_model.py
  - Option B: Build processed dataset then train:
      python -m src.ai_court.data.prepare_dataset
      python train_model.py
"""

import os
from src.ai_court.model.legal_case_classifier import main as auto_main

if __name__ == "__main__":
    # If processed dataset exists, temporarily point the trainer at it by environment variable (optional future enhancement)
    processed = os.path.join("data", "processed", "all_cases.csv")
    # Currently, main() auto-collects from candidate CSVs in repo; we could enhance to read env var later if needed.
    auto_main()

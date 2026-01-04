"""
Train the boosted RandomForest legal case outcome model and persist it to models/legal_case_classifier.pkl.

Usage (from project root):
  - Option A: python scripts/train_model.py
  - Option B: Build processed dataset then train:
    python -m src.ai_court.data.prepare_dataset
    python scripts/train_model.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

try:
  from ai_court.model.legal_case_classifier import main as auto_main
except ModuleNotFoundError as e:
  raise ModuleNotFoundError(
    "Could not import ai_court from src/. Please run from project root or ensure 'src' is on PYTHONPATH."
  ) from e


if __name__ == "__main__":
  auto_main()

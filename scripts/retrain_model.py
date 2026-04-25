"""Retrain the model with all available data and full 11-class support.

Usage:
    python scripts/retrain_model.py                       # Train with all data, SMOTE on
    python scripts/retrain_model.py --no-smote            # Disable SMOTE
    python scripts/retrain_model.py --rebuild-dataset     # Rebuild all_cases.csv first
    python scripts/retrain_model.py --rebuild-search      # Also rebuild search index

This script:
1. Optionally rebuilds data/processed/all_cases.csv from raw CSVs
2. Loads and normalizes data through the full pipeline (11 outcome classes)
3. Trains with SMOTE enabled for minority class augmentation
4. Evaluates and saves metrics
5. Optionally rebuilds the TF-IDF search index
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain AI Court model")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild all_cases.csv from raw CSVs")
    parser.add_argument("--rebuild-search", action="store_true", help="Rebuild TF-IDF search index after training")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE resampling")
    parser.add_argument("--min-class-samples", type=int, default=5, help="Min samples per class before merging into Other")
    args = parser.parse_args()

    # Set environment
    if not args.no_smote:
        os.environ["ENABLE_SMOTE"] = "1"
        os.environ["SMOTE_SAMPLING_STRATEGY"] = "auto"
    else:
        os.environ["ENABLE_SMOTE"] = "0"

    os.environ["MIN_CLASS_SAMPLES"] = str(args.min_class_samples)

    # Step 1: Rebuild dataset if requested
    processed_csv = os.path.join(PROJECT_ROOT, "data", "processed", "all_cases.csv")
    master_csv = os.path.join(PROJECT_ROOT, "data", "processed", "all_cases_master.csv")
    if args.rebuild_dataset:
        logger.info("Rebuilding dataset from raw CSVs...")
        from ai_court.data.prepare_dataset import build_from_dirs
        raw_dirs = [
            os.path.join(PROJECT_ROOT, "data", "raw"),
            os.path.join(PROJECT_ROOT, "data", "raw_enriched"),
        ]
        build_from_dirs(raw_dirs, processed_csv, min_text_len=20, dedupe=True)
        logger.info("Dataset rebuilt: %s", processed_csv)

    # Step 2: Load data
    logger.info("Loading data...")
    import pandas as pd
    from ai_court.model.legal_case_classifier import LegalCaseClassifier

    clf = LegalCaseClassifier()

    # Load from raw CSVs (DataLoader handles normalization + ontology mapping)
    raw_dirs = [
        os.path.join(PROJECT_ROOT, "data", "raw"),
        os.path.join(PROJECT_ROOT, "data", "raw_enriched"),
    ]
    csv_files = []
    for d in raw_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(d, f))

    # Also include master CSV from harvest_100k pipeline if it exists
    if os.path.isfile(master_csv):
        csv_files.append(master_csv)
        logger.info("Including master CSV: %s", master_csv)

    df = clf.data_loader.load_data(csv_files)
    stats = clf.data_loader.analyze_dataset(df)
    logger.info("Dataset stats: %s", json.dumps(stats, indent=2))

    # Step 3: Prepare data
    logger.info("Preparing data (normalization + label encoding)...")
    X_train, X_test, y_train, y_test = clf.data_loader.prepare_data(df)
    logger.info(
        "Train: %d, Test: %d, Classes: %d (%s)",
        len(X_train), len(X_test),
        len(clf.data_loader.label_encoder.classes_),
        list(clf.data_loader.label_encoder.classes_),
    )

    # Step 4: Train
    logger.info("Training model (SMOTE=%s)...", os.environ.get("ENABLE_SMOTE", "0"))
    t0 = time.perf_counter()
    pipeline, f1 = clf.trainer.train(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Training completed in %.1fs, CV F1: %.3f", elapsed, f1)

    # Step 5: Evaluate
    logger.info("Evaluating on test set...")
    clf.model = pipeline
    clf.label_encoder = clf.data_loader.label_encoder
    metrics = clf.trainer.evaluate(pipeline, X_test, y_test, clf.data_loader.label_encoder)
    logger.info("Test accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
                metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])

    # Step 6: Save model
    model_path = os.path.join(PROJECT_ROOT, "models", "legal_case_classifier.pkl")
    clf.trainer.save_model(model_path, clf.data_loader.label_encoder)
    logger.info("Model saved to %s", model_path)

    # Step 7: Save metrics
    run_id = time.strftime("%Y%m%d%H%M%S") + "_retrain"
    metrics_data = {
        "final_model": {
            "run_id": run_id,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "cv_f1": f1,
            "num_classes": len(clf.data_loader.label_encoder.classes_),
            "classes": list(clf.data_loader.label_encoder.classes_),
            "class_distribution": {
                str(clf.data_loader.label_encoder.inverse_transform([i])[0]): int(c)
                for i, c in zip(*__import__("numpy").unique(
                    __import__("numpy").concatenate([y_train, y_test]), return_counts=True
                ))
            },
            "per_class_f1": metrics.get("per_class_f1", {}),
            "smote_enabled": os.environ.get("ENABLE_SMOTE", "0") == "1",
            "training_time_seconds": round(elapsed, 1),
        },
        "data": {
            "total_cases": stats["total_cases"],
            "num_csv_files": len(csv_files),
        },
    }
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Step 8: Rebuild search index if requested
    if args.rebuild_search:
        logger.info("Rebuilding search index...")
        try:
            from scripts.build_search_index import main as build_search
            build_search()
            logger.info("Search index rebuilt")
        except Exception as exc:
            logger.warning("Search index rebuild failed: %s", exc)

    logger.info("Done! Model retrained with %d classes on %d cases.", 
                len(clf.data_loader.label_encoder.classes_), stats["total_cases"])


if __name__ == "__main__":
    main()

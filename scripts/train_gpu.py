"""
GPU-accelerated model training for AI Court System.
====================================================

This script trains TWO models:
1. TF-IDF + RandomForest (CPU) — fast, serves as baseline + production fallback
2. DistilBERT classifier (GPU) — higher accuracy, transformer-based

With 43K+ labeled cases and an RTX 4050 (6GB VRAM), DistilBERT training
takes ~15-30 minutes for 3 epochs.

Usage:
    python scripts/train_gpu.py                    # Train both models
    python scripts/train_gpu.py --rf-only          # Only RF (no GPU needed)
    python scripts/train_gpu.py --bert-only        # Only DistilBERT (GPU)
    python scripts/train_gpu.py --epochs 5         # More epochs
    python scripts/train_gpu.py --batch-size 16    # Smaller batch if OOM
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gpu")


def load_all_data():
    """Load and prepare all labeled data."""
    from ai_court.data.loader import DataLoader

    dl = DataLoader()
    csv_files = []
    for d in ["data/raw", "data/raw_enriched"]:
        if os.path.isdir(d):
            csv_files.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))

    master = "data/processed/all_cases_master.csv"
    if os.path.isfile(master):
        csv_files.append(master)

    logger.info(f"Loading from {len(csv_files)} CSV files...")
    df = dl.load_data(csv_files)

    # Filter out "Other" — only keep labeled data for classification training
    labeled = df[df["judgement"] != "Other"].copy()
    logger.info(f"Total rows: {len(df)}, Labeled (non-Other): {len(labeled)}")

    stats = dl.analyze_dataset(labeled)
    logger.info(f"Classes: {stats['num_classes']}")
    logger.info(f"Distribution: {json.dumps(stats['judgement_distribution'], indent=2)}")

    return df, labeled, dl


def train_rf(labeled_df, dl, enable_smote=True):
    """Train TF-IDF + RandomForest classifier on CPU."""
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING: TF-IDF + RandomForest (CPU)")
    logger.info("=" * 60)

    os.environ["ENABLE_SMOTE"] = "1" if enable_smote else "0"
    os.environ["MIN_CLASS_SAMPLES"] = "3"

    X_train, X_test, y_train, y_test = dl.prepare_data(labeled_df)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {len(dl.label_encoder.classes_)}")

    from ai_court.model.trainer import Trainer
    trainer = Trainer()

    t0 = time.perf_counter()
    pipeline, cv_f1 = trainer.train(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info(f"RF training done in {elapsed:.1f}s, CV F1: {cv_f1:.3f}")

    # Evaluate
    metrics = trainer.evaluate(pipeline, X_test, y_test, dl.label_encoder)
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    # Save
    model_path = "models/legal_case_classifier.pkl"
    trainer.save_model(model_path, dl.label_encoder)
    logger.info(f"RF model saved: {model_path}")

    return metrics, dl.label_encoder


def train_bert(labeled_df, dl, epochs=3, batch_size=24, max_len=256, lr=2e-5):
    """Train DistilBERT classifier on GPU."""
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader as TorchDL
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, f1_score, accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60}")
    logger.info(f"  TRAINING: DistilBERT ({device})")
    logger.info(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    logger.info(f"  Epochs: {epochs}, Batch: {batch_size}, MaxLen: {max_len}, LR: {lr}")
    logger.info(f"{'='*60}")

    # Prepare data
    texts = (labeled_df["case_type"].astype(str) + " " + labeled_df["case_data"].astype(str)).tolist()
    labels_raw = labeled_df["judgement"].astype(str).tolist()

    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    num_classes = len(le.classes_)
    logger.info(f"Classes ({num_classes}): {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class LegalDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_ds = LegalDataset(X_train, y_train)
    test_ds = LegalDataset(X_test, y_test)
    train_loader = TorchDL(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = TorchDL(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_classes
    )
    model.to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Enable mixed precision for faster training on RTX 4050
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

            if (step + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{epochs} Step {step+1}/{len(train_loader)} "
                    f"Loss: {total_loss/(step+1):.4f} Acc: {correct/total:.4f}"
                )

        elapsed = time.perf_counter() - t0
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

        logger.info(
            f"\n  Epoch {epoch+1}/{epochs}: {elapsed:.1f}s | "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  ** New best model (F1: {best_f1:.4f}) **")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    final_weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL RESULTS (DistilBERT)")
    logger.info(f"  Accuracy: {final_acc:.4f}")
    logger.info(f"  Macro F1: {final_macro_f1:.4f}")
    logger.info(f"  Weighted F1: {final_weighted_f1:.4f}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=le.classes_, zero_division=0)}")

    # Save model
    save_dir = Path("models/production/bert_classifier")
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # Save label encoder mapping
    label_mapping = {str(i): str(c) for i, c in enumerate(le.classes_)}
    with open(save_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Save metrics
    metrics = {
        "model": "distilbert-base-uncased",
        "accuracy": final_acc,
        "macro_f1": final_macro_f1,
        "weighted_f1": final_weighted_f1,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "lr": lr,
        "num_classes": num_classes,
        "classes": list(le.classes_),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "per_class": {
            cls: {"f1": report[cls]["f1-score"], "precision": report[cls]["precision"],
                  "recall": report[cls]["recall"], "support": report[cls]["support"]}
            for cls in le.classes_ if cls in report
        },
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"DistilBERT model saved to {save_dir}/")

    # Cleanup GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def rebuild_search_index(all_df):
    """Rebuild the TF-IDF search index with all data."""
    logger.info("\nRebuilding search index...")
    from ai_court.search.indexer import build_index

    # Prepare DataFrame in the format build_index expects: text, title, url
    index_df = pd.DataFrame({
        "text": (all_df["case_type"].astype(str) + " " + all_df["case_data"].astype(str)),
        "title": all_df.get("title", pd.Series([""] * len(all_df))).fillna(""),
        "url": all_df.get("url", pd.Series([""] * len(all_df))).fillna(""),
    })

    # Limit to 200K for memory
    max_index = 200000
    if len(index_df) > max_index:
        logger.info(f"Sampling {max_index} from {len(index_df)} for search index")
        # Prioritize labeled cases
        labeled_mask = all_df["judgement"] != "Other"
        labeled_idx = index_df.index[labeled_mask].tolist()
        other_idx = index_df.index[~labeled_mask].tolist()
        import random
        random.seed(42)
        remaining = max_index - len(labeled_idx)
        if remaining > 0:
            sampled_other = random.sample(other_idx, min(remaining, len(other_idx)))
            keep_idx = labeled_idx + sampled_other
        else:
            keep_idx = labeled_idx[:max_index]
        index_df = index_df.loc[keep_idx].reset_index(drop=True)

    logger.info(f"Building index on {len(index_df)} texts...")
    out_path = "models/search_index.pkl"
    build_index(index_df, out_path)
    logger.info(f"Search index rebuilt: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated AI Court model training")
    parser.add_argument("--rf-only", action="store_true", help="Only train RandomForest (CPU)")
    parser.add_argument("--bert-only", action="store_true", help="Only train DistilBERT (GPU)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-search", action="store_true", help="Skip search index rebuild")
    args = parser.parse_args()

    train_both = not args.rf_only and not args.bert_only

    # Load data
    all_df, labeled_df, dl = load_all_data()

    results = {}

    # Train RF
    if args.rf_only or train_both:
        rf_metrics, _ = train_rf(labeled_df, dl)
        results["rf"] = rf_metrics

    # Train DistilBERT
    if args.bert_only or train_both:
        bert_metrics = train_bert(
            labeled_df, dl,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
        )
        results["bert"] = bert_metrics

    # Rebuild search index
    if not args.no_search:
        rebuild_search_index(all_df)

    # Save combined metrics
    combined = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cases": len(all_df),
        "labeled_cases": len(labeled_df),
        "models": results,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Total data: {len(all_df):,} cases")
    logger.info(f"  Labeled: {len(labeled_df):,} cases")
    if "rf" in results:
        logger.info(f"  RF — Accuracy: {results['rf']['accuracy']:.4f}, Weighted F1: {results['rf']['weighted_f1']:.4f}")
    if "bert" in results:
        logger.info(f"  BERT — Accuracy: {results['bert']['accuracy']:.4f}, Weighted F1: {results['bert']['weighted_f1']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Professional Batch Training Pipeline
====================================
Automatically trains on new batches of legal cases every 5 hours.

Features:
- Batch processing (memory efficient)
- Auto-checkpointing (resume on crash)
- GPU optimization (RTX 4050)
- Metrics tracking
- Production-ready

Usage:
    python batch_trainer.py --batch_size 1000
"""

import os
import sys
import sqlite3
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import argparse

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "legal_cases_10M.db"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'batch_trainer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 8  # Small batch for 6GB VRAM
GRADIENT_ACCUMULATION = 4  # Effective batch = 32
EPOCHS_PER_BATCH = 2
LEARNING_RATE = 2e-5
FP16 = True  # Mixed precision for RTX 4050


class BatchTrainer:
    """Professional batch training system"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.label_map = {
            'Convicted': 0,
            'Acquitted': 1,
            'Dismissed': 2,
            'Allowed': 3,
            'Partly Allowed': 4,
            'Remanded': 5,
            'Unknown': 6
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        logger.info(f"Initialized BatchTrainer on {self.device}")
        logger.info(f"Batch size: {batch_size} cases")
        
    def get_total_cases(self) -> int:
        """Get total number of cases in database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cases")
            total = cursor.fetchone()[0]
            conn.close()
            return total
        except Exception as e:
            logger.error(f"Error getting total cases: {e}")
            return 0
            
    def get_last_trained_count(self) -> int:
        """Get number of cases used in last training"""
        metadata_file = PRODUCTION_DIR / "training_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('total_cases', 0)
            except:
                return 0
        return 0
        
    def should_train(self) -> bool:
        """Check if we have enough new cases to train"""
        total = self.get_total_cases()
        last_trained = self.get_last_trained_count()
        new_cases = total - last_trained
        
        logger.info(f"Total cases: {total}")
        logger.info(f"Last trained on: {last_trained}")
        logger.info(f"New cases: {new_cases}")
        
        if new_cases >= self.batch_size:
            logger.info(f"✓ Ready to train! ({new_cases} >= {self.batch_size})")
            return True
        else:
            logger.info(f"✗ Not enough cases ({new_cases} < {self.batch_size})")
            return False
            
    def load_batch_data(self) -> pd.DataFrame:
        """Load cases from database"""
        logger.info("Loading data from database...")
        
        try:
            conn = sqlite3.connect(DB_PATH)
            
            # Load all cases
            query = """
                SELECT 
                    case_id,
                    title,
                    court,
                    decision_text,
                    outcome
                FROM cases
                WHERE decision_text IS NOT NULL 
                AND decision_text != ''
                AND outcome IS NOT NULL
                ORDER BY scraped_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} cases")
            
            # Clean text
            df['text'] = df.apply(
                lambda row: f"{row['title']} [SEP] {row['decision_text'][:2000]}", 
                axis=1
            )
            
            # Map outcomes to labels
            df['label'] = df['outcome'].map(self.label_map).fillna(6).astype(int)
            
            # Remove unknown outcomes for training
            df = df[df['label'] != 6].reset_index(drop=True)
            
            logger.info(f"After filtering: {len(df)} cases")
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """Prepare train/test datasets"""
        logger.info("Preparing datasets...")
        
        # Split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        logger.info(f"Train: {len(train_df)} cases")
        logger.info(f"Test: {len(test_df)} cases")
        
        # Convert to HF datasets
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, test_dataset
        
    def train_batch(self):
        """Train on current batch"""
        logger.info("=" * 80)
        logger.info("STARTING BATCH TRAINING")
        logger.info("=" * 80)
        
        # Check GPU
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load data
        df = self.load_batch_data()
        
        if len(df) < 100:
            logger.warning("Not enough data to train! Need at least 100 cases.")
            return False
            
        # Initialize model
        logger.info("Loading model...")
        num_labels = len([k for k in self.label_map.keys() if k != 'Unknown'])
        
        # Check if we have a checkpoint
        checkpoint_files = list(CHECKPOINT_DIR.glob("checkpoint-*"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                latest_checkpoint,
                num_labels=num_labels
            )
        else:
            logger.info(f"Starting fresh from {MODEL_NAME}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_labels,
                use_safetensors=True  # Use safetensors to avoid PyTorch 2.6 requirement
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        
        # Prepare datasets
        train_dataset, test_dataset = self.prepare_datasets(df)
        
        # Training arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = CHECKPOINT_DIR / f"batch_{timestamp}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=EPOCHS_PER_BATCH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=200,
            save_steps=200,  # Must be multiple of eval_steps when using load_best_model_at_end
            save_total_limit=3,
            fp16=FP16,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="none",
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)
            
            acc = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average='macro')
            f1_weighted = f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train!
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Evaluate
        logger.info("Evaluating...")
        eval_result = trainer.evaluate()
        
        # Save production model
        logger.info("Saving production model...")
        production_path = PRODUCTION_DIR / f"model_{timestamp}"
        trainer.save_model(str(production_path))
        self.tokenizer.save_pretrained(str(production_path))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_cases': len(df),
            'train_cases': len(train_dataset),
            'test_cases': len(test_dataset),
            'accuracy': eval_result['eval_accuracy'],
            'f1_macro': eval_result['eval_f1_macro'],
            'f1_weighted': eval_result['eval_f1_weighted'],
            'model_path': str(production_path),
            'device': self.device,
            'epochs': EPOCHS_PER_BATCH,
            'batch_size': BATCH_SIZE,
            'label_distribution': df['label'].value_counts().to_dict()
        }
        
        with open(PRODUCTION_DIR / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
        logger.info(f"Macro F1: {eval_result['eval_f1_macro']:.4f}")
        logger.info(f"Weighted F1: {eval_result['eval_f1_weighted']:.4f}")
        logger.info(f"Model saved to: {production_path}")
        logger.info("=" * 80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Batch Trainer')
    parser.add_argument('--batch_size', type=int, default=1000, help='Minimum cases to trigger training')
    parser.add_argument('--force', action='store_true', help='Force training even if not enough new cases')
    args = parser.parse_args()
    
    trainer = BatchTrainer(batch_size=args.batch_size)
    
    if args.force or trainer.should_train():
        success = trainer.train_batch()
        if success:
            logger.info("✓ Batch training completed successfully")
        else:
            logger.error("✗ Batch training failed")
            sys.exit(1)
    else:
        logger.info("Waiting for more cases...")
        

if __name__ == "__main__":
    main()

import os
import json
import uuid
import dill
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ai_court.model.preprocessor import TextPreprocessor
from ai_court.data.loader import DataLoader
from ai_court.ontology import ontology_metadata

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader()
        self.model: Optional[Pipeline] = None
        
    def train(self, X_train: pd.Series, y_train: np.ndarray, sample_weight=None) -> Tuple[Pipeline, float]:
        """Train TF-IDF + AdaBoost(RandomForest) pipeline."""
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=list(self.preprocessor.stop_words - self.preprocessor.legal_terms),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
        )

        rf_base = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
        )

        try:
            boosted_rf = AdaBoostClassifier(
                estimator=rf_base,
                n_estimators=10,
                learning_rate=0.5,
                algorithm='SAMME',
                random_state=42,
            )
        except TypeError:
            boosted_rf = AdaBoostClassifier(
                base_estimator=rf_base,
                n_estimators=10,
                learning_rate=0.5,
                algorithm='SAMME',
                random_state=42,
            )

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', boosted_rf)
        ])
        
        if sample_weight is not None:
            try:
                pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
            except Exception:
                pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)
            
        train_pred = pipeline.predict(X_train)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        logger.info(f"Training F1 score (weighted): {train_f1:.3f}")
        
        self.model = pipeline
        return pipeline, float(train_f1)

    def train_logreg_baseline(self, X_train: pd.Series, y_train: np.ndarray) -> Pipeline:
        """Train a fast Logistic Regression baseline."""
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=list(self.preprocessor.stop_words - self.preprocessor.legal_terms),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
        )
        logreg = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='ovr',
            C=2.0,
            random_state=42,
        )
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', logreg)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, model: Pipeline, X_test: pd.Series, y_test: np.ndarray, label_encoder) -> Dict:
        if len(X_test) == 0:
            logger.warning("No test samples available")
            return {'accuracy': 0.0, 'report': ''}

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        present = sorted(set(y_test) | set(y_pred))
        target_names = [label_encoder.inverse_transform([i])[0] for i in present]

        logger.info(f"\nTest Accuracy: {accuracy:.4f}\n")
        report = classification_report(
            y_test, y_pred, labels=present, target_names=target_names, digits=4, zero_division=0
        )
        logger.info(report)

        return {'accuracy': float(accuracy), 'report': report}

    def save_model(self, filepath: str, label_encoder):
        """Save with dill for compatibility."""
        with open(filepath, 'wb') as f:
            dill.dump({
                'model': self.model,
                'label_encoder': label_encoder,
                'preprocessor': self.preprocessor.preprocess
            }, f)

    def run_training_pipeline(self, csv_paths: list[str], output_dir: str = "models"):
        """Execute the full training pipeline."""
        logger.info("Loading data...")
        df = self.data_loader.load_data(csv_paths)
        
        # Basic filtering
        min_text_len = int(os.getenv('MIN_TEXT_LEN', '0'))
        if min_text_len > 0:
            df = df[df['case_data'].astype(str).str.len() >= min_text_len]
            
        if os.getenv('EXCLUDE_OTHER', '1') != '0':
            df = df[~df['judgement'].astype(str).str.lower().isin(['unrecognized','other'])]

        # Deduplicate
        _canon = (
            df['case_data']
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df = df.loc[~_canon.duplicated()].reset_index(drop=True)

        logger.info("Analyzing dataset...")
        analysis = self.data_loader.analyze_dataset(df)
        
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(df)
        
        logger.info("Training model...")
        self.model, train_f1 = self.train(X_train, y_train, sample_weight=self.data_loader._train_weights)
        
        logger.info("Evaluating model...")
        eval_metrics = self.evaluate(self.model, X_test, y_test, self.data_loader.label_encoder)
        
        # Save artifacts
        run_id = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        run_dir = os.path.join(output_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        model_path = os.path.join(run_dir, "legal_case_classifier.pkl")
        self.save_model(model_path, self.data_loader.label_encoder)
        
        # Save metrics
        metrics = {
            "final_model": {
                "train_f1_weighted": train_f1,
                "test_accuracy": eval_metrics['accuracy'],
                "num_classes": int(self.data_loader.label_encoder.classes_.shape[0]),
                "classes": list(self.data_loader.label_encoder.classes_),
                "ontology": ontology_metadata(),
            },
            "data": {
                "duplicate_ratio": analysis.get('duplicate_ratio'),
                "total_cases": analysis.get('total_cases'),
            }
        }
        
        with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        # Update latest
        import shutil
        shutil.copy2(model_path, os.path.join(output_dir, "legal_case_classifier.pkl"))
        shutil.copy2(os.path.join(run_dir, 'metrics.json'), os.path.join(output_dir, 'metrics.json'))
        
        logger.info(f"Training complete. Model saved to {model_path}")
        return run_dir

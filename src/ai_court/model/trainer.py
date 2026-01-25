import os
import json
import uuid
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from ai_court.model.preprocessor import TextPreprocessor
from ai_court.data.loader import DataLoader
from ai_court.ontology import ontology_metadata

logger = logging.getLogger(__name__)

# Check for imbalanced-learn availability (optional dependency)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    ImbPipeline = None
    SMOTE = None
    ADASYN = None

# Configuration
ENABLE_SMOTE = os.getenv('ENABLE_SMOTE', '0') == '1'
SMOTE_SAMPLING_STRATEGY = float(os.getenv('SMOTE_SAMPLING_STRATEGY', '0.5'))


class Trainer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader()
        self.model: Optional[Pipeline] = None
        
    def _apply_resampling(self, X_tfidf, y_train) -> Tuple:
        """Apply SMOTE/ADASYN resampling if enabled and available.
        
        Args:
            X_tfidf: TF-IDF transformed features (sparse matrix)
            y_train: Target labels
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not ENABLE_SMOTE or not IMBALANCED_AVAILABLE:
            return X_tfidf, y_train
        
        try:
            # Count samples per class
            unique, counts = np.unique(y_train, return_counts=True)
            min_samples = counts.min()
            
            # SMOTE needs at least k_neighbors + 1 samples per class
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            if k_neighbors < 1:
                logger.warning(f"Not enough samples for SMOTE (min class has {min_samples} samples)")
                return X_tfidf, y_train
            
            logger.info(f"Applying SMOTE with k_neighbors={k_neighbors}, strategy={SMOTE_SAMPLING_STRATEGY}")
            
            smote = SMOTE(
                sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                k_neighbors=k_neighbors,
                random_state=42,
                n_jobs=-1
            )
            
            X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_train)
            
            # Log resampling results
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampling complete: {len(y_train)} -> {len(y_resampled)} samples")
            for cls, cnt in zip(unique_new, counts_new):
                logger.info(f"  Class {cls}: {cnt} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE resampling failed: {e}. Continuing without resampling.")
            return X_tfidf, y_train
        
    def train(self, X_train: pd.Series, y_train: np.ndarray, sample_weight=None) -> Tuple[Pipeline, float]:
        """Train TF-IDF + Optimized RandomForest pipeline.
        
        Optionally applies SMOTE resampling if ENABLE_SMOTE=1.
        """
        # Reduce max_features to save memory and size
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=list(self.preprocessor.stop_words - self.preprocessor.legal_terms),
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
        )

        # Use a single optimized RandomForest
        # Relaxed constraints to better capture minority classes while maintaining efficiency
        rf_optimized = RandomForestClassifier(
            n_estimators=200,      # Increased from 100 to reduce variance
            max_depth=None,        # Removed limit (was 20) to allow learning complex legal patterns
            min_samples_split=5,   # Reduced from 10 to capture finer details
            min_samples_leaf=2,    # Reduced from 4 for better minority class sensitivity
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            max_samples=0.9        # Increased slightly
        )
        
        # Apply resampling if enabled
        if ENABLE_SMOTE and IMBALANCED_AVAILABLE:
            logger.info("SMOTE enabled - transforming features before resampling")
            X_tfidf = vectorizer.fit_transform(X_train)
            X_resampled, y_resampled = self._apply_resampling(X_tfidf, y_train)
            
            # Train classifier on resampled data
            if sample_weight is not None and len(sample_weight) == len(y_resampled):
                rf_optimized.fit(X_resampled, y_resampled, sample_weight=sample_weight)
            else:
                rf_optimized.fit(X_resampled, y_resampled)
            
            # Create pipeline with pre-fitted vectorizer
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', rf_optimized)
            ])
            
            # Use resampled data for CV
            X_for_cv, y_for_cv = X_train, y_train  # Use original for realistic CV
        else:
            # Standard pipeline without resampling
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', rf_optimized)
            ])
            
            if sample_weight is not None:
                try:
                    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
                except Exception:
                    pipeline.fit(X_train, y_train)
            else:
                pipeline.fit(X_train, y_train)
            
            X_for_cv, y_for_cv = X_train, y_train
            
        # Perform Cross-Validation for a realistic metric
        try:
            logger.info("Running 5-fold Cross-Validation for realistic F1 estimation...")
            cv_scores = cross_val_score(pipeline, X_for_cv, y_for_cv, cv=5, scoring='f1_weighted', n_jobs=-1)
            f1_metric = float(cv_scores.mean())
            logger.info(f"Cross-Validation F1 (mean): {f1_metric:.3f} (+/- {cv_scores.std() * 2:.3f})")
        except Exception as e:
            logger.warning(f"Cross-Validation failed: {e}. Falling back to training score.")
            train_pred = pipeline.predict(X_train)
            f1_metric = f1_score(y_train, train_pred, average='weighted')
            logger.info(f"Training F1 score (Likely Overfitted): {f1_metric:.3f}")
        
        self.model = pipeline
        return pipeline, f1_metric

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
        """Save with joblib for better compatibility and valid metrics."""
        # Use joblib instead of dill
        joblib.dump({
            'model': self.model,
            'label_encoder': label_encoder,
            # We don't save preprocessor function to avoid serialization issues.
            # The loader should instantiate a fresh TextPreprocessor.
            'meta': {'version': '2.0', 'serialization': 'joblib'}
        }, filepath)

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

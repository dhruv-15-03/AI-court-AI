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
# SMOTE enabled by default for fine-grained (11-class) training
ENABLE_SMOTE = os.getenv('ENABLE_SMOTE', '1') == '1'
SMOTE_SAMPLING_STRATEGY = os.getenv('SMOTE_SAMPLING_STRATEGY', 'auto')
# 'auto' resamples all minority classes to match the majority class
# A float like '0.5' targets 50% of the majority class count


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
            max_samples = counts.max()
            
            # Check if resampling is beneficial (imbalance ratio > 3:1)
            if max_samples / max(min_samples, 1) < 3:
                logger.info("Class balance is acceptable (ratio %.1f:1), skipping SMOTE", max_samples / max(min_samples, 1))
                return X_tfidf, y_train
            
            # SMOTE needs at least k_neighbors + 1 samples per class
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            if k_neighbors < 1:
                logger.warning(f"Not enough samples for SMOTE (min class has {min_samples} samples)")
                return X_tfidf, y_train
            
            # Parse sampling strategy
            strategy = SMOTE_SAMPLING_STRATEGY
            if isinstance(strategy, str) and strategy != 'auto':
                try:
                    strategy = float(strategy)
                except ValueError:
                    strategy = 'auto'
            
            logger.info(
                "Applying SMOTE: k_neighbors=%d, strategy=%s, classes=%d, min=%d, max=%d",
                k_neighbors, strategy, len(unique), min_samples, max_samples,
            )
            
            smote = SMOTE(
                sampling_strategy=strategy,
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
            max_features=8000,
            ngram_range=(1, 2),
            stop_words=list(self.preprocessor.stop_words - self.preprocessor.legal_terms),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
        )

        # Regularised RandomForest — prevents memorisation and improves minority-class recall.
        # max_depth=30 stops leaf nodes from memorising individual training examples.
        # min_samples_leaf=4 / min_samples_split=10 permit finer splits for minority classes.
        # class_weight='balanced_subsample' up-weights minority classes per bootstrap sample.
        rf_optimized = RandomForestClassifier(
            n_estimators=300,          # more trees → better minority class coverage
            max_depth=30,              # hard cap prevents pure-memorisation leaves
            min_samples_split=10,      # allow splits at smaller sample sizes
            min_samples_leaf=4,        # each leaf must cover ≥4 training samples
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            max_samples=0.9
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
            return {'accuracy': 0.0, 'report': '', 'macro_f1': 0.0, 'weighted_f1': 0.0, 'per_class_f1': {}}

        from sklearn.metrics import precision_recall_fscore_support
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        present = sorted(set(y_test) | set(y_pred))
        target_names = [label_encoder.inverse_transform([i])[0] for i in present]

        logger.info(f"\nTest Accuracy: {accuracy:.4f}\n")
        report = classification_report(
            y_test, y_pred, labels=present, target_names=target_names, digits=4, zero_division=0
        )
        logger.info(report)

        macro_f1 = float(f1_score(y_test, y_pred, average='macro', labels=present, zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', labels=present, zero_division=0))

        # Per-class F1 keyed by human-readable label name
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=present, zero_division=0
        )
        per_class_f1 = {
            label_encoder.inverse_transform([i])[0]: round(float(f), 4)
            for i, f in zip(present, f1_per_class)
        }
        logger.info("Per-class F1: %s", per_class_f1)

        return {
            'accuracy': float(accuracy),
            'report': report,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_f1': per_class_f1,
        }

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

    def _oversample_minority_classes(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
    ) -> tuple[pd.Series, np.ndarray]:
        """Oversample minority classes on the *training split only*.

        Any class whose sample count falls below ``MIN_CLASS_SAMPLES`` (env var,
        default 500) is duplicated with replacement until it reaches that floor.
        This prevents the RandomForest from ignoring rare classes entirely.
        If imbalanced-learn is available and ENABLE_SMOTE=1, RandomOverSampler
        from imbalanced-learn is used instead for cleaner API semantics.
        """
        min_threshold = int(os.getenv('MIN_CLASS_SAMPLES', '500'))
        classes, counts = np.unique(y_train, return_counts=True)

        # Skip if all classes already meet the threshold
        if counts.min() >= min_threshold:
            return X_train, y_train

        logger.info(
            "Oversampling minority classes to %d samples minimum. "
            "Class counts before: %s",
            min_threshold,
            dict(zip(classes.tolist(), counts.tolist())),
        )

        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(
                sampling_strategy={c: max(int(n), min_threshold) for c, n in zip(classes, counts)},
                random_state=42,
            )
            X_arr = X_train.to_numpy().reshape(-1, 1)
            X_resampled, y_resampled = ros.fit_resample(X_arr, y_train)
            X_out = pd.Series(X_resampled.ravel(), name=X_train.name)
        except ImportError:
            # Manual duplication fallback (no imbalanced-learn required)
            parts_X = [X_train]
            parts_y = [y_train]
            for cls, cnt in zip(classes, counts):
                if cnt < min_threshold:
                    need = min_threshold - cnt
                    mask = y_train == cls
                    idx = np.random.choice(np.where(mask)[0], size=need, replace=True)
                    parts_X.append(X_train.iloc[idx])
                    parts_y.append(y_train[idx])
            X_out = pd.concat(parts_X, ignore_index=True)
            y_resampled = np.concatenate(parts_y)

        _, new_counts = np.unique(y_resampled, return_counts=True)
        logger.info("Class counts after oversampling: %s",
                    dict(zip(classes.tolist(), new_counts.tolist())))
        return X_out, y_resampled

    def run_training_pipeline(self, csv_paths: list[str], output_dir: str = "models"):
        """Execute the full training pipeline."""
        import hashlib
        import shutil
        from datetime import timezone as _tz

        logger.info("Loading data...")
        df = self.data_loader.load_data(csv_paths)

        # Basic filtering
        min_text_len = int(os.getenv('MIN_TEXT_LEN', '0'))
        if min_text_len > 0:
            df = df[df['case_data'].astype(str).str.len() >= min_text_len]

        if os.getenv('EXCLUDE_OTHER', '1') != '0':
            df = df[~df['judgement'].astype(str).str.lower().isin(['unrecognized', 'other'])]

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

        # Dataset fingerprint
        _hash = hashlib.md5(
            "".join(sorted(df['case_data'].astype(str).tolist())).encode()
        ).hexdigest()

        logger.info("Analyzing dataset...")
        analysis = self.data_loader.analyze_dataset(df)

        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(df)

        # ── Minority-class oversampling on TRAINING SPLIT ONLY ──────────────
        X_train, y_train = self._oversample_minority_classes(X_train, y_train)

        logger.info("Training model...")
        self.model, train_f1 = self.train(X_train, y_train, sample_weight=self.data_loader._train_weights)

        logger.info("Evaluating model...")
        eval_metrics = self.evaluate(self.model, X_test, y_test, self.data_loader.label_encoder)

        # ── Persist artifacts ───────────────────────────────────────────────
        run_id = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        trained_at = datetime.now(tz=_tz.utc).isoformat().replace('+00:00', 'Z')
        run_dir = os.path.join(output_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        model_path = os.path.join(run_dir, "legal_case_classifier.pkl")
        self.save_model(model_path, self.data_loader.label_encoder)

        label_encoder = self.data_loader.label_encoder
        classes = list(label_encoder.classes_)
        class_distribution = analysis.get('judgement_distribution', {})

        # Load previous run_id for lineage tracking
        _prev_run_id = None
        _prev_meta_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(_prev_meta_path):
            try:
                with open(_prev_meta_path, 'r', encoding='utf-8') as _f:
                    _prev_run_id = json.load(_f).get('run_id')
            except Exception:
                pass

        # ------------------------------------------------------------------
        # metrics.json  (consumed by API /api/model_metrics, test guards)
        # ------------------------------------------------------------------
        metrics = {
            "final_model": {
                "run_id": run_id,
                "trained_at": trained_at,
                "train_f1_weighted": train_f1,
                "test_accuracy": eval_metrics['accuracy'],
                "test_macro_f1": eval_metrics['macro_f1'],
                "test_weighted_f1": eval_metrics['weighted_f1'],
                "cv_macro_f1_mean": None,  # filled by cross-val inside train()
                "cv_macro_f1_std": None,
                "num_classes": int(label_encoder.classes_.shape[0]),
                "classes": classes,
                "class_distribution": class_distribution,
                "per_class_f1": eval_metrics['per_class_f1'],
                "ontology": ontology_metadata(),
            },
            "data": {
                "duplicate_ratio": analysis.get('duplicate_ratio'),
                "total_cases": analysis.get('total_cases'),
                "dataset_hash": _hash,
            },
        }

        # ------------------------------------------------------------------
        # metadata.json  (consumed by /version, MODEL_CARD generator)
        # ------------------------------------------------------------------
        metadata = {
            "model_path": os.path.join(output_dir, "legal_case_classifier.pkl"),
            "trained_at": trained_at,
            "dataset_hash": _hash,
            "dataset_rows": analysis.get('total_cases'),
            "num_classes": int(label_encoder.classes_.shape[0]),
            "classes": classes,
            "class_distribution": class_distribution,
            "duplicate_ratio": analysis.get('duplicate_ratio'),
            "train_f1_weighted": train_f1,
            "test_accuracy": eval_metrics['accuracy'],
            "test_macro_f1": eval_metrics['macro_f1'],
            "test_weighted_f1": eval_metrics['weighted_f1'],
            "cv_macro_f1_mean": None,
            "cv_macro_f1_std": None,
            "per_class_f1": eval_metrics['per_class_f1'],
            "head_macro_f1": None,
            "tail_macro_f1": None,
            "pre_upsampling_distribution": class_distribution,
            "post_upsampling_distribution": None,
            "run_id": run_id,
            "previous_run": _prev_run_id,
            "ontology": ontology_metadata(),
            "holdout_metrics": None,
        }

        # Write run-level copies
        with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Promote to top-level (both files written from the SAME run atomically)
        shutil.copy2(model_path, os.path.join(output_dir, "legal_case_classifier.pkl"))
        shutil.copy2(os.path.join(run_dir, 'metrics.json'), os.path.join(output_dir, 'metrics.json'))
        shutil.copy2(os.path.join(run_dir, 'metadata.json'), os.path.join(output_dir, 'metadata.json'))

        # Append to history log
        _history_path = os.path.join(output_dir, 'history.log')
        with open(_history_path, 'a', encoding='utf-8') as _hf:
            _hf.write(json.dumps(metadata) + '\n')

        logger.info("Training complete. Run artifacts in %s", run_dir)
        return run_dir

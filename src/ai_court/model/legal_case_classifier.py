import os
import re
import warnings
from typing import List, Tuple, Dict, Any
import json
import hashlib
import uuid
from datetime import datetime
from sklearn.metrics import confusion_matrix
try:
    import mlflow  
except Exception:  
    mlflow = None  

try:
    from ai_court.ontology import map_coarse_label, ontology_metadata  # type: ignore
except Exception:  # pragma: no cover
    def map_coarse_label(label: str):  # fallback stub
        return label, False
    def ontology_metadata():
        return {}

import dill
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import argparse
from importlib import import_module

warnings.filterwarnings('ignore')

NLTK_DATA_DIR = os.path.join(os.path.expanduser("~"), ".nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", NLTK_DATA_DIR)

for pkg, kind in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers"), ("stopwords", "corpora"), ("wordnet", "corpora"), ("omw-1.4", "corpora")]:
    try:
        nltk.data.find(f"{kind}/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass


class LegalCaseClassifier:
    """
    End-to-end text model to predict legal case judgement/outcome.

    Architecture:
      - Preprocess: lowercase, clean, tokenize, stopword-aware lemmatization with legal-term preservation
      - Vectorizer: TF-IDF (word n-grams 1-3), sublinear TF, max_features tuned for small/medium data
      - Classifier: AdaBoost(SAMME.R) over RandomForest base estimator (RandomForest + Boost as requested)

    Saved artifact includes: model (sklearn Pipeline), label_encoder, preprocessor function
    """

    def __init__(self):
        self.model: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Preserve important legal tokens even if they are stopwords
        self.legal_terms = {
            'plaintiff', 'defendant', 'appeal', 'appeals', 'judgment', 'judgement', 'court', 'section',
            'act', 'article', 'respondent', 'appellant', 'petitioner', 'accused', 'evidence', 'conviction',
            'acquittal', 'forensic', 'witness', 'testimony', 'murder', 'rape', 'ipc', 'crpc', 'dismissed',
            'upheld', 'affirmed', 'ballistic', 'circumstantial', 'alibi', 'bail', 'trial', 'sentence',
            'sentencing', 'compromise', 'settlement', 'damages', 'compensation', 'negligence', 'property'
        }

    def preprocess_text(self, text: str) -> str:
        """Normalize and lemmatize text while keeping key legal terms."""
        if not isinstance(text, str):
            return ""
        # Local import for robustness if global import context altered after dill load
        import re  # type: ignore
        # Basic cleanup
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)  # Keep ASCII letters and spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)
        # Keep tokens if either not a stopword or part of a curated legal vocabulary
        filtered = [t for t in tokens if (t not in self.stop_words) or (t in self.legal_terms)]
        # Lemmatize (verbs then nouns)
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in filtered]
        lemmas = [self.lemmatizer.lemmatize(t) for t in lemmas]
        return " ".join(lemmas)

    def _read_single(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        required_cols = ['case_data', 'case_type', 'judgement']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"{os.path.basename(file_path)} must contain columns: {required_cols}")
        # Basic cleaning
        df = df.dropna(subset=required_cols)
        df = df[df['case_data'].astype(str).str.strip().astype(bool)]
        df = df[df['judgement'].astype(str).str.strip().astype(bool)]
        # Preserve optional weak supervision columns if present
        optional = [c for c in ['weak_label','weak_label_sources','weak_label_count'] if c in df.columns]
        df = df[required_cols + optional].copy()
        use_curated = bool(os.getenv('USE_CURATED_LABELS'))
        if use_curated:
            # Treat existing judgement values as final curated labels; no normalization / mapping
            df['coarse_judgement'] = df['judgement'].astype(str)
            # Optional phase2 demotion + overrides
            if os.getenv('CURATED_APPLY_PHASE2'):
                try:
                    import yaml  # type: ignore
                    overrides_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..','ontology','curated_overrides.yml')
                    overrides_path = os.path.normpath(overrides_path)
                    overrides = {}
                    if os.path.exists(overrides_path):
                        with open(overrides_path,'r',encoding='utf-8') as f:
                            overrides = yaml.safe_load(f) or {}
                    try:
                        min_support = int(os.getenv('CURATED_MIN_SUPPORT','5'))
                    except ValueError:
                        min_support = 5
                    protected = set(overrides.keys())
                    counts = df['judgement'].value_counts().to_dict()
                    demote_labels = [lbl for lbl,cnt in counts.items() if cnt < min_support and lbl not in protected]
                    if demote_labels:
                        df.loc[df['judgement'].isin(demote_labels),'judgement'] = os.getenv('UNRECOGNIZED_TARGET','Other')
                    if os.getenv('CURATED_MAP_TO_ONTOLOGY'):
                        df['judgement'] = df['judgement'].apply(lambda x: overrides.get(x, x))
                except Exception as ce:  # pragma: no cover
                    print(f"[curated] Phase2 overrides skipped: {ce}")
        else:
            # Map normalization -> ontology leaf id
            df['coarse_judgement'] = df['judgement'].astype(str).apply(self.normalize_outcome)
            df['judgement'] = df['coarse_judgement'].apply(lambda x: map_coarse_label(x)[0])
        return df

    @staticmethod
    def normalize_outcome(text: str) -> str:
        """Map raw judgment text to a manageable set of outcome classes."""
        t = (text or "").lower()
        # Strong class indicators
        # Acquittal / Conviction overturned
        if any(k in t for k in [
            "acquitted", "acquittal", "conviction overturned", "set aside conviction", "set aside the conviction",
            "set aside", "reversal of conviction", "benefit of doubt", "give benefit of doubt", "acquit the accused",
            "appeal allowed and conviction set aside", "conviction quashed"
        ]):
            return "Acquittal/Conviction Overturned"
        # Conviction upheld / Appeal dismissed
        if any(k in t for k in [
            "appeal dismissed", "appeal is dismissed", "conviction upheld", "conviction affirmed", "convictions upheld",
            "life sentence upheld", "appeal fails", "dismissed on merits", "dismissed in limine", "leave refused",
            "revision dismissed", "petition dismissed as devoid of merit"
        ]):
            return "Conviction Upheld/Appeal Dismissed"
        # Charges / Proceedings quashed
        # More specific: charge sheet quashed (must check before generic quash tokens)
        if any(k in t for k in [
            "charge sheet quashed", "chargesheet quashed", "charge-sheet quashed", "quash the charge sheet",
            "quashing of charge sheet", "charge sheet is quashed", "charge-sheet is quashed"
        ]):
            return "Charge Sheet Quashed"
        if any(k in t for k in [
            "quash", "quashed", "quashing", "fir quashed", "charges quashed", "proceedings quashed",
            "section 482 allowed", "u/s 482 allowed", "proceedings under section 482 are quashed"
        ]):
            return "Charges/Proceedings Quashed"
        # Sentence reduced / modified
        if any(k in t for k in [
            "sentence reduced", "sentence modified", "reduced sentence", "commuted", "converted to",
            "altered sentence", "sentence altered", "imprisonment reduced", "fine reduced"
        ]):
            return "Sentence Reduced/Modified"
        # Bail granted / denied
        if any(k in t for k in [
            "bail granted", "anticipatory bail granted", "interim bail", "enlarged on bail", "released on bail"
        ]):
            return "Bail Granted"
        if any(k in t for k in ["bail denied", "bail rejected", "bail refused"]):
            return "Bail Denied"
        # Relief granted / convicted
        if any(k in t for k in [
            "petition allowed", "writ allowed", "relief granted", "granted protection", "mandated", "directed",
            "ordered", "convicted", "allowed in part", "partly allowed", "set aside order and remand"
        ]):
            return "Relief Granted/Convicted"
        # Relief denied / dismissed
        if any(k in t for k in [
            "petition dismissed", "relief denied", "dismissed as infructuous", "dismissed on merits", "dismissed",
            "no interference called for", "no merit in the petition"
        ]):
            return "Relief Denied/Dismissed"
        return "Other"

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and concatenate datasets from multiple CSVs containing required columns."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        frames = []
        errors: Dict[str, str] = {}
        for p in file_paths:
            try:
                frames.append(self._read_single(p))
            except Exception as e:
                errors[p] = str(e)
        if not frames:
            msg = "No valid datasets loaded. "
            if errors:
                msg += "Errors: " + "; ".join([f"{os.path.basename(k)}: {v}" for k, v in errors.items()])
            raise ValueError(msg)
        df = pd.concat(frames, ignore_index=True)
        if len(df) < 20:
            warnings.warn("Dataset is quite small (<20 rows); results may be unstable.")
        return df

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Compute basic dataset stats plus duplicate ratio using simple canonical hashing."""
        # Canonicalize text for duplicate detection
        canon = (
            df['case_data']
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        unique_canon = canon.nunique()
        total = len(df)
        duplicate_ratio = float(0.0 if total == 0 else (total - unique_canon) / total)
        judgement_counts = df['judgement'].value_counts().to_dict()
        analysis = {
            'total_cases': int(total),
            'case_types': df['case_type'].value_counts().to_dict(),
            'judgement_distribution': judgement_counts,
            'min_samples_per_class': int(df['judgement'].value_counts().min()),
            'num_classes': int(df['judgement'].nunique()),
            'duplicate_ratio': duplicate_ratio,
        }
        if analysis['num_classes'] < 2:
            raise ValueError("At least 2 judgement classes required")
        return analysis

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """Build text features and encode labels.

        Returns train/test splits plus (optionally) sample weights if '__sample_weight' is present.
        """
        df['legal_features'] = df['case_type'].astype(str).str.lower() + " " + df['case_data'].astype(str)
        df['processed_text'] = df['legal_features'].apply(self.preprocess_text)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['judgement'].astype(str))

        weights = None
        if '__sample_weight' in df.columns:
            weights = df['__sample_weight'].astype(float).to_numpy()

        test_size = 0.2 if len(df) >= 25 else max(0.15, min(0.2, 3 / len(df)))
        value_counts = pd.Series(y).value_counts()
        can_stratify = (value_counts.min() >= 2)
        stratify = y if can_stratify else None
        if weights is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                df['processed_text'], y, weights, test_size=test_size, random_state=42, stratify=stratify
            )
            # Attach for downstream use
            self._train_weights = w_train  # type: ignore[attr-defined]
            self._test_weights = w_test   # may be used for future evaluation weighting
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], y, test_size=test_size, random_state=42, stratify=stratify
            )
            return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.Series, y_train: np.ndarray, sample_weight=None) -> Tuple[Pipeline, float]:
        """Train TF-IDF + AdaBoost(RandomForest) pipeline as requested.

        Accepts optional sample_weight for weak supervision down-weighting.
        """
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=list(self.stop_words - self.legal_terms),
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
                # Fallback without weights if underlying estimator rejects sample_weight
                pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        print(f"Training F1 score (weighted): {train_f1:.3f}")
        return pipeline, float(train_f1)

    def train_logreg_baseline(self, X_train: pd.Series, y_train: np.ndarray) -> Pipeline:
        """Train a fast Logistic Regression baseline over TF-IDF for side-by-side comparison."""
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # a bit leaner for speed
            stop_words=list(self.stop_words - self.legal_terms),
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

    def evaluate(self, model: Pipeline, X_test: pd.Series, y_test: np.ndarray) -> Dict:
        if len(X_test) == 0:
            print("Warning: No test samples available")
            return {'accuracy': 0.0, 'report': ''}

        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        present = sorted(set(y_test) | set(y_pred))
        target_names = [self.label_encoder.inverse_transform([i])[0] for i in present]

        print(f"\nTest Accuracy: {accuracy:.4f}\n")
        report = classification_report(
            y_test, y_pred, labels=present, target_names=target_names, digits=4, zero_division=0
        )
        print(report)

        return {'accuracy': float(accuracy), 'report': report}

    def save_model(self, filepath: str):
        """Save with dill for compatibility with the Flask app loader."""
        with open(filepath, 'wb') as f:
            dill.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'preprocessor': self.preprocess_text
            }, f)

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            saved = dill.load(f)
            self.model = saved['model']
            self.label_encoder = saved['label_encoder']
            # Overwrite the preprocessing function if provided in artifact
            self.preprocess_text = saved.get('preprocessor', self.preprocess_text)  # type: ignore[method-assign]

    def predict_judgement(self, case_data: str, case_type: str) -> str:
        if not self.model:
            raise ValueError("Model not trained or loaded")
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted")
        legal_input = f"{str(case_type).lower()} {str(case_data)}"
        processed_input = self.preprocess_text(legal_input)
        pred = self.model.predict([processed_input])[0]
        return self.label_encoder.inverse_transform([pred])[0]

# === Multi-axis inference utilities (module level) ===
def load_multi_axis_model(model_dir: str = 'models/multi_axis'):
    """Load multi-axis transformer model for inference.

    Returns (model, tokenizer, label_maps).
    """
    from transformers import AutoTokenizer
    import torch
    from .multi_axis_transformer import MultiAxisModel
    ckpt_path = os.path.join(model_dir, 'multi_axis.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Multi-axis checkpoint not found at {ckpt_path}")
    bundle = torch.load(ckpt_path, map_location='cpu')
    label_maps = bundle['label_maps']
    model = MultiAxisModel(bundle['backbone'], {ax: len(m) for ax,m in label_maps.items()})
    model.load_state_dict(bundle['model_state'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(bundle['backbone'])
    return model, tokenizer, label_maps


def multi_axis_predict(texts, model_dir: str = 'models/multi_axis'):
    """Batch predict returning axis -> list[label]."""
    import torch
    from .multi_axis_transformer import AXES
    model, tokenizer, label_maps = load_multi_axis_model(model_dir)
    inv_maps = {ax: {v:k for k,v in mp.items()} for ax, mp in label_maps.items()}
    max_len = 512
    out = {ax: [] for ax in label_maps}
    for i in range(0, len(texts), 8):
        chunk = texts[i:i+8]
        enc = tokenizer(chunk, truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
        for ax in AXES:
            preds = logits[ax].argmax(dim=1).cpu().tolist()
            out[ax].extend(inv_maps[ax][p] for p in preds)
    return out

def main():
    """Train and persist the boosted RandomForest model using all compatible CSVs in the repo."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--abstention-report', action='store_true', help='Print abstention confusion summary after training')
    args, _ = parser.parse_known_args()
    try:
        print("=== Legal Case Outcome Model Trainer (RF + Boost) ===")
        classifier = LegalCaseClassifier()

        # Discover CSVs that likely contain the right schema
        processed_all = os.path.join('data', 'processed', 'all_cases.csv')
        curated_path = os.path.join('data','processed','all_cases_curated.csv')
        override_csv = os.environ.get('TRAIN_CSV')
        if override_csv and os.path.exists(override_csv):
            print('[trainer] Using override TRAIN_CSV:', override_csv)
            candidate_csvs = [override_csv]
        elif os.getenv('USE_CURATED_LABELS') and os.path.exists(curated_path):
            print('[trainer] Using curated dataset (raw labels trusted):', curated_path)
            candidate_csvs = [curated_path]
        elif os.path.exists(processed_all):
            candidate_csvs = [processed_all]
        else:
            candidate_csvs = [
                'legal_cases.csv',
                'property_disputes.csv',
                'rape.csv',
                'child_labour.csv',
                os.path.join('data', 'rape_murder_cases.csv'),
                os.path.join('data', 'rape_murder_cases_summarized.csv'),
            ]
        existing = [p for p in candidate_csvs if os.path.exists(p)]
        if not existing:
            raise FileNotFoundError("No training CSVs found. Place CSVs with columns ['case_data','case_type','judgement'] in the project.")

        print("\nLoading legal case data from:")
        for p in existing:
            print(f" - {p}")
        df = classifier.load_data(existing)

        # === Quality filters (exclude 'Other' class and too-short texts, optional) ===
        exclude_other = os.getenv('EXCLUDE_OTHER', '1') != '0'
        min_text_len_env = os.getenv('MIN_TEXT_LEN', '0')
        try:
            min_text_len = int(min_text_len_env)
        except ValueError:
            min_text_len = 0
        if min_text_len > 0:
            df = df[df['case_data'].astype(str).str.len() >= min_text_len]
        if exclude_other:
            df = df[df['judgement'].astype(str).str.lower().isin(['unrecognized','other']) == False]
        # lightweight dedupe on canonicalized text
        _canon = (
            df['case_data']
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df = df.loc[~_canon.duplicated()].reset_index(drop=True)

        # === Weak Label Integration (Phase 5) ===
        weak_enabled = bool(os.getenv('USE_WEAK_LABELS'))
        if weak_enabled and 'weak_label' in df.columns:
            WEAK_LABEL_WEIGHT = float(os.getenv('WEAK_LABEL_WEIGHT','0.3'))
            mask = (df['judgement'].astype(str) == 'Other') & (df['weak_label'].astype(str).str.len() > 0)
            applied = int(mask.sum())
            if applied > 0:
                df['original_judgement'] = df['judgement']
                df.loc[mask, 'judgement'] = df.loc[mask, 'weak_label']
                df['__is_weak'] = 0
                df.loc[mask, '__is_weak'] = 1
                # Sample weights: weak instances down-weighted
                df['__sample_weight'] = 1.0
                df.loc[df['__is_weak'] == 1, '__sample_weight'] = WEAK_LABEL_WEIGHT
                print(f"[weak] Converted {applied} 'Other' rows to weak labels (weight={WEAK_LABEL_WEIGHT}).")
            else:
                print('[weak] No applicable weak labels found (mask empty).')
        else:
            df['__is_weak'] = 0

        # === Optional Rare Class Merge (pre-analysis) ===
        rare_merge_info = None
        min_support_env = os.getenv('MERGE_RARE_MIN_SUPPORT')
        if min_support_env:
            try:
                min_support_val = int(min_support_env)
            except ValueError:
                min_support_val = -1
            if min_support_val > 0:
                counts_before = df['judgement'].value_counts().to_dict()
                rare_labels = [lbl for lbl,cnt in counts_before.items() if cnt < min_support_val and lbl not in {'unrecognized','Other'}]
                if rare_labels:
                    df.loc[df['judgement'].isin(rare_labels),'judgement'] = 'unrecognized'
                    counts_after = df['judgement'].value_counts().to_dict()
                    rare_merge_info = {
                        'min_support': min_support_val,
                        'merged_labels': rare_labels,
                        'counts_before': counts_before,
                        'counts_after': counts_after,
                    }
                    print(f"[rare-merge] Merged {len(rare_labels)} rare labels into 'unrecognized' (min_support={min_support_val}).")
                else:
                    print(f"[rare-merge] No labels below min_support={min_support_val} to merge.")
            else:
                print("[rare-merge] Invalid MERGE_RARE_MIN_SUPPORT value; skipping.")

        print("\nAnalyzing dataset (pre-balance)...")
        analysis = classifier.analyze_dataset(df)
        original_distribution = analysis.get('judgement_distribution', {}).copy()

        # Optional class rebalancing (simple majority down-sampling)
        balance_max_ratio = os.getenv('BALANCE_MAX_RATIO')
        balanced_info: dict[str, Any] | None = None
        if balance_max_ratio:
            try:
                max_ratio = float(balance_max_ratio)
            except ValueError:
                max_ratio = None
            if max_ratio and max_ratio > 0:
                print(f"\nApplying down-sampling with max ratio={max_ratio} (env BALANCE_MAX_RATIO)")
                # Compute counts per class (leaf id)
                counts = df['judgement'].value_counts().to_dict()
                min_count = min(counts.values()) if counts else 0
                target_counts = {c: min(min_count * max_ratio, cnt) for c, cnt in counts.items()}
                # Down-sample each class
                parts = []
                rng = np.random.default_rng(seed=42)
                for cls, target in target_counts.items():
                    subset = df[df['judgement'] == cls]
                    if len(subset) > target:
                        take = int(target)
                        subset = subset.sample(n=take, random_state=42)
                    parts.append(subset)
                df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
                # Re-run analysis on balanced df
                post_analysis = classifier.analyze_dataset(df)
                balanced_info = {
                    'strategy': 'downsample_max_ratio',
                    'max_ratio': max_ratio,
                    'original_distribution': original_distribution,
                    'final_distribution': post_analysis.get('judgement_distribution'),
                }
                analysis = post_analysis
                print("Balanced class distribution:")
                print(balanced_info['final_distribution'])
            else:
                print("BALANCE_MAX_RATIO provided but invalid; skipping balancing.")
        print(f"Total cases: {analysis['total_cases']}")
        print(f"Case types: {analysis['case_types']}")
        print(f"Judgement distribution (ontology leaf ids): {analysis['judgement_distribution']}")
        print(f"Classes: {analysis['num_classes']} | Min samples/class: {analysis['min_samples_per_class']}")
        print(f"Duplicate ratio: {analysis['duplicate_ratio']:.4f}")

        # === Optional holdout split BEFORE any upsampling (unbiased eval) ===
        holdout_df = None
        holdout_info: dict[str, Any] | None = None
        holdout_fraction_env = os.getenv('HOLDOUT_FRACTION')
        if holdout_fraction_env:
            try:
                hold_frac = float(holdout_fraction_env)
            except ValueError:
                hold_frac = -1.0
            if 0.0 < hold_frac < 0.5 and analysis['min_samples_per_class'] >= 2:
                from sklearn.model_selection import StratifiedShuffleSplit
                print(f"\nCreating holdout split (fraction={hold_frac}) before upsampling...")
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=hold_frac, random_state=42)
                y_labels = df['judgement'].astype(str)
                (train_idx, hold_idx), = list(splitter.split(df, y_labels))
                holdout_df = df.iloc[hold_idx].reset_index(drop=True)
                df = df.iloc[train_idx].reset_index(drop=True)
                holdout_info = {
                    'fraction': hold_frac,
                    'size': int(len(holdout_df)),
                    'distribution': holdout_df['judgement'].value_counts().to_dict()
                }
                print(f"Holdout size: {holdout_info['size']} | Distribution: {holdout_info['distribution']}")
            else:
                print("Holdout requested but skipped (invalid fraction or insufficient per-class support).")

        # Capture distribution prior to any upsampling
        pre_upsampling_distribution = df['judgement'].value_counts().to_dict()

        # Temporal evaluation scaffold (older vs newer) if retrieval_ts present
        temporal_eval = None
        if 'retrieval_ts' in df.columns:
            try:
                ts_ser = pd.to_datetime(df['retrieval_ts'], errors='coerce')
                valid = df[ts_ser.notna()].copy()
                if len(valid) > 40 and valid['judgement'].nunique() > 1:
                    valid = valid.sort_values('retrieval_ts')
                    cut = int(len(valid) * 0.8)
                    older_slice = valid.iloc[:cut]
                    newer_slice = valid.iloc[cut:]
                    if older_slice['judgement'].nunique() >= 2 and newer_slice['judgement'].nunique() >= 2:
                        # Lightweight logistic baseline for drift signal
                        def _prep(df_part):
                            txt = (df_part['case_type'].astype(str).str.lower() + ' ' + df_part['case_data'].astype(str))
                            return txt.apply(classifier.preprocess_text)
                        X_old = _prep(older_slice)
                        X_new = _prep(newer_slice)
                        le_tmp = LabelEncoder()
                        y_old = le_tmp.fit_transform(older_slice['judgement'].astype(str))
                        # Map unknown new labels to first class to keep shape stable (rare in curated path)
                        y_new = le_tmp.transform(newer_slice['judgement'].where(newer_slice['judgement'].isin(le_tmp.classes_), le_tmp.classes_[0]).astype(str))
                        vec_tmp = TfidfVectorizer(max_features=4000, ngram_range=(1,2), sublinear_tf=True)
                        X_old_vec = vec_tmp.fit_transform(X_old)
                        X_new_vec = vec_tmp.transform(X_new)
                        lr_tmp = LogisticRegression(max_iter=400, class_weight='balanced', multi_class='ovr')
                        lr_tmp.fit(X_old_vec, y_old)
                        from sklearn.metrics import f1_score as _f1
                        preds_new = lr_tmp.predict(X_new_vec)
                        temporal_eval = {
                            'older_train_rows': int(len(older_slice)),
                            'newer_test_rows': int(len(newer_slice)),
                            'newer_macro_f1': float(_f1(y_new, preds_new, average='macro'))
                        }
                        print(f"[temporal] newer_macro_f1={temporal_eval['newer_macro_f1']:.3f} (older={len(older_slice)} newer={len(newer_slice)})")
            except Exception as _te:
                print(f"[temporal] skipped: {_te}")

        # Optional minority upsampling (simple replicate with replacement) AFTER holdout removal & balancing
        upsample_info = None
        if os.getenv('UPSAMPLE_MAX_TARGET'):
            try:
                import math
                max_target = int(os.getenv('UPSAMPLE_MAX_TARGET','50'))
                ratio = float(os.getenv('UPSAMPLE_RATIO','0.3'))  # fraction of majority cap
                min_support_for_upsample = int(os.getenv('UPSAMPLE_MIN_SUPPORT','5'))
                counts = df['judgement'].value_counts().to_dict()
                majority = max(counts.values()) if counts else 0
                desired_target = max(1, min(max_target, int(math.ceil(majority * ratio))))
                before_counts = counts.copy()
                parts = []
                rng = np.random.default_rng(seed=42)
                for label, subset in df.groupby('judgement'):
                    subset_rows = subset
                    c = len(subset_rows)
                    if c >= min_support_for_upsample and c < desired_target:
                        needed = desired_target - c
                        take_idx = rng.choice(subset_rows.index.to_numpy(), size=needed, replace=True)
                        dup_rows = subset_rows.loc[take_idx].copy()
                        subset_rows = pd.concat([subset_rows, dup_rows], ignore_index=True)
                    parts.append(subset_rows)
                if parts:
                    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
                    after_counts = df['judgement'].value_counts().to_dict()
                    upsample_info = {
                        'strategy': 'simple_replicate',
                        'max_target': max_target,
                        'ratio': ratio,
                        'desired_target': desired_target,
                        'min_support_for_upsample': min_support_for_upsample,
                        'before': before_counts,
                        'after': after_counts,
                    }
                    print("Applied upsampling:")
                    print({k: after_counts[k] for k in sorted(after_counts)})
            except Exception as ue:  # pragma: no cover
                print(f"Upsampling skipped (error): {ue}")

        print("\nPreparing data...")
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        sample_weight_train = getattr(classifier, '_train_weights', None)
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        print("\nTraining boosted RandomForest model...")
        classifier.model, train_f1 = classifier.train_model(X_train, y_train, sample_weight=sample_weight_train)

        # Start MLflow run (if available)
        mlflow_run = None
        if mlflow is not None:
            try:
                mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT','legal_outcomes'))
                mlflow_run = mlflow.start_run(run_name=f"rf_boost_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                mlflow.log_param('model_type','AdaBoost(RandomForest)')
                mlflow.log_param('n_classes', analysis['num_classes'])
                mlflow.log_params({'ontology_version': ontology_metadata().get('version'), 'num_ontology_leaves': ontology_metadata().get('num_leaves')})
            except Exception as e:
                print(f"MLflow logging disabled: {e}")

        # Optional: quick cross-validation on training fold for signal
        cv_scores = None
        try:
            skf = StratifiedKFold(n_splits=min(5, len(np.unique(y_train))), shuffle=True, random_state=42)
            # Use a lean version for speed (vectorizer shared settings)
            cv_vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
            cv_rf = RandomForestClassifier(n_estimators=150, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
            from sklearn.pipeline import Pipeline as _P
            cv_pipe = _P([('v', cv_vec), ('c', cv_rf)])
            cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
            print(f"CV macro-F1 (train split): mean={cv_scores.mean():.3f} +- {cv_scores.std():.3f}")
        except Exception as e:
            print(f"CV skipped: {e}")

        print("\nEvaluating boosted RF model...")
        eval_pre_cal = classifier.evaluate(classifier.model, X_test, y_test)
        # Collect predictions for richer metrics
        y_test_pred_initial = None
        if len(X_test) > 0:
            try:
                y_test_pred_initial = classifier.model.predict(X_test)
            except Exception:
                y_test_pred_initial = None

        # Try probability calibration only if each class has sufficient samples; safeguard against degradation
        eval_post_cal = eval_pre_cal
        try:
            cal_min = int(os.getenv('CALIBRATION_MIN_SAMPLES','10'))
        except ValueError:
            cal_min = 10
        train_label_counts = pd.Series(y_train).value_counts().to_dict()
        if min(train_label_counts.values()) >= cal_min and classifier.model is not None:
            try:
                print("\nCalibrating probabilities (Platt scaling)...")
                # Save pre-cal state for potential revert
                pre_cal_model = classifier.model
                pre_cal_macro = None
                try:
                    from sklearn.metrics import f1_score as _f1
                    if y_test_pred_initial is not None:
                        pre_cal_macro = float(_f1(y_test, y_test_pred_initial, average='macro'))
                except Exception:
                    pre_cal_macro = None
                calibrator = CalibratedClassifierCV(classifier.model.named_steps['classifier'], method='sigmoid', cv=3)
                vec = classifier.model.named_steps['vectorizer']
                from sklearn.pipeline import Pipeline as _P
                calibrated = _P([('vectorizer', vec), ('classifier', calibrator)])
                calibrated.fit(X_train, y_train)
                classifier.model = calibrated
                eval_post_cal = classifier.evaluate(classifier.model, X_test, y_test)
                # Degradation safeguard
                try:
                    tol = float(os.getenv('CAL_DEGRADE_TOL','0.01'))
                except ValueError:
                    tol = 0.01
                if pre_cal_macro is not None:
                    try:
                        post_preds = classifier.model.predict(X_test)
                        from sklearn.metrics import f1_score as _f1
                        post_macro = float(_f1(y_test, post_preds, average='macro'))
                        if (pre_cal_macro - post_macro) > tol:
                            print(f"Calibration degraded macro-F1 ({pre_cal_macro:.3f} -> {post_macro:.3f}) > tol={tol}; reverting to pre-cal model.")
                            classifier.model = pre_cal_model
                            eval_post_cal = eval_pre_cal
                    except Exception as _ce:
                        print(f"Calibration safeguard skipped: {_ce}")
            except Exception as e:
                print(f"Calibration skipped: {e}")
        else:
            print(f"Calibration skipped: min class train support < {cal_min}")

        print("\nTraining Logistic Regression baseline (side-by-side)...")
        baseline = classifier.train_logreg_baseline(X_train, y_train)
        print("\nEvaluating Logistic Regression baseline...")
        baseline_eval = classifier.evaluate(baseline, X_test, y_test)

        # Compare macro F1 (prefer macro for tail sensitivity) for model selection
        from sklearn.metrics import f1_score as _f1
        boosted_macro = None
        baseline_macro = None
        try:
            if y_test_pred_initial is not None:
                boosted_macro = float(_f1(y_test, y_test_pred_initial, average='macro'))
            baseline_preds = baseline.predict(X_test)
            baseline_macro = float(_f1(y_test, baseline_preds, average='macro'))
        except Exception:
            pass
        selected_model = 'boosted_rf'
        selection_reason = 'default'
        if os.getenv('SELECT_BEST_MODEL') and baseline_macro is not None and boosted_macro is not None:
            if baseline_macro >= boosted_macro + float(os.getenv('MODEL_SELECT_MARGIN','0.0')):
                classifier.model = baseline  # adopt baseline
                selected_model = 'logreg'
                selection_reason = 'higher_macro_f1'
                print(f"[selection] Logistic Regression chosen over boosted RF (macro-F1 {baseline_macro:.3f} vs {boosted_macro:.3f})")

        # Head/Tail metrics (based on training distribution BEFORE any potential upsampling info for conceptual grouping)
        train_counts_labels = df['judgement'].value_counts()
        median_support = train_counts_labels.median() if len(train_counts_labels) else 0
        head_labels = set(train_counts_labels[train_counts_labels > median_support].index)
        tail_labels = set(train_counts_labels[train_counts_labels <= median_support].index)
        head_macro_f1 = None
        tail_macro_f1 = None
        if classifier.label_encoder is not None and len(X_test) > 0:
            try:
                final_preds = classifier.model.predict(X_test)  # type: ignore[arg-type]
                inv = classifier.label_encoder.inverse_transform
                y_test_labels = [inv([y])[0] for y in y_test]
                import numpy as _np
                def macro_f1_for(label_set):
                    idx = [i for i,l in enumerate(y_test_labels) if l in label_set]
                    if not idx:
                        return None
                    y_true_sub = [_np.array(y_test)[i] for i in idx]
                    y_pred_sub = [final_preds[i] for i in idx]
                    return float(_f1(y_true_sub, y_pred_sub, average='macro'))
                head_macro_f1 = macro_f1_for(head_labels)
                tail_macro_f1 = macro_f1_for(tail_labels)
            except Exception:
                pass

        # === Abstention Metrics (Phase 5) ===
        abstention_metrics = None
        try:
            from math import isfinite
            p_thresh = float(os.getenv('ABSTAIN_PROB_THRESHOLD','0.35'))
            m_thresh = float(os.getenv('ABSTAIN_MARGIN_THRESHOLD','0.10'))
            if classifier.model is not None and len(X_test) > 0:
                proba = classifier.model.predict_proba(X_test)
                top2 = np.sort(proba, axis=1)[:, -2:]
                top1 = top2[:,1]
                second = top2[:,0]
                margin = top1 - second
                abstain = (top1 < p_thresh) | (margin < m_thresh)
                coverage = float((~abstain).mean())
                abstain_rate = float(abstain.mean())
                # Macro-F1 on non-abstained subset
                if (~abstain).sum() > 0:
                    kept_idx = np.where(~abstain)[0]
                    kept_true = y_test[kept_idx]
                    kept_pred = classifier.model.predict(X_test.iloc[kept_idx])  # type: ignore[index]
                    from sklearn.metrics import f1_score as _f1
                    kept_macro = float(_f1(kept_true, kept_pred, average='macro'))
                else:
                    kept_macro = None
                abstention_metrics = {
                    'prob_threshold': p_thresh,
                    'margin_threshold': m_thresh,
                    'abstain_rate': abstain_rate,
                    'coverage': coverage,
                    'macro_f1_non_abstained': kept_macro,
                }
                print(f"[abstain] rate={abstain_rate:.3f} coverage={coverage:.3f} macro_f1_non_abstained={kept_macro}")
        except Exception as ae:
            print(f"[abstain] metrics skipped: {ae}")

    # Save model under models/ with run history
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{uuid.uuid4().hex[:8]}"  # local time acceptable for run_id ordering
        run_dir = os.path.join(models_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "legal_case_classifier.pkl")
        classifier.save_model(out_path)
        print(f"\nModel saved to: {out_path}")

        # === Evaluation Artifacts ===
        try:
            os.makedirs("models", exist_ok=True)
            # Confusion matrix & rich metrics (final model)
            if classifier.label_encoder is not None and len(X_test) > 0:
                try:
                    y_test_pred = classifier.model.predict(X_test)  # type: ignore[arg-type]
                except Exception:
                    y_test_pred = None
                if y_test_pred is not None:
                    cm = confusion_matrix(y_test, y_test_pred).tolist()
                    labels = list(classifier.label_encoder.classes_)
                    # Additional metrics
                    try:
                        from sklearn.metrics import f1_score, classification_report
                        macro_f1 = float(f1_score(y_test, y_test_pred, average='macro'))
                        weighted_f1 = float(f1_score(y_test, y_test_pred, average='weighted'))
                        report_dict = classification_report(y_test, y_test_pred, target_names=labels, output_dict=True, zero_division=0)
                        per_class_f1 = {lbl: float(report_dict.get(lbl, {}).get('f1-score', 0.0)) for lbl in labels}
                    except Exception:
                        macro_f1 = None
                        weighted_f1 = None
                        per_class_f1 = {}
                else:
                    cm, labels, macro_f1, weighted_f1, per_class_f1 = [], [], None, None, {}
            else:
                cm, labels, macro_f1, weighted_f1, per_class_f1 = [], [], None, None, {}

            # Dataset hash for provenance
            hash_source = df[['case_data','case_type','judgement']].astype(str).to_csv(index=False)
            dataset_hash = hashlib.sha256(hash_source.encode('utf-8')).hexdigest()

            # Optional holdout evaluation
            holdout_metrics = None
            if holdout_df is not None and classifier.label_encoder is not None:
                try:
                    hold_text = (holdout_df['case_type'].astype(str).str.lower() + ' ' + holdout_df['case_data'].astype(str))
                    hold_proc = hold_text.apply(classifier.preprocess_text)
                    hold_preds = classifier.model.predict(hold_proc)  # type: ignore[arg-type]
                    hold_y = classifier.label_encoder.transform(holdout_df['judgement'].astype(str))
                    from sklearn.metrics import f1_score as _f1
                    hold_acc = float(accuracy_score(hold_y, hold_preds))
                    hold_macro = float(_f1(hold_y, hold_preds, average='macro'))
                    hold_weighted = float(_f1(hold_y, hold_preds, average='weighted'))
                    holdout_metrics = {
                        'accuracy': hold_acc,
                        'macro_f1': hold_macro,
                        'weighted_f1': hold_weighted,
                        'size': len(holdout_df)
                    }
                    print(f"\nHoldout Accuracy: {hold_acc:.4f} | Macro-F1: {hold_macro:.3f} | Weighted-F1: {hold_weighted:.3f}")
                except Exception as he:
                    print(f"Holdout evaluation failed: {he}")

            # Slice metrics (case_type & length buckets) if possible
            slice_metrics = None
            try:
                if classifier.label_encoder is not None and y_test_pred is not None and 'case_type' in df.columns:
                    from importlib import import_module as _imp
                    slices_mod = _imp('evaluation.slices')
                    # Build a dataframe aligned with test indices by matching processed_text tokens
                    if 'processed_text' in df.columns:
                        # Build mapping from processed text to indices (many-to-one collapse risk handled by subset)
                        processed_set = set(X_test)
                        test_df = df[df['processed_text'].isin(processed_set)].copy()
                    else:
                        test_df = df.head(0)
                    slice_metrics = slices_mod.compute_slices(test_df.reset_index(drop=True), y_test, y_test_pred, classifier.label_encoder)
            except Exception as se:  # pragma: no cover
                slice_metrics = {'error': str(se)}

            metrics: Dict[str, Any] = {
                "final_model": {
                    "train_f1_weighted": train_f1,
                    "test_accuracy": eval_post_cal.get('accuracy'),
                    "cv_macro_f1_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                    "cv_macro_f1_std": float(cv_scores.std()) if cv_scores is not None else None,
                    "num_classes": int(classifier.label_encoder.classes_.shape[0]) if classifier.label_encoder else None,
                    "classes": list(classifier.label_encoder.classes_) if classifier.label_encoder else None,
                    "test_macro_f1": macro_f1,
                    "test_weighted_f1": weighted_f1,
                    "per_class_f1": per_class_f1,
                    "ontology": ontology_metadata(),
                    "model_selected": selected_model,
                    "selection_reason": selection_reason,
                    "boosted_macro_f1_pre_selection": boosted_macro,
                    "baseline_macro_f1": baseline_macro,
                    "head_macro_f1": head_macro_f1,
                    "tail_macro_f1": tail_macro_f1,
                    "holdout": holdout_metrics,
                    "abstention": abstention_metrics,
                    "slices": slice_metrics,
                    "temporal_eval": temporal_eval,
                },
                "baseline": {
                    "test_accuracy": baseline_eval.get('accuracy') if 'baseline_eval' in locals() else None
                },
                "data": {
                    "duplicate_ratio": analysis.get('duplicate_ratio'),
                    "total_cases": analysis.get('total_cases'),
                    "pre_upsampling_distribution": pre_upsampling_distribution,
                    "weak_label_applied_rows": int(df['__is_weak'].sum()) if '__is_weak' in df.columns else 0,
                }
            }
            if upsample_info:
                metrics['upsampling'] = upsample_info
            if balanced_info:
                metrics['balancing'] = balanced_info
            if holdout_info:
                metrics['holdout_info'] = holdout_info
            if rare_merge_info:
                metrics['rare_merge'] = rare_merge_info

            with open(os.path.join(run_dir,'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(run_dir,'confusion_matrix.json'), 'w', encoding='utf-8') as f:
                json.dump({"labels": labels, "matrix": cm}, f, indent=2)

            # Metadata summary
            # Determine previous run for lineage
            runs_root = os.path.join(models_dir, 'runs')
            previous_run = None
            try:
                existing_runs = sorted([d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root,d))])
                prior = [r for r in existing_runs if r < run_id]
                if prior:
                    previous_run = prior[-1]
            except Exception:
                previous_run = None

            metadata = {
                "model_path": out_path,
                "trained_at": datetime.now().astimezone().isoformat(),
                "dataset_hash": dataset_hash,
                "dataset_rows": int(len(df)),
                "num_classes": metrics['final_model']['num_classes'],
                "classes": metrics['final_model']['classes'],
                "class_distribution": analysis.get('judgement_distribution'),
                "duplicate_ratio": analysis.get('duplicate_ratio'),
                "train_f1_weighted": train_f1,
                "test_accuracy": metrics['final_model']['test_accuracy'],
                "cv_macro_f1_mean": metrics['final_model']['cv_macro_f1_mean'],
                "cv_macro_f1_std": metrics['final_model']['cv_macro_f1_std'],
                "test_macro_f1": metrics['final_model']['test_macro_f1'],
                "test_weighted_f1": metrics['final_model']['test_weighted_f1'],
                "head_macro_f1": metrics['final_model']['head_macro_f1'],
                "tail_macro_f1": metrics['final_model']['tail_macro_f1'],
                "pre_upsampling_distribution": metrics['data'].get('pre_upsampling_distribution'),
                "post_upsampling_distribution": (metrics.get('upsampling') or {}).get('after'),
                "run_id": run_id,
                "previous_run": previous_run,
                "ontology": ontology_metadata(),
                "holdout_metrics": holdout_metrics,
            }
            if balanced_info:
                metadata['balancing'] = balanced_info
            if rare_merge_info:
                metadata['rare_merge'] = rare_merge_info
            with open(os.path.join(run_dir,'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            # Update top-level latest copies for API consumption
            for name in ["legal_case_classifier.pkl", "metrics.json", "confusion_matrix.json", "metadata.json"]:
                src = os.path.join(run_dir, name)
                dst = os.path.join(models_dir, name)
                try:
                    # Copy binary or text generically
                    import shutil
                    shutil.copy2(src, dst)
                except Exception as ce:
                    print(f"Copy {name} failed: {ce}")

            # Append to history log (JSON lines)
            try:
                history_path = os.path.join(models_dir, 'history.log')
                with open(history_path, 'a', encoding='utf-8') as hf:
                    hf.write(json.dumps(metadata) + "\n")
            except Exception as he:
                print(f"History log append failed: {he}")
            print("Saved evaluation artifacts: metrics.json, confusion_matrix.json, metadata.json")
            # MLflow metrics logging (best-effort)
            if mlflow is not None and mlflow_run is not None:
                try:
                    final = metrics['final_model']
                    mlflow.log_metrics({
                        'test_accuracy': final.get('test_accuracy') or 0.0,
                        'test_macro_f1': final.get('test_macro_f1') or 0.0,
                        'test_weighted_f1': final.get('test_weighted_f1') or 0.0,
                        'train_f1_weighted': final.get('train_f1_weighted') or 0.0,
                    })
                    # Log per-class f1 as separate metrics
                    pc = final.get('per_class_f1') or {}
                    for lbl, val in pc.items():
                        # Sanitize metric name
                        key = 'f1_' + lbl.lower().replace(' ','_').replace('/','_')
                        mlflow.log_metric(key, val)
                except Exception as le:
                    print(f"MLflow metric logging failed: {le}")
        except Exception as e:
            print(f"Artifact persistence skipped: {e}")
        finally:
            if mlflow is not None and mlflow_run is not None:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        # Quick smoke prediction
        try:
            demo_text = (
                "The accused was found with the weapon and ballistic evidence matched; key witness testimony corroborated the events."
            )
            pred = classifier.predict_judgement(demo_text, "Criminal")
            print(f"Sample prediction: {pred}")
        except Exception:
            pass

        # After metrics are computed and abstention_metrics is available
        if args.abstention_report and abstention_metrics is not None and classifier.model is not None and len(X_test) > 0:
            proba = classifier.model.predict_proba(X_test)
            top2 = np.sort(proba, axis=1)[:, -2:]
            top1 = top2[:,1]
            second = top2[:,0]
            margin = top1 - second
            p_thresh = abstention_metrics['prob_threshold']
            m_thresh = abstention_metrics['margin_threshold']
            abstain = (top1 < p_thresh) | (margin < m_thresh)
            kept_idx = np.where(~abstain)[0]
            abstained_idx = np.where(abstain)[0]
            print('\n[Abstention Confusion Summary]')
            print(f'Total test samples: {len(X_test)}')
            print(f'Kept (non-abstained): {len(kept_idx)}')
            print(f'Abstained: {len(abstained_idx)}')
            if len(kept_idx) > 0:
                kept_true = y_test[kept_idx]
                kept_pred = classifier.model.predict(X_test.iloc[kept_idx])
                from sklearn.metrics import f1_score, accuracy_score
                print(f'Kept Macro-F1: {f1_score(kept_true, kept_pred, average="macro"):.3f}')
                print(f'Kept Accuracy: {accuracy_score(kept_true, kept_pred):.3f}')
            if len(abstained_idx) > 0:
                abstained_true = y_test[abstained_idx]
                print(f'Abstained true label distribution: {pd.Series(abstained_true).value_counts().to_dict()}')
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()


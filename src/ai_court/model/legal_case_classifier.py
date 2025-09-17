import os
import re
import warnings
from typing import List, Tuple, Dict

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
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
        # Normalize outcome labels into coarse classes for learnability
        df = df[required_cols].copy()
        df['judgement'] = df['judgement'].astype(str).apply(self.normalize_outcome)
        return df

    @staticmethod
    def normalize_outcome(text: str) -> str:
        """Map raw judgment text to a manageable set of outcome classes."""
        t = (text or "").lower()
        # Strong class indicators
        if any(k in t for k in ["acquitted", "acquittal", "conviction overturned", "set aside", "reversal of conviction", "benefit of doubt"]):
            return "Acquittal/Conviction Overturned"
        if any(k in t for k in ["appeal dismissed", "conviction upheld", "conviction affirmed", "convictions upheld", "life sentence upheld", "appeal fails"]):
            return "Conviction Upheld/Appeal Dismissed"
        if any(k in t for k in ["quash", "quashed", "quashing", "fir quashed", "charges quashed"]):
            return "Charges/Proceedings Quashed"
        if any(k in t for k in ["sentence reduced", "sentence modified", "reduced sentence", "commuted", "converted to", "altered sentence"]):
            return "Sentence Reduced/Modified"
        if any(k in t for k in ["bail granted", "anticipatory bail granted", "interim bail"]):
            return "Bail Granted"
        if any(k in t for k in ["bail denied", "bail rejected", "bail refused"]):
            return "Bail Denied"
        if any(k in t for k in ["petition allowed", "writ allowed", "relief granted", "granted protection", "mandated", "directed", "ordered", "convicted"]):
            return "Relief Granted/Convicted"
        if any(k in t for k in ["petition dismissed", "relief denied", "dismissed as infructuous", "dismissed on merits", "dismissed"]):
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
        analysis = {
            'total_cases': int(len(df)),
            'case_types': df['case_type'].value_counts().to_dict(),
            'judgement_distribution': df['judgement'].value_counts().to_dict(),
            'min_samples_per_class': int(df['judgement'].value_counts().min()),
            'num_classes': int(df['judgement'].nunique()),
        }
        if analysis['num_classes'] < 2:
            raise ValueError("At least 2 judgement classes required")
        return analysis

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """Build text features and encode labels."""
        # Combine simple structured signal + raw narrative
        df['legal_features'] = df['case_type'].astype(str).str.lower() + " " + df['case_data'].astype(str)
        df['processed_text'] = df['legal_features'].apply(self.preprocess_text)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['judgement'].astype(str))

        # Ensure at least a few test samples, but keep stratification when possible
        test_size = 0.2 if len(df) >= 25 else max(0.15, min(0.2, 3 / len(df)))
        # Use stratify only if every class has at least 2 samples
        value_counts = pd.Series(y).value_counts()
        can_stratify = (value_counts.min() >= 2)
        stratify = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], y, test_size=test_size, random_state=42, stratify=stratify
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.Series, y_train: np.ndarray) -> Pipeline:
        """Train TF-IDF + AdaBoost(RandomForest) pipeline as requested."""
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

        # scikit-learn >=1.2 uses 'estimator'; older versions used 'base_estimator'.
        try:
            boosted_rf = AdaBoostClassifier(
                estimator=rf_base,
                n_estimators=10,           # number of boosting rounds over RF base learner
                learning_rate=0.5,
                algorithm='SAMME',         # use SAMME for compatibility across sklearn versions
                random_state=42,
            )
        except TypeError:
            # Fallback for older sklearn
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

        pipeline.fit(X_train, y_train)

        # Report a quick training F1 (can indicate overfitting if very high)
        train_pred = pipeline.predict(X_train)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        print(f"Training F1 score (weighted): {train_f1:.3f}")
        return pipeline

    def evaluate(self, model: Pipeline, X_test: pd.Series, y_test: np.ndarray) -> Dict:
        if len(X_test) == 0:
            print("Warning: No test samples available")
            return {'accuracy': 0.0, 'report': ''}

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
            self.preprocess_text = saved.get('preprocessor', self.preprocess_text)

    def predict_judgement(self, case_data: str, case_type: str) -> str:
        if not self.model:
            raise ValueError("Model not trained or loaded")
        legal_input = f"{str(case_type).lower()} {str(case_data)}"
        processed_input = self.preprocess_text(legal_input)
        pred = self.model.predict([processed_input])[0]
        return self.label_encoder.inverse_transform([pred])[0]


def main():
    """Train and persist the boosted RandomForest model using all compatible CSVs in the repo."""
    try:
        print("=== Legal Case Outcome Model Trainer (RF + Boost) ===")
        classifier = LegalCaseClassifier()

        # Discover CSVs that likely contain the right schema
        processed_all = os.path.join('data', 'processed', 'all_cases.csv')
        if os.path.exists(processed_all):
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

        print("\nAnalyzing dataset...")
        analysis = classifier.analyze_dataset(df)
        print(f"Total cases: {analysis['total_cases']}")
        print(f"Case types: {analysis['case_types']}")
        print(f"Judgement distribution: {analysis['judgement_distribution']}")
        print(f"Classes: {analysis['num_classes']} | Min samples/class: {analysis['min_samples_per_class']}")

        print("\nPreparing data...")
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        print("\nTraining boosted RandomForest model...")
        classifier.model = classifier.train_model(X_train, y_train)

        print("\nEvaluating model...")
        _ = classifier.evaluate(classifier.model, X_test, y_test)

        # Save model under models/
        os.makedirs(os.path.join("models"), exist_ok=True)
        out_path = os.path.join("models", "legal_case_classifier.pkl")
        classifier.save_model(out_path)
        print(f"\nModel saved to: {out_path}")

        # Quick smoke prediction
        try:
            demo_text = (
                "The accused was found with the weapon and ballistic evidence matched; key witness testimony corroborated the events."
            )
            pred = classifier.predict_judgement(demo_text, "Criminal")
            print(f"Sample prediction: {pred}")
        except Exception:
            pass

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()


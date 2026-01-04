import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ai_court.model.preprocessor import TextPreprocessor
from ai_court.ontology import map_coarse_label

class DataLoader:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        self._train_weights = None
        self._test_weights = None

    def _read_single(self, path: str) -> pd.DataFrame:
        """Read a single CSV and normalize columns."""
        df = pd.read_csv(path)
        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        
        # Map text column
        if 'case_data' not in cols and 'text' in cols:
            df.rename(columns={cols['text']: 'case_data'}, inplace=True)
        elif 'summary' in cols:
            df.rename(columns={cols['summary']: 'case_data'}, inplace=True)
            
        # Map type column
        if 'case_type' not in cols and 'type' in cols:
            df.rename(columns={cols['type']: 'case_type'}, inplace=True)
            
        # Map label column
        if 'judgement' not in cols and 'outcome' in cols:
            df.rename(columns={cols['outcome']: 'judgement'}, inplace=True)
        elif 'judgment' in cols:
            df.rename(columns={cols['judgment']: 'judgement'}, inplace=True)

        # Ensure required columns exist
        required_cols = ['case_data', 'case_type', 'judgement']
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            raise ValueError(f"Missing columns {missing} in {path}")

        # Clean empty rows
        df = df.dropna(subset=required_cols)
        df = df[df['case_data'].astype(str).str.strip().astype(bool)]
        df = df[df['judgement'].astype(str).str.strip().astype(bool)]
        
        # Preserve optional weak supervision columns
        optional = [c for c in ['weak_label','weak_label_sources','weak_label_count'] if c in df.columns]
        df = df[required_cols + optional].copy()
        
        use_curated = bool(os.getenv('USE_CURATED_LABELS'))
        if use_curated:
            df['coarse_judgement'] = df['judgement'].astype(str)
            # Phase 2 overrides logic omitted for brevity, can be re-added if needed
        else:
            # Map normalization -> ontology leaf id
            df['coarse_judgement'] = df['judgement'].astype(str).apply(self.preprocessor.normalize_outcome)
            df['judgement'] = df['coarse_judgement'].apply(lambda x: map_coarse_label(x)[0])
            
        return df

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and concatenate datasets from multiple CSVs."""
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
        """Compute basic dataset stats."""
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
        """Build text features and encode labels."""
        df['legal_features'] = df['case_type'].astype(str).str.lower() + " " + df['case_data'].astype(str)
        df['processed_text'] = df['legal_features'].apply(self.preprocessor.preprocess)

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
            self._train_weights = w_train
            self._test_weights = w_test
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], y, test_size=test_size, random_state=42, stratify=stratify
            )
            return X_train, X_test, y_train, y_test

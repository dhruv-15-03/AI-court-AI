import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
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
        
        # Map text column — handle various column names from different sources
        if 'case_data' not in cols:
            for alt in ['case_summary', 'text', 'summary', 'doc', 'case_text']:
                if alt in cols:
                    df.rename(columns={cols[alt]: 'case_data'}, inplace=True)
                    break
            
        # Map type column — handle various column names
        if 'case_type' not in cols:
            for alt in ['type', 'category', 'case_category']:
                if alt in cols:
                    df.rename(columns={cols[alt]: 'case_type'}, inplace=True)
                    break
            else:
                # If no type column exists, derive from filename
                basename = os.path.splitext(os.path.basename(path))[0]
                df['case_type'] = basename.replace('kanoon_', '').replace('_bulk', '')
            
        # Map label column - prefer refined_label if available (AI-curated)
        if 'refined_label' in df.columns:
            df['judgement'] = df['refined_label']
        elif 'judgement' not in cols and 'outcome' in cols:
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
            
            # For rows that landed in "Other", try to infer label from the filename
            # e.g. kanoon_bail_granted.csv -> "Bail Granted"
            #      kanoon_murder_IPC_302_conviction.csv -> "Relief Granted/Convicted"
            basename = os.path.splitext(os.path.basename(path))[0].lower()
            other_mask = df['judgement'] == 'Other'
            if other_mask.any():
                inferred = self._infer_label_from_filename(basename)
                if inferred and inferred != 'Other':
                    df.loc[other_mask, 'judgement'] = inferred
                elif 'query_source' in df.columns:
                    # For bulk harvest data, also try inferring from the query_source column
                    for idx in df.index[other_mask]:
                        qs = str(df.at[idx, 'query_source']).lower()
                        qs_label = self._infer_label_from_filename(qs.replace(' ', '_'))
                        if qs_label and qs_label != 'Other':
                            df.at[idx, 'judgement'] = qs_label
            
        return df

    @staticmethod
    def _infer_label_from_filename(basename: str) -> str:
        """Infer an outcome label from the CSV filename when text-based normalization fails."""
        # Map filename patterns to known outcome classes
        FILENAME_LABEL_MAP = [
            # Bail
            (["bail_granted", "bail_allowed", "anticipatory_bail"], "Bail Granted"),
            (["bail_denied", "bail_rejected", "bail_refused"], "Bail Denied"),
            # Criminal outcomes
            (["acquittal", "acquitted", "conviction_overturned"], "Acquittal/Conviction Overturned"),
            (["conviction_upheld", "conviction_confirmed", "appeal_dismissed"], "Conviction Upheld/Appeal Dismissed"),
            (["conviction", "convicted", "ipc_302_conviction", "ipc_376_conviction", "pocso_conviction"], "Relief Granted/Convicted"),
            # Quashing
            (["quashed", "quashing", "charges_quashed", "fir_quashed", "section_482"], "Charges/Proceedings Quashed"),
            (["chargesheet_quashed", "charge_sheet_quashed"], "Charge Sheet Quashed"),
            # Sentencing
            (["sentence_reduced", "sentence_modified", "commuted"], "Sentence Reduced/Modified"),
            # Remand
            (["remanded", "sent_back", "remand"], "Case Remanded/Sent Back"),
            # Withdrawn
            (["withdrawn", "not_pressed"], "Petition Withdrawn/Dismissed as Withdrawn"),
            # Civil outcomes — based on query category
            (["allowed", "relief_granted", "writ_allowed", "petition_allowed"], "Relief Granted/Convicted"),
            (["dismissed", "relief_denied", "petition_dismissed"], "Relief Denied/Dismissed"),
        ]
        
        for patterns, label in FILENAME_LABEL_MAP:
            for pattern in patterns:
                if pattern in basename:
                    return label
        return "Other"

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

        # Filter ultra-rare classes (less than MIN_CLASS_SAMPLES) to "Other"
        # This prevents stratified split failures while preserving as many classes as possible
        min_class_samples = int(os.getenv('MIN_CLASS_SAMPLES', '5'))
        value_counts = pd.Series(y).value_counts()
        rare_classes = value_counts[value_counts < min_class_samples].index.tolist()
        if rare_classes:
            # Map rare classes to the 'Other' label
            other_label = None
            for label_str in self.label_encoder.classes_:
                if label_str == 'Other':
                    other_label = self.label_encoder.transform(['Other'])[0]
                    break
            if other_label is not None:
                for rc in rare_classes:
                    y[y == rc] = other_label
                # Re-fit label encoder to remove empty classes
                present_labels = [self.label_encoder.inverse_transform([v])[0] for v in sorted(set(y))]
                self.label_encoder = LabelEncoder()
                label_map = {old: new for new, old in enumerate(sorted(set(y)))}
                inv_map = {old: present_labels[i] for i, old in enumerate(sorted(set(y)))}
                y = np.array([label_map[v] for v in y])
                self.label_encoder.fit(present_labels)
                import logging
                logging.getLogger(__name__).info(
                    "Merged %d rare classes (<%d samples) into 'Other'. Remaining classes: %d",
                    len(rare_classes), min_class_samples, len(present_labels),
                )

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

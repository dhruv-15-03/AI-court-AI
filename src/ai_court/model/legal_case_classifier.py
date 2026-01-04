import os
import re
import warnings
import json
import dill
import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

import logging

try:
    import mlflow
except Exception:
    mlflow = None

from ai_court.model.preprocessor import TextPreprocessor
from ai_court.data.loader import DataLoader
from ai_court.model.trainer import Trainer
from ai_court.ontology import ontology_metadata

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LegalCaseClassifier:
    """
    End-to-end text model to predict legal case judgement/outcome.
    
    This class now acts as a facade over the modular components:
    - TextPreprocessor: For text cleaning and normalization.
    - DataLoader: For loading and preparing datasets.
    - Trainer: For model training and evaluation.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader()
        self.trainer = Trainer()
        
        # Expose for backward compatibility
        self.model = self.trainer.model
        self.label_encoder = self.data_loader.label_encoder
        self.stop_words = self.preprocessor.stop_words
        self.legal_terms = self.preprocessor.legal_terms
        self.lemmatizer = self.preprocessor.lemmatizer

    def preprocess_text(self, text: str) -> str:
        """Normalize and lemmatize text while keeping key legal terms."""
        return self.preprocessor.preprocess(text)

    @staticmethod
    def normalize_outcome(text: str) -> str:
        """Map raw judgment text to a manageable set of outcome classes."""
        return TextPreprocessor.normalize_outcome(text)

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and concatenate datasets."""
        return self.data_loader.load_data(file_paths)

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Compute basic dataset stats."""
        return self.data_loader.analyze_dataset(df)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """Build text features and encode labels."""
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(df)
        # Sync internal state for backward compatibility
        self.label_encoder = self.data_loader.label_encoder
        self._train_weights = self.data_loader._train_weights
        self._test_weights = self.data_loader._test_weights
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.Series, y_train: np.ndarray, sample_weight=None) -> Tuple[Pipeline, float]:
        """Train TF-IDF + AdaBoost(RandomForest) pipeline."""
        pipeline, f1 = self.trainer.train(X_train, y_train, sample_weight)
        self.model = pipeline
        return pipeline, f1

    def train_logreg_baseline(self, X_train: pd.Series, y_train: np.ndarray) -> Pipeline:
        """Train a fast Logistic Regression baseline."""
        return self.trainer.train_logreg_baseline(X_train, y_train)

    def evaluate(self, model: Pipeline, X_test: pd.Series, y_test: np.ndarray) -> Dict:
        return self.trainer.evaluate(model, X_test, y_test, self.label_encoder)

    def save_model(self, filepath: str):
        """Save with dill for compatibility."""
        self.trainer.model = self.model
        self.trainer.save_model(filepath, self.label_encoder)

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            saved = dill.load(f)
            self.model = saved['model']
            self.label_encoder = saved['label_encoder']
            # Ensure preprocessor is available
            self.preprocessor = TextPreprocessor() 
            # If saved has a preprocessor function, we could use it, but we prefer the class method
            
    def predict(self, case_data: str, case_type: str) -> Dict[str, Any]:
        """
        Predict judgment with confidence score.
        
        Args:
            case_data: The text description of the case.
            case_type: The type/category of the case.
            
        Returns:
            Dict containing 'judgment', 'confidence', and 'processed_text'.
            
        Raises:
            ValueError: If model is not loaded or input is invalid.
            RuntimeError: If prediction fails.
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        text = f"{case_type} {case_data}"
        processed = self.preprocess_text(text)
        
        # Check for empty input after processing
        if not processed.strip():
            raise ValueError("Insufficient text data after preprocessing")
            
        try:
            # Try predict_proba first if available
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([processed])[0]
                pred_idx = int(proba.argmax())
                confidence = float(proba.max())
            else:
                pred_idx = int(self.model.predict([processed])[0])
                confidence = None
                
            judgment = self.label_encoder.inverse_transform([pred_idx])[0]
            
            return {
                "judgment": judgment,
                "confidence": confidence,
                "processed_text": processed
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def predict_judgement(self, case_data: str, case_type: str) -> str:
        """
        Legacy method for prediction. Use predict() instead.
        """
        try:
            result = self.predict(case_data, case_type)
            return result["judgment"]
        except Exception as e:
            return f"Error during prediction: {str(e)}"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Train Legal Case Classifier")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing processed CSVs')
    parser.add_argument('--models_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--abstention_report', action='store_true', help='Print abstention analysis')
    args = parser.parse_args()

    # Use the Trainer's pipeline for the main execution flow
    trainer = Trainer()
    
    # Find CSV files
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return

    csv_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and 'backup' not in f
    ]
    
    if not csv_files:
        logger.warning("No CSV files found.")
        return

    logger.info(f"Found {len(csv_files)} CSV files.")
    
    # Run the full pipeline
    try:
        run_dir = trainer.run_training_pipeline(csv_files, args.models_dir)
        logger.info(f"Training pipeline completed successfully. Run artifacts in {run_dir}")
        
        # Optional: Abstention report (re-using logic if needed, but Trainer handles basic eval)
        if args.abstention_report:
            logger.info("Abstention report is not yet fully integrated into the new Trainer pipeline.")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()


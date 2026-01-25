"""Explainability utilities for model predictions.

Provides feature importance extraction from TF-IDF + RandomForest pipeline
to explain which tokens contributed most to a prediction.

Memory-efficient: Uses existing model, no additional loading.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def extract_top_features(
    model_pipeline,
    processed_text: str,
    prediction_idx: int,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Extract top-K features that contributed most to a prediction.
    
    Args:
        model_pipeline: Trained sklearn Pipeline with vectorizer and classifier
        processed_text: Preprocessed text that was used for prediction
        prediction_idx: The predicted class index
        top_k: Number of top features to return
        
    Returns:
        List of dicts with 'term', 'importance', 'direction' keys
    """
    try:
        # Get pipeline components
        vectorizer = model_pipeline.named_steps.get('vectorizer')
        classifier = model_pipeline.named_steps.get('classifier')
        
        if vectorizer is None or classifier is None:
            logger.warning("Pipeline missing vectorizer or classifier")
            return []
        
        # Transform text to TF-IDF features
        tfidf_vector = vectorizer.transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importances based on classifier type
        importances = _get_feature_importances(classifier, tfidf_vector, prediction_idx, len(feature_names))
        
        if importances is None or len(importances) == 0:
            return []
        
        # Get non-zero features from the input
        nonzero_indices = tfidf_vector.nonzero()[1]
        
        # Filter to only features present in input
        feature_scores = []
        for idx in nonzero_indices:
            if idx < len(importances) and idx < len(feature_names):
                tfidf_weight = float(tfidf_vector[0, idx])
                importance = float(importances[idx])
                combined_score = tfidf_weight * importance
                feature_scores.append({
                    'term': str(feature_names[idx]),
                    'importance': round(combined_score, 4),
                    'tfidf_weight': round(tfidf_weight, 4),
                    'feature_importance': round(importance, 4),
                    'direction': 'positive' if combined_score > 0 else 'neutral'
                })
        
        # Sort by absolute importance and take top-k
        feature_scores.sort(key=lambda x: abs(x['importance']), reverse=True)
        return feature_scores[:top_k]
        
    except Exception as e:
        logger.warning(f"Failed to extract features: {e}")
        return []


def _get_feature_importances(
    classifier,
    tfidf_vector,
    prediction_idx: int,
    n_features: int
) -> Optional[np.ndarray]:
    """Extract feature importances from various classifier types."""
    try:
        # RandomForest and tree-based models
        if hasattr(classifier, 'feature_importances_'):
            return classifier.feature_importances_
        
        # AdaBoost wrapping RandomForest
        if hasattr(classifier, 'estimators_'):
            # Average feature importances across base estimators
            all_importances = []
            for estimator in classifier.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)
            if all_importances:
                return np.mean(all_importances, axis=0)
        
        # Logistic Regression (use coefficients)
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_
            if len(coef.shape) > 1:
                # Multi-class: use coefficients for predicted class
                if prediction_idx < coef.shape[0]:
                    return np.abs(coef[prediction_idx])
                return np.abs(coef[0])
            return np.abs(coef)
        
        logger.debug(f"Classifier type {type(classifier)} not supported for importance extraction")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to get feature importances: {e}")
        return None


def format_explanation(
    key_factors: List[Dict[str, Any]],
    judgment: str,
    confidence: Optional[float]
) -> str:
    """Format a human-readable explanation string.
    
    Args:
        key_factors: List of top features from extract_top_features
        judgment: The predicted judgment
        confidence: Confidence score (0-1)
        
    Returns:
        Human-readable explanation string
    """
    if not key_factors:
        return f"Predicted: {judgment}"
    
    conf_str = f" ({confidence:.0%} confidence)" if confidence else ""
    terms = [f['term'] for f in key_factors[:3]]
    terms_str = ", ".join(f"'{t}'" for t in terms)
    
    return f"Predicted {judgment}{conf_str} based on key terms: {terms_str}"


__all__ = ['extract_top_features', 'format_explanation']

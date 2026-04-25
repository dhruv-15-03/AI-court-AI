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


# Feature humanization and description generation
LEGAL_TERM_MAPPINGS = {
    # Criminal law
    'bail': 'Bail Status',
    'accused': 'Accused Profile',
    'evidence': 'Evidence Strength',
    'witness': 'Witness Testimony',
    'conviction': 'Prior Conviction',
    'acquittal': 'Acquittal History',
    'murder': 'Offense Type - Murder',
    'theft': 'Offense Type - Theft',
    'assault': 'Offense Type - Assault',
    'fraud': 'Offense Type - Fraud',
    'police': 'Police Investigation',
    'forensic': 'Forensic Evidence',
    'custody': 'Custody Status',
    'flight': 'Flight Risk',
    'fir': 'First Information Report',
    'chargesheet': 'Charge Sheet Filed',
    'prosecution': 'Prosecution Case',
    'confession': 'Confession / Admission',
    'abscond': 'Absconding Accused',
    'weapon': 'Weapon Involved',
    'premeditation': 'Premeditation',
    'motive': 'Motive Established',
    'dowry': 'Dowry Related',
    'cruelty': 'Cruelty Allegation',
    # Civil / compensation
    'compensation': 'Compensation Claim',
    'damages': 'Damages Assessment',
    'contract': 'Contract Dispute',
    'injunction': 'Injunction Sought',
    'specific_performance': 'Specific Performance',
    'decree': 'Decree / Order',
    'suit': 'Civil Suit',
    'plaintiff': 'Plaintiff Arguments',
    'defendant': 'Defendant Arguments',
    # Family law
    'maintenance': 'Maintenance Claim',
    'divorce': 'Divorce Proceedings',
    'alimony': 'Alimony Claim',
    'child': 'Child Welfare',
    'marriage': 'Marriage Duration',
    'matrimonial': 'Matrimonial Dispute',
    'guardianship': 'Guardianship',
    'adoption': 'Adoption Proceedings',
    'domestic_violence': 'Domestic Violence',
    # Property law
    'partition': 'Property Partition',
    'succession': 'Succession / Inheritance',
    'possession': 'Possession Dispute',
    'tenant': 'Tenancy Dispute',
    'eviction': 'Eviction Proceedings',
    'encroachment': 'Encroachment',
    'title': 'Title Dispute',
    'land': 'Land Dispute',
    'property': 'Property Matter',
    # Labor / employment
    'employment': 'Employment Status',
    'termination': 'Employment Termination',
    'retrenchment': 'Retrenchment',
    'reinstatement': 'Reinstatement Claim',
    'wages': 'Wage Dispute',
    'gratuity': 'Gratuity Claim',
    'provident': 'Provident Fund',
    'workman': 'Workman Status',
    'industrial': 'Industrial Dispute',
    'unfair_labor': 'Unfair Labor Practice',
    # Parties / procedure
    'appellant': 'Appellant Arguments',
    'respondent': 'Respondent Arguments',
    'petition': 'Petition Filed',
    'appeal': 'Appeal Filed',
    'revision': 'Revision Application',
    'writ': 'Writ Petition',
    'jurisdiction': 'Jurisdictional Issue',
    'limitation': 'Limitation Period',
    'contempt': 'Contempt Proceedings',
    'arbitration': 'Arbitration',
}

DIRECTION_TEMPLATES = {
    'positive': [
        "This factor supports the predicted outcome",
        "Presence of this element strengthens the case",
        "Courts typically view this favorably for the outcome",
    ],
    'negative': [
        "This factor weighs against the outcome",
        "This element may complicate the case",
        "Courts may consider this as a counter-factor",
    ],
}


def _humanize_feature(term: str) -> str:
    """Convert raw TF-IDF term to human-readable feature name."""
    term_lower = term.lower().strip()
    
    # Check direct mapping
    if term_lower in LEGAL_TERM_MAPPINGS:
        return LEGAL_TERM_MAPPINGS[term_lower]
    
    # Check partial matches
    for key, value in LEGAL_TERM_MAPPINGS.items():
        if key in term_lower:
            return value
    
    # Fallback: Title case the term
    return term.replace('_', ' ').title()


def _generate_feature_description(term: str, direction: str) -> str:
    """Generate a contextual description for a feature."""
    import random
    
    term_lower = term.lower()
    templates = DIRECTION_TEMPLATES.get(direction, DIRECTION_TEMPLATES['positive'])
    base_template = templates[hash(term) % len(templates)]  # Deterministic selection
    
    # Add term-specific context
    if 'bail' in term_lower:
        if direction == 'positive':
            return "Bail-related factors favor release"
        return "Bail-related concerns noted"
    elif 'evidence' in term_lower or 'forensic' in term_lower:
        if direction == 'positive':
            return "Evidence supports the predicted outcome"
        return "Evidence quality may be challenged"
    elif 'witness' in term_lower:
        if direction == 'positive':
            return "Witness testimony strengthens the case"
        return "Witness reliability may be questioned"
    elif any(x in term_lower for x in ['murder', 'assault', 'theft', 'fraud']):
        if direction == 'positive':
            return "Offense characteristics favor the outcome"
        return "Offense severity impacts the case"
    elif 'prior' in term_lower or 'conviction' in term_lower:
        if direction == 'positive':
            return "No prior record or clean history"
        return "Prior record may affect judgment"
    elif 'compensation' in term_lower or 'damages' in term_lower:
        if direction == 'positive':
            return "Compensation claim appears justified"
        return "Damages assessment may be contested"
    
    return base_template


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
                direction = 'positive' if combined_score > 0 else 'negative'
                feature_name = str(feature_names[idx])
                feature_scores.append({
                    'feature': _humanize_feature(feature_name),
                    'importance': round(abs(combined_score), 4),
                    'direction': direction,
                    'description': _generate_feature_description(feature_name, direction),
                    # Keep raw data for debugging/advanced use
                    '_raw': {
                        'term': feature_name,
                        'tfidf_weight': round(tfidf_weight, 4),
                        'feature_importance': round(importance, 4)
                    }
                })
        
        # Sort by absolute importance and take top-k
        def _importance_key(item: Dict[str, Any]) -> float:
            val = item.get('importance', 0)
            return abs(float(val)) if val is not None else 0.0
        feature_scores.sort(key=_importance_key, reverse=True)
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

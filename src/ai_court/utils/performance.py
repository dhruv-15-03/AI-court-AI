"""Performance utilities: caching, confidence language, and response formatting."""

import functools
import logging
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# LRU Caching for Text Preprocessing
# =============================================================================

def create_cached_preprocessor(
    preprocess_fn: Callable[[str], str],
    maxsize: int = 512
) -> Callable[[str], str]:
    """
    Wrap a text preprocessing function with LRU caching.
    
    Args:
        preprocess_fn: Original preprocessing function
        maxsize: Maximum cache size (default 512)
        
    Returns:
        Cached version of the preprocessing function
    """
    @functools.lru_cache(maxsize=maxsize)
    def cached_preprocess(text: str) -> str:
        return preprocess_fn(text)
    
    return cached_preprocess


def get_cache_stats(cached_fn: Callable[..., Any]) -> Dict[str, Any]:
    """Get cache statistics from a cached function."""
    try:
        info = cached_fn.cache_info()  # type: ignore[attr-defined]
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize or 0,
            "currsize": info.currsize
        }
    except AttributeError:
        return {"error": "Function is not cached"}


# =============================================================================
# Confidence Language Helper
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "very_high": 0.90,
    "high": 0.75,
    "moderate": 0.60,
    "low": 0.40,
    "very_low": 0.0
}

CONFIDENCE_LANGUAGE = {
    "very_high": {
        "level": "Very High",
        "description": "The model is highly confident in this prediction.",
        "recommendation": "This prediction is reliable for decision support.",
        "icon": "✅"
    },
    "high": {
        "level": "High", 
        "description": "The model is confident in this prediction.",
        "recommendation": "This prediction can be used with reasonable assurance.",
        "icon": "🟢"
    },
    "moderate": {
        "level": "Moderate",
        "description": "The model has moderate confidence in this prediction.",
        "recommendation": "Consider reviewing with additional context before acting.",
        "icon": "🟡"
    },
    "low": {
        "level": "Low",
        "description": "The model has low confidence in this prediction.",
        "recommendation": "Expert review is strongly recommended before using this prediction.",
        "icon": "🟠"
    },
    "very_low": {
        "level": "Very Low",
        "description": "The model is uncertain about this prediction.",
        "recommendation": "This prediction should not be relied upon without expert verification.",
        "icon": "🔴"
    }
}


def get_confidence_level(confidence: Optional[float]) -> str:
    """Map confidence score to categorical level."""
    if confidence is None:
        return "unknown"
    
    for level, threshold in CONFIDENCE_THRESHOLDS.items():
        if confidence >= threshold:
            return level
    return "very_low"


def get_confidence_language(confidence: Optional[float]) -> Dict[str, Any]:
    """
    Generate human-friendly confidence information.
    
    Args:
        confidence: Model confidence score (0-1)
        
    Returns:
        Dictionary with level, description, recommendation, and icon
    """
    level = get_confidence_level(confidence)
    
    if level == "unknown":
        return {
            "level": "Unknown",
            "description": "Confidence score not available.",
            "recommendation": "Unable to assess prediction reliability.",
            "icon": "❓",
            "score": None,
            "percentile": None
        }
    
    base_info = CONFIDENCE_LANGUAGE.get(level, CONFIDENCE_LANGUAGE["very_low"])
    return {
        "level": base_info["level"],
        "description": base_info["description"],
        "recommendation": base_info["recommendation"],
        "icon": base_info["icon"],
        "score": round(confidence or 0, 3),
        "percentile": f"{int((confidence or 0) * 100)}%"
    }


# =============================================================================
# Response Format Helpers
# =============================================================================

def format_minimal_response(
    judgment: str,
    confidence: Optional[float],
    case_type: str
) -> Dict[str, Any]:
    """Generate minimal response format for fast API responses."""
    return {
        "judgment": judgment,
        "confidence": round(confidence or 0, 3),
        "case_type": case_type
    }


def format_full_response(
    judgment: str,
    confidence: Optional[float],
    case_type: str,
    key_factors: Optional[List[Dict[str, Any]]] = None,
    explanation: Optional[str] = None,
    needs_review: bool = False,
    abstention_reason: Optional[str] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate full response format with all details."""
    response: Dict[str, Any] = {
        "judgment": judgment,
        "confidence": round(confidence or 0, 3),
        "case_type": case_type,
        "confidence_info": get_confidence_language(confidence),
        "needs_review": needs_review,
    }
    
    if key_factors:
        response["key_factors"] = key_factors
    if explanation:
        response["explanation"] = explanation
    if abstention_reason:
        response["abstention_reason"] = abstention_reason
    if similar_cases:
        response["similar_cases"] = similar_cases
        
    return response


def format_detailed_response(
    judgment: str,
    confidence: Optional[float],
    case_type: str,
    answers: Dict[str, Any],
    key_factors: Optional[List[Dict[str, Any]]] = None,
    explanation: Optional[str] = None,
    needs_review: bool = False,
    abstention_reason: Optional[str] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    shadow_result: Optional[Dict[str, Any]] = None,
    agreement_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Generate detailed response format with full audit trail."""
    response = format_full_response(
        judgment=judgment,
        confidence=confidence,
        case_type=case_type,
        key_factors=key_factors,
        explanation=explanation,
        needs_review=needs_review,
        abstention_reason=abstention_reason,
        similar_cases=similar_cases
    )
    
    response["answers"] = answers
    response["audit"] = {
        "model_source": "classical" if shadow_result is None else "multi_axis",
        "shadow_multi_axis": shadow_result,
        "agreement_rate": round(agreement_rate, 3) if agreement_rate else None,
    }
    
    return response


# =============================================================================
# Outcome Description Helper
# =============================================================================

OUTCOME_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "Bail Granted": {
        "meaning": "The court has granted bail to the accused.",
        "implications": "The accused can be released from custody pending trial, usually with conditions.",
        "next_steps": "Comply with bail conditions, attend all court dates."
    },
    "Bail Denied": {
        "meaning": "The court has denied bail to the accused.",
        "implications": "The accused must remain in custody pending trial.",
        "next_steps": "Consider appealing the decision or reapplying with stronger grounds."
    },
    "Conviction": {
        "meaning": "The accused has been found guilty of the charges.",
        "implications": "Sentencing will follow; possible imprisonment, fines, or other penalties.",
        "next_steps": "Consider appeal options, prepare for sentencing hearing."
    },
    "Acquittal": {
        "meaning": "The accused has been found not guilty.",
        "implications": "The accused is free from the charges; cannot be tried again for the same offense.",
        "next_steps": "No further criminal action on these charges."
    },
    "Charges Quashed": {
        "meaning": "The charges have been dismissed or set aside by the court.",
        "implications": "The prosecution cannot proceed on these specific charges.",
        "next_steps": "Monitor for any new charges; prosecution may appeal."
    },
    "Compensation Awarded": {
        "meaning": "The court has ordered monetary compensation.",
        "implications": "The losing party must pay the specified amount.",
        "next_steps": "Ensure compliance with payment timelines; consider enforcement if needed."
    },
    "Petition Dismissed": {
        "meaning": "The court has dismissed the petition/application.",
        "implications": "The relief sought has been denied.",
        "next_steps": "Consider appeal options or alternative legal remedies."
    },
    "Injunction Granted": {
        "meaning": "The court has issued an injunction/restraining order.",
        "implications": "The restrained party must comply with the court's directions.",
        "next_steps": "Monitor compliance; seek contempt proceedings if violated."
    },
    "Maintenance Ordered": {
        "meaning": "The court has ordered payment of maintenance/alimony.",
        "implications": "Regular payments must be made as per court order.",
        "next_steps": "Set up payment mechanism; modify if circumstances change significantly."
    },
    "Other": {
        "meaning": "The outcome does not fit standard categories.",
        "implications": "Review the specific judgment for details.",
        "next_steps": "Consult with a legal professional for specific guidance."
    }
}


def get_outcome_description(judgment: str) -> Dict[str, str]:
    """Get detailed description of an outcome."""
    return OUTCOME_DESCRIPTIONS.get(judgment, OUTCOME_DESCRIPTIONS["Other"])

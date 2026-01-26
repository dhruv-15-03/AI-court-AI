"""Utility modules for AI Court."""

from ai_court.utils.explainability import (
    extract_top_features,
    format_explanation,
)

from ai_court.utils.performance import (
    create_cached_preprocessor,
    get_cache_stats,
    get_confidence_language,
    get_confidence_level,
    get_outcome_description,
    format_minimal_response,
    format_full_response,
    format_detailed_response,
    CONFIDENCE_THRESHOLDS,
    CONFIDENCE_LANGUAGE,
    OUTCOME_DESCRIPTIONS,
)

__all__ = [
    # Explainability
    "extract_top_features",
    "format_explanation",
    # Performance
    "create_cached_preprocessor",
    "get_cache_stats",
    "get_confidence_language",
    "get_confidence_level",
    "get_outcome_description",
    "format_minimal_response",
    "format_full_response",
    "format_detailed_response",
    "CONFIDENCE_THRESHOLDS",
    "CONFIDENCE_LANGUAGE",
    "OUTCOME_DESCRIPTIONS",
]

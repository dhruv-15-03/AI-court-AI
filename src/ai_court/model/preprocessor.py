"""
Text preprocessing utilities for legal case analysis.

This module provides text normalization, lemmatization, and outcome classification
functionality for the AI Court legal case prediction system.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, FrozenSet, List, Optional

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from ai_court.utils.nltk_setup import ensure_nltk_resources

__all__ = ["TextPreprocessor"]

logger = logging.getLogger(__name__)

# Thread-safe cache for outcome rules
_OUTCOME_RULES: Optional[List[dict[str, Any]]] = None
_OUTCOME_RULES_LOCK = threading.Lock()

# Fallback rules used when config file cannot be loaded
_FALLBACK_OUTCOME_RULES: List[dict[str, Any]] = [
    {"label": "Acquittal/Conviction Overturned", "phrases": ["acquitted", "acquittal", "conviction overturned"]},
    {"label": "Conviction Upheld/Appeal Dismissed", "phrases": ["appeal dismissed", "conviction upheld"]},
    {"label": "Bail Granted", "phrases": ["bail granted", "anticipatory bail granted"]},
    {"label": "Bail Denied", "phrases": ["bail denied", "bail rejected"]},
    {"label": "Charges/Proceedings Quashed", "phrases": ["quashed", "quash", "fir quashed"]},
    {"label": "Sentence Reduced/Modified", "phrases": ["sentence reduced", "sentence modified"]},
    {"label": "Case Remanded/Sent Back", "phrases": ["remanded", "sent back"]},
]


def _load_outcome_rules() -> None:
    """
    Load outcome classification rules from configuration file.
    
    This function is thread-safe and loads rules only once. If the config file
    is not found or fails to load, fallback rules are used instead.
    """
    global _OUTCOME_RULES
    
    # Fast path: already loaded
    if _OUTCOME_RULES is not None:
        return
    
    with _OUTCOME_RULES_LOCK:
        # Double-check after acquiring lock
        if _OUTCOME_RULES is not None:
            return
        
        config_path: Optional[str] = None
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(base_dir, '..', 'config', 'outcome_rules.json'))
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_rules = json.load(f)
                
                # Validate the loaded structure
                if not isinstance(loaded_rules, list):
                    raise ValueError("Outcome rules must be a list")
                
                for idx, rule in enumerate(loaded_rules):
                    if not isinstance(rule, dict):
                        raise ValueError(f"Rule at index {idx} must be a dictionary")
                    if 'label' not in rule or 'phrases' not in rule:
                        raise ValueError(f"Rule at index {idx} missing 'label' or 'phrases' key")
                    if not isinstance(rule.get('phrases'), list):
                        raise ValueError(f"Rule at index {idx} 'phrases' must be a list")
                
                _OUTCOME_RULES = loaded_rules
                logger.info("Loaded %d outcome rules from %s", len(_OUTCOME_RULES), config_path)
            else:
                logger.warning(
                    "Outcome rules config not found at %s. Using fallback rules.", 
                    config_path
                )
                _OUTCOME_RULES = _FALLBACK_OUTCOME_RULES.copy()
                
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in outcome rules config %s: %s", config_path, e)
            _OUTCOME_RULES = _FALLBACK_OUTCOME_RULES.copy()
        except (OSError, IOError) as e:
            logger.error("Failed to read outcome rules config %s: %s", config_path, e)
            _OUTCOME_RULES = _FALLBACK_OUTCOME_RULES.copy()
        except ValueError as e:
            logger.error("Invalid outcome rules structure in %s: %s", config_path, e)
            _OUTCOME_RULES = _FALLBACK_OUTCOME_RULES.copy()

# Ensure resources are available when this module is imported
ensure_nltk_resources()


class TextPreprocessor:
    """
    Text preprocessing for legal case documents.
    
    This class provides text normalization, tokenization, and lemmatization
    specifically tuned for legal domain text. It preserves important legal
    terminology while removing noise and standardizing the text format.
    
    Attributes:
        stop_words: Set of English stopwords to filter out.
        lemmatizer: WordNet-based lemmatizer for word normalization.
        legal_terms: Frozenset of domain-specific terms to preserve.
        
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.preprocess("The defendant was acquitted.")
        'defendant acquit'
    """
    
    # Class-level constants for legal vocabulary (immutable)
    LEGAL_TERMS: FrozenSet[str] = frozenset({
        'plaintiff', 'defendant', 'appeal', 'appeals', 'judgment', 'judgement', 
        'court', 'section', 'act', 'article', 'respondent', 'appellant', 
        'petitioner', 'accused', 'evidence', 'conviction', 'acquittal', 
        'forensic', 'witness', 'testimony', 'murder', 'rape', 'ipc', 'crpc', 
        'dismissed', 'upheld', 'affirmed', 'ballistic', 'circumstantial', 
        'alibi', 'bail', 'trial', 'sentence', 'sentencing', 'compromise', 
        'settlement', 'damages', 'compensation', 'negligence', 'property',
        'habeas', 'corpus', 'mandamus', 'certiorari', 'injunction', 'quash',
        'remand', 'custody', 'parole', 'probation', 'affidavit', 'summons'
    })
    
    # Compiled regex patterns for performance
    _PATTERN_NON_ALPHA = re.compile(r"[^a-z0-9\s]")
    _PATTERN_WHITESPACE = re.compile(r"\s+")

    def __init__(self) -> None:
        """Initialize the preprocessor with NLTK resources."""
        self.stop_words: FrozenSet[str] = frozenset(stopwords.words('english'))
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        # Instance attribute for backward compatibility
        self.legal_terms: FrozenSet[str] = self.LEGAL_TERMS

    def preprocess(self, text: str) -> str:
        """
        Normalize and lemmatize text while preserving key legal terms.
        
        Performs the following transformations:
        1. Converts text to lowercase
        2. Removes non-alphanumeric characters (except spaces)
        3. Tokenizes the text
        4. Filters stopwords (preserving legal terminology)
        5. Lemmatizes tokens (verb form, then noun form)
        
        Args:
            text: The input text to preprocess. Non-string inputs return empty string.
            
        Returns:
            Preprocessed text with tokens joined by single spaces.
            
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess("The DEFENDANT was convicted of murder.")
            'defendant convict murder'
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Basic cleanup using compiled patterns
        text = text.lower()
        text = self._PATTERN_NON_ALPHA.sub(" ", text)
        text = self._PATTERN_WHITESPACE.sub(" ", text).strip()
        
        if not text:
            return ""

        # Tokenize
        tokens: List[str] = word_tokenize(text)
        
        # Keep tokens if either not a stopword or part of legal vocabulary
        filtered: List[str] = [
            t for t in tokens 
            if t not in self.stop_words or t in self.legal_terms
        ]
        
        if not filtered:
            return ""
        
        # Lemmatize (verbs then nouns for better normalization)
        lemmas: List[str] = [
            self.lemmatizer.lemmatize(
                self.lemmatizer.lemmatize(t, pos='v'), 
                pos='n'
            ) 
            for t in filtered
        ]
        
        return " ".join(lemmas)

    @staticmethod
    def normalize_outcome(text: Optional[str]) -> str:
        """
        Map raw judgment text to a standardized outcome class.
        
        Uses a configuration-driven rule system to classify legal case outcomes
        into a manageable set of categories. Rules are loaded from
        `config/outcome_rules.json` and cached for performance.
        
        Args:
            text: Raw judgment or outcome text from case data.
                  None or empty values are handled gracefully.
                  
        Returns:
            A standardized outcome label string. Returns "Other" if no
            matching rule is found or if input is empty/invalid.
            
        Example:
            >>> TextPreprocessor.normalize_outcome("Appeal dismissed on merits")
            'Conviction Upheld/Appeal Dismissed'
            >>> TextPreprocessor.normalize_outcome("bail granted")
            'Bail Granted'
        """
        _load_outcome_rules()
        
        # Handle None/empty input
        if not text:
            return "Other"
            
        normalized_text: str = text.lower().strip()
        
        if not normalized_text:
            return "Other"
        
        # _OUTCOME_RULES is guaranteed to be non-None after _load_outcome_rules()
        assert _OUTCOME_RULES is not None
        
        for rule in _OUTCOME_RULES:
            phrases = rule.get('phrases', [])
            if any(phrase in normalized_text for phrase in phrases):
                return rule['label']
                
        return "Other"

import re
import os
import json
import logging
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ai_court.utils.nltk_setup import ensure_nltk_resources

logger = logging.getLogger(__name__)

# Cache for outcome rules
_OUTCOME_RULES = None

def _load_outcome_rules():
    global _OUTCOME_RULES
    if _OUTCOME_RULES is not None:
        return
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, '..', 'config', 'outcome_rules.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                _OUTCOME_RULES = json.load(f)
            logger.info(f"Loaded {len(_OUTCOME_RULES)} outcome rules from {config_path}")
        else:
            logger.warning(f"Outcome rules config not found at {config_path}. Using empty fallback rules.")
            _OUTCOME_RULES = []
            # Log the expected path for debugging
            logger.debug(f"Expected config at: {os.path.abspath(config_path)}")
    except Exception as e:
        logger.error(f"Failed to load outcome rules: {e}")
        _OUTCOME_RULES = []

# Ensure resources are available when this module is imported
ensure_nltk_resources()

class TextPreprocessor:
    def __init__(self):
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

    def preprocess(self, text: str) -> str:
        """Normalize and lemmatize text while keeping key legal terms."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleanup
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # Keep ASCII letters, digits and spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)
        
        # Keep tokens if either not a stopword or part of a curated legal vocabulary
        filtered = [t for t in tokens if (t not in self.stop_words) or (t in self.legal_terms)]
        
        # Lemmatize (verbs then nouns)
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in filtered]
        lemmas = [self.lemmatizer.lemmatize(t, pos='n') for t in lemmas]
        
        return " ".join(lemmas)

    @staticmethod
    def normalize_outcome(text: str) -> str:
        """Map raw judgment text to a manageable set of outcome classes using configured rules."""
        _load_outcome_rules()
        
        t = (text or "").lower()
        
        if not _OUTCOME_RULES:
            # Fallback for safety if config failed to load
            return "Other"
            
        for rule in _OUTCOME_RULES:
            if any(phrase in t for phrase in rule.get('phrases', [])):
                return rule['label']
                
        return "Other"

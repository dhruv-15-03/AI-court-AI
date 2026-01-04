from typing import Any, Dict, List, Optional, Callable
import threading

# Global Model State
classifier: Any = None  # Instance of LegalCaseClassifier
classifier_model: Any = None # Deprecated: Use classifier.model
label_encoder: Any = None # Deprecated: Use classifier.label_encoder
preprocess_fn: Callable[[str], str] = lambda x: x # Deprecated: Use classifier.preprocess_text

# Multi-axis Model State
multi_axis_bundle: Optional[Dict[str, Any]] = None
multi_axis_lock = threading.Lock()

# Search Indices
search_index: Optional[Dict[str, Any]] = None
semantic_index: Optional[Dict[str, Any]] = None

# Active Learning Queue
AL_QUEUE: List[Dict[str, Any]] = []

# Agreement Stats
agreement_stats = {
    'total_compared': 0,
    'agreements': 0,
    'last_samples': []  # last discrepancy samples
}

# Metrics
REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None
PREDICTIONS_TOTAL: Any = None

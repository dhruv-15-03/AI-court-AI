from typing import Any, Dict, List, Optional
import threading
import time

# Global Model State
classifier: Any = None  # Instance of LegalCaseClassifier
classifier_model: Any = None # Deprecated: Use classifier.model
label_encoder: Any = None # Deprecated: Use classifier.label_encoder
def preprocess_fn(x: str) -> str: # Deprecated: Use classifier.preprocess_text
    return x

# Multi-axis Model State
multi_axis_bundle: Optional[Dict[str, Any]] = None
multi_axis_lock = threading.Lock()

# Search Indices
search_index: Optional[Dict[str, Any]] = None
semantic_index: Optional[Dict[str, Any]] = None

# Active Learning Queue
AL_QUEUE: List[Dict[str, Any]] = []
AL_QUEUE_MAX_SIZE = 1000  # Prevent unbounded growth

# Agreement Stats
agreement_stats = {
    'total_compared': 0,
    'agreements': 0,
    'last_samples': []  # last discrepancy samples
}

# Prediction Stats (for monitoring)
prediction_stats = {
    'total_predictions': 0,
    'low_confidence_count': 0,
    'abstentions': 0,
    'last_prediction_time': None,
    'avg_confidence': 0.0,
    '_confidence_sum': 0.0,
}

# Metrics
REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None
PREDICTIONS_TOTAL: Any = None


def update_prediction_stats(confidence: float, abstained: bool = False):
    """Update prediction statistics for monitoring."""
    prediction_stats['total_predictions'] += 1
    prediction_stats['last_prediction_time'] = time.time()
    
    if confidence is not None:
        prediction_stats['_confidence_sum'] += confidence
        prediction_stats['avg_confidence'] = (
            prediction_stats['_confidence_sum'] / prediction_stats['total_predictions']
        )
        if confidence < 0.5:  # Hardcoded threshold for stats
            prediction_stats['low_confidence_count'] += 1
    
    if abstained:
        prediction_stats['abstentions'] += 1


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': round(mem_info.rss / (1024 * 1024), 2),
            'vms_mb': round(mem_info.vms / (1024 * 1024), 2),
            'percent': round(process.memory_percent(), 2),
        }
    except ImportError:
        # psutil not available, use basic approach
        import resource
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {
                'max_rss_mb': round(usage.ru_maxrss / 1024, 2),  # Linux reports in KB
            }
        except Exception:
            return {'error': 'memory stats unavailable'}
    except Exception as e:
        return {'error': str(e)}

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '1048576'))  # 1 MB default
GOVERNANCE_STATUS_PATH = os.getenv('GOVERNANCE_STATUS_PATH', 'governance_status.json')
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
SENTRY_PROFILES_SAMPLE_RATE = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
APP_ENV = os.getenv("APP_ENV", "production")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/legal_case_classifier.pkl")
SEARCH_INDEX_PATH = os.getenv("SEARCH_INDEX_PATH", "models/search_index.pkl")

# Confidence & Abstention
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
AUTO_QUEUE_LOW_CONFIDENCE = os.getenv("AUTO_QUEUE_LOW_CONFIDENCE", "1") == "1"

# Explainability
EXPLAIN_TOP_K = int(os.getenv("EXPLAIN_TOP_K", "5"))

# Memory Optimization
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
LAZY_LOAD_SEARCH = os.getenv("LAZY_LOAD_SEARCH", "0") == "1"

# Local Summarization (no HF API)
USE_LOCAL_SUMMARY = os.getenv("USE_LOCAL_SUMMARY", "1") == "1"

# Performance Optimizations
PREPROCESSING_CACHE_SIZE = int(os.getenv("PREPROCESSING_CACHE_SIZE", "512"))
ENABLE_PREPROCESSING_CACHE = os.getenv("ENABLE_PREPROCESSING_CACHE", "1") == "1"
BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "50"))

# Customization Options
INCLUDE_KEY_FACTORS = os.getenv("INCLUDE_KEY_FACTORS", "1") == "1"
INCLUDE_EXPLANATION = os.getenv("INCLUDE_EXPLANATION", "1") == "1"
INCLUDE_CONFIDENCE_LANGUAGE = os.getenv("INCLUDE_CONFIDENCE_LANGUAGE", "1") == "1"
INCLUDE_SIMILAR_CASES = os.getenv("INCLUDE_SIMILAR_CASES", "1") == "1"
SIMILAR_CASES_COUNT = int(os.getenv("SIMILAR_CASES_COUNT", "3"))

# Response Format Customization
RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "full")  # full, minimal, detailed

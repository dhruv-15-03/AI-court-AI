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

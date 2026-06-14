import os
import sys

# The server validates config at import time and refuses to boot in production
# without API_KEY (Phase 0 fail-fast hardening). Run the test suite in a
# non-production environment so importing ai_court.api.server succeeds and the
# API-key guard stays disabled (require_api_key is a no-op when API_KEY is empty),
# matching how the suite exercises endpoints without auth headers.
os.environ.setdefault("APP_ENV", "testing")

# Ensure the `src/` directory is on sys.path so we can import `ai_court` package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

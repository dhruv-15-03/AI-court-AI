import os
import sys
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from flask import Flask, request, g
from flask_cors import CORS
from flasgger import Swagger
from prometheus_client import Counter, Histogram
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from ai_court.api import config, state, dependencies
from ai_court.api.routes import analysis_bp, search_bp, monitoring_bp, feedback_bp
from ai_court.api.extensions import limiter

# Ensure project root is on path (legacy support)
if config.PROJECT_ROOT not in sys.path:
    sys.path.append(config.PROJECT_ROOT)
if config.SRC_DIR not in sys.path:
    sys.path.append(config.SRC_DIR)

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("api")

# Metrics
try:
    state.REQUEST_COUNT = Counter('ai_court_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
    state.REQUEST_LATENCY = Histogram('ai_court_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
    state.PREDICTIONS_TOTAL = Counter('ai_court_predictions_total', 'Total predictions served', ['endpoint', 'label'])
except ValueError:
    # Metrics might be already defined if reloaded
    pass

# Sentry
if config.SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            integrations=[FlaskIntegration()],
            traces_sample_rate=config.SENTRY_TRACES_SAMPLE_RATE,
            profiles_sample_rate=config.SENTRY_PROFILES_SAMPLE_RATE,
            environment=config.APP_ENV,
            release=config.APP_VERSION,
        )
        logger.info("Sentry initialized")
    except Exception as _e:
        logger.error(f"Sentry init failed: {_e}")

app = Flask(__name__)
CORS(app)
Swagger(app)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Initialize extensions
limiter.init_app(app)

# Register Blueprints
app.register_blueprint(analysis_bp)
app.register_blueprint(search_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(feedback_bp)

# Middleware
@app.before_request
def _before_request():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.started_at = time.time()
    g.endpoint_for_metrics = request.endpoint or request.path

@app.after_request
def _after_request(response):
    try:
        duration = (time.time() - getattr(g, 'started_at', time.time()))
        duration_ms = int(duration * 1000)
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "lvl": "info",
            "request_id": getattr(g, 'request_id', None),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "remote_addr": request.headers.get('X-Forwarded-For', request.remote_addr),
            "ua": request.headers.get('User-Agent')
        }
        logger.info(json.dumps(log_obj, ensure_ascii=False))
    except Exception:
        pass
    try:
        ep = getattr(g, 'endpoint_for_metrics', request.path)
        if state.REQUEST_COUNT:
            state.REQUEST_COUNT.labels(request.method, ep, response.status_code).inc()
        if state.REQUEST_LATENCY:
            state.REQUEST_LATENCY.labels(ep).observe(duration)
    except Exception:
        pass
    if getattr(g, 'request_id', None):
        response.headers["X-Request-ID"] = g.request_id
    return response

# Load models on startup
dependencies.load_model()
dependencies.load_multi_axis()
dependencies.load_search_index()
dependencies.load_semantic_index()

if __name__ == "__main__":
    app.run(debug=True, port=5002, host="0.0.0.0")

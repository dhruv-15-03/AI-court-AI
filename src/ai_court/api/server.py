import signal
import sys
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from flask import Flask, request, g, jsonify
from flask_cors import CORS
from flasgger import Swagger
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from ai_court.api import config, state, dependencies
from ai_court.api.routes import analysis_bp, search_bp, monitoring_bp, feedback_bp, agent_bp, audit_bp
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
    _model_inference_latency = Histogram('ai_court_model_inference_seconds', 'Model inference latency')
    _memory_rss_gauge = Gauge('ai_court_memory_rss_mb', 'Process RSS memory in MB')
    _al_queue_gauge = Gauge('ai_court_al_queue_size', 'Active learning queue size')
    _errors_total = Counter('ai_court_errors_total', 'Total error responses', ['endpoint', 'status'])
except ValueError:
    # Metrics might be already defined if reloaded
    _model_inference_latency = None  # type: ignore[assignment]
    _memory_rss_gauge = None  # type: ignore[assignment]
    _al_queue_gauge = None  # type: ignore[assignment]
    _errors_total = None  # type: ignore[assignment]

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

# ── Startup config validation ──────────────────────────────────────────
def _validate_config() -> None:
    """Fail fast if critical production config is missing."""
    if config.APP_ENV == "production" and not config.API_KEY:
        logger.warning(
            "API_KEY is empty in production mode. "
            "Set API_KEY env var to enable authentication."
        )
    if config.CORS_ORIGINS == ["*"] and config.APP_ENV == "production":
        logger.warning(
            "CORS_ORIGINS=* in production — any origin can call this API. "
            "Set CORS_ORIGINS to your frontend domain(s)."
        )

_validate_config()

# ── CORS ───────────────────────────────────────────────────────────────
CORS(
    app,
    origins=config.CORS_ORIGINS,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key"],
    supports_credentials=False,
)
Swagger(app)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Initialize extensions
limiter.init_app(app)

# Register Blueprints
app.register_blueprint(analysis_bp)
app.register_blueprint(search_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(agent_bp)
app.register_blueprint(audit_bp)


# ── Global error handlers ─────────────────────────────────────────────
@app.errorhandler(400)
def _handle_bad_request(e):
    return jsonify({"error": "bad_request", "message": str(e)}), 400

@app.errorhandler(404)
def _handle_not_found(e):
    return jsonify({"error": "not_found"}), 404

@app.errorhandler(405)
def _handle_method_not_allowed(e):
    return jsonify({"error": "method_not_allowed"}), 405

@app.errorhandler(413)
def _handle_payload_too_large(e):
    return jsonify({"error": "payload_too_large", "max_bytes": config.MAX_CONTENT_LENGTH}), 413

@app.errorhandler(429)
def _handle_rate_limited(e):
    return jsonify({"error": "rate_limited", "message": "Too many requests. Please try again later."}), 429

@app.errorhandler(500)
def _handle_internal_error(e):
    logger.error("Unhandled server error: %s", e, exc_info=True)
    return jsonify({
        "error": "internal_server_error",
        "request_id": getattr(g, 'request_id', None),
        "message": "An internal error occurred. Please try again or contact support.",
    }), 500


# ── Middleware ─────────────────────────────────────────────────────────
@app.before_request
def _before_request():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.started_at = time.time()
    g.endpoint_for_metrics = request.endpoint or request.path

@app.after_request
def _after_request(response):
    duration = time.time() - getattr(g, 'started_at', time.time())
    duration_ms = int(duration * 1000)

    # Structured JSON request log
    try:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "lvl": "info",
            "request_id": getattr(g, 'request_id', None),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "remote_addr": request.headers.get('X-Forwarded-For', request.remote_addr),
            "ua": request.headers.get('User-Agent'),
        }
        logger.info(json.dumps(log_obj, ensure_ascii=False))
    except Exception:
        pass

    # Prometheus metrics
    try:
        ep = getattr(g, 'endpoint_for_metrics', request.path)
        if state.REQUEST_COUNT:
            state.REQUEST_COUNT.labels(request.method, ep, response.status_code).inc()
        if state.REQUEST_LATENCY:
            state.REQUEST_LATENCY.labels(ep).observe(duration)
        if response.status_code >= 400 and _errors_total:
            _errors_total.labels(ep, response.status_code).inc()
    except Exception:
        pass

    # Update gauges (cheap, once per request is fine)
    try:
        if _al_queue_gauge:
            _al_queue_gauge.set(len(state.AL_QUEUE))
        if _memory_rss_gauge:
            mem = state.get_memory_usage()
            rss = mem.get('rss_mb')
            if rss:
                _memory_rss_gauge.set(rss)
    except Exception:
        pass

    # Propagate request-id + security headers
    if getattr(g, 'request_id', None):
        response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


# ── Graceful shutdown ──────────────────────────────────────────────────
def _graceful_shutdown(signum, _frame):
    """Flush AL queue and exit cleanly on SIGTERM / SIGINT."""
    logger.info("Received signal %s — shutting down gracefully…", signum)
    try:
        from ai_court.api.routes.feedback import save_queue
        save_queue()
        logger.info("Active-learning queue flushed.")
    except Exception as exc:
        logger.warning("AL queue flush failed: %s", exc)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)

# ── Load models on startup ────────────────────────────────────────────
dependencies.load_model()
dependencies.load_multi_axis()
dependencies.load_search_index()
dependencies.load_semantic_index()
dependencies.load_agent()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8000"))
    if config.APP_ENV == "production":
        raise RuntimeError(
            "Do not run server.py directly in production.  "
            "Use: gunicorn -c gunicorn.conf.py src.ai_court.api.server:app"
        )
    app.run(debug=False, port=port, host="0.0.0.0")

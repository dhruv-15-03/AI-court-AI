import os
import sys
import time
import uuid
import logging
import re
from datetime import datetime, timezone
import platform
import json
import threading
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flasgger import Swagger, swag_from
from werkzeug.exceptions import BadRequest
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import dill
import numpy as np
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
try:  # Ontology helpers (best-effort)
    from ai_court.ontology import (
        ontology_metadata,  # type: ignore
        load_ontology,      # type: ignore
        flatten_leaves,     # type: ignore
        map_coarse_label,   # type: ignore
    )
except Exception:  # pragma: no cover - provide safe fallbacks if ontology optional deps missing
    def ontology_metadata():  # type: ignore
        return {}
    def load_ontology():  # type: ignore
        return {}
    def flatten_leaves(_data):  # type: ignore
        return []
    def map_coarse_label(label: str):  # type: ignore
        return label, False

# Ensure project root is on path for local runs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# Also ensure the 'src' directory is importable for ai_court package references during dill.load
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

load_dotenv()

# Initialize Flask app (was missing after refactor)
app = Flask(__name__)
CORS(app)
Swagger(app)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', '1048576'))  # 1 MB default

@app.route('/api/reload_multi_axis', methods=['POST'])
def reload_multi_axis():
    """Reload the promoted multi-axis model bundle without restarting the server."""
    global multi_axis_bundle
    from flask import current_app
    try:
        multi_axis_bundle = _load_multi_axis()
        if multi_axis_bundle:
            return jsonify({'status':'ok','run_id':multi_axis_bundle.get('run_id')}), 200
        return jsonify({'status':'error','detail':'No promoted model found'}), 404
    except Exception as e:
        return jsonify({'status':'error','detail':str(e)}), 500

@app.route('/api/governance/status', methods=['GET'])
def governance_status():
    """Return consolidated governance status if generated (best-effort)."""
    path = os.getenv('GOVERNANCE_STATUS_PATH','governance_status.json')
    if not os.path.exists(path):
        return jsonify({'error':'status_unavailable'}), 404
    try:
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error':'load_failed','detail':str(e)}), 500

@app.route('/api/governance/refresh', methods=['POST'])
def governance_refresh():
    """Force a refresh of governance_status.json by executing the script inline."""
    try:
        import subprocess, sys
        cmd = [sys.executable, 'scripts/governance_status.py']
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            return jsonify({'status':'error','detail': rc.stderr.strip()}), 500
        return jsonify({'status':'ok'}), 200
    except Exception as e:
        return jsonify({'status':'error','detail':str(e)}), 500

# Basic API key auth via header X-API-Key
API_KEY = os.getenv("API_KEY", "")


class AnalyzeRequest(BaseModel):
    case_type: Optional[str] = Field(default="Unknown", max_length=100)
    parties: Optional[str] = Field(default=None, max_length=2000)
    violence_level: Optional[str] = Field(default=None, max_length=100)
    weapon: Optional[str] = Field(default=None, max_length=50)
    police_report: Optional[str] = Field(default=None, max_length=50)
    witnesses: Optional[str] = Field(default=None, max_length=50)
    premeditation: Optional[str] = Field(default=None, max_length=50)
    employment_duration: Optional[str] = Field(default=None, max_length=100)
    children: Optional[str] = Field(default=None, max_length=50)
    marriage_duration: Optional[str] = Field(default=None, max_length=100)
    dispute_type: Optional[str] = Field(default=None, max_length=100)
    document_evidence: Optional[str] = Field(default=None, max_length=50)
    monetary_value: Optional[str] = Field(default=None, max_length=100)
    prior_relationship: Optional[str] = Field(default=None, max_length=200)
    attempts_resolution: Optional[str] = Field(default=None, max_length=50)

    def combined_text(self) -> str:
        parts: List[str] = []
        data = self.model_dump()
        ct = data.get("case_type", "") or ""
        for k, v in data.items():
            if k != "case_type" and v:
                parts.append(f"{k}: {v}")
        # Keep similar structure to previous implementation
        return f"{ct} " + ". ".join(parts)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    k: int = Field(default=5, ge=1, le=20)


class DriftCompareRequest(BaseModel):
    counts: Dict[str, int]
    minimum_total: int = Field(default=1, ge=1)

    def normalized(self, classes: List[str]) -> List[float]:
        total = sum(v for v in self.counts.values() if isinstance(v, (int, float)))
        if total <= 0:
            return [0.0 for _ in classes]
        return [float(self.counts.get(c, 0)) / total for c in classes]

def require_api_key():
    if API_KEY:
        key = request.headers.get("X-API-Key", "")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
    return None

# Simple rate limiting (configurable storage)
RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],  # tune as needed
    storage_uri=RATE_LIMIT_STORAGE_URI,
)

# Logging: structured JSON with request ID and latency
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("api")

# Prometheus metrics (idempotent across repeated test imports)
try:
    REQUEST_COUNT  # type: ignore[name-defined]
except Exception:  # metrics not yet defined
    try:
        REQUEST_COUNT = Counter('ai_court_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        REQUEST_LATENCY = Histogram('ai_court_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
        PREDICTIONS_TOTAL = Counter('ai_court_predictions_total', 'Total predictions served', ['endpoint', 'label'])
    except ValueError:
        # Already registered in global REGISTRY (pytest re-import); ignore
        class _Dummy:
            def labels(self, *_, **__):
                return self
            def inc(self, *_args, **_kw):
                return None
            def observe(self, *_args, **_kw):
                return None
        REQUEST_COUNT = REQUEST_LATENCY = PREDICTIONS_TOTAL = _Dummy()  # type: ignore

# Sentry (optional)
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
SENTRY_PROFILES_SAMPLE_RATE = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
if SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FlaskIntegration()],
            traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
            profiles_sample_rate=SENTRY_PROFILES_SAMPLE_RATE,
            environment=os.getenv("APP_ENV", "production"),
            release=os.getenv("APP_VERSION", "0.1.0"),
        )
        logger.info("Sentry initialized")
    except Exception as _e:
        logger.error(f"Sentry init failed: {_e}")


@app.route("/version", methods=["GET"])
def version():
    app_version = os.getenv("APP_VERSION", "0.1.0")
    app_env = os.getenv("APP_ENV", "production")
    commit = os.getenv("GIT_COMMIT", None)
    metadata_path = os.path.join(PROJECT_ROOT, 'models', 'metadata.json')
    model_meta = None
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_meta = json.load(f)
    except Exception:
        model_meta = None
    return jsonify({
        "version": app_version,
        "env": app_env,
        "commit": commit,
        "python": platform.python_version(),
        "model": model_meta,
        "ontology": ontology_metadata(),
    })

# Back-compat alias under /api
@app.route("/api/version", methods=["GET"])
def api_version():
    return version()


@app.route("/metrics", methods=["GET"])
def metrics():
    return app.response_class(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/api/drift/baseline", methods=["GET"])
def drift_baseline():
    """Return baseline class distribution and data quality signals for drift monitoring.
    Falls back gracefully if metadata not available."""
    metadata_path = os.path.join(PROJECT_ROOT, 'models', 'metadata.json')
    baseline = {}
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f) or {}
            baseline = {
                "class_distribution": meta.get('class_distribution'),
                "duplicate_ratio": meta.get('duplicate_ratio'),
                "dataset_rows": meta.get('dataset_rows'),
                "dataset_hash": meta.get('dataset_hash'),
                "trained_at": meta.get('trained_at'),
            }
    except Exception as e:
        baseline = {"error": f"failed to load metadata: {e}"}
    return jsonify(baseline)


# ----------------------------- Ontology & Hierarchical Metrics -----------------------------

def _ontology_tree_cached() -> Dict[str, Any]:  # lightweight in-process cache via global attribute
    cache_key = '_ontology_cache'
    if not hasattr(app, cache_key):
        try:
            data = load_ontology() or {}
        except Exception:
            data = {}
        setattr(app, cache_key, data)
    return getattr(app, cache_key)


def _build_parent_map(node: Dict[str, Any], parent: Optional[str], acc: Dict[str, Optional[str]]):
    nid = node.get('id')
    if nid:
        acc[nid] = parent
    for ch in (node.get('children') or []):
        _build_parent_map(ch, nid, acc)


def _aggregate_hierarchical_f1(per_class_f1: Dict[str, float], class_counts: Dict[str, int]) -> Dict[str, Any]:
    """Aggregate leaf-level F1 and (optional) supports count-weighted macro for internal nodes.

    We treat provided per_class_f1 keys as either leaf ids or legacy coarse labels; if legacy, we
    attempt to map them through ontology mapping (map_coarse_label) to leaf ids before aggregation.
    """
    ontology = _ontology_tree_cached()
    root = ontology.get('root') or {}
    if not root:
        return {}
    # Parent map & collect leaves
    parent_map: Dict[str, Optional[str]] = {}
    _build_parent_map(root, None, parent_map)
    leaves = {l['id'] for l in flatten_leaves(ontology)} if ontology else set()

    # Normalize keys -> leaf ids
    leaf_f1: Dict[str, float] = {}
    for k, v in per_class_f1.items():
        leaf_id, _ = map_coarse_label(k)
        # If map returns same k and it's already a leaf keep it
        if leaf_id not in leaves and k in leaves:
            leaf_id = k
        leaf_f1[leaf_id] = float(v)

    # Build children index
    children: Dict[str, List[str]] = {nid: [] for nid in parent_map}
    for nid, parent in parent_map.items():
        if parent is not None and parent in children:
            children[parent].append(nid)

    # Post-order traversal for aggregation
    aggregated: Dict[str, Dict[str, Any]] = {}

    def visit(nid: str):
        ch = children.get(nid, [])
        if not ch:  # leaf
            f1 = leaf_f1.get(nid)
            cnt = class_counts.get(nid) or 0
            aggregated[nid] = {"f1": f1, "count": cnt, "children": []}
            return aggregated[nid]
        child_nodes = [visit(c) for c in ch]
        # Compute macro (unweighted) over available child f1 values
        child_f1_vals = [c.get('f1') for c in child_nodes if c.get('f1') is not None]
        macro = float(sum(child_f1_vals) / len(child_f1_vals)) if child_f1_vals else None
        # Weighted by counts if counts available
        total_count = sum(c.get('count', 0) for c in child_nodes)
        if total_count > 0:
            weighted_num = sum((c.get('f1') or 0.0) * c.get('count', 0) for c in child_nodes)
            weighted = float(weighted_num / total_count) if weighted_num else 0.0
        else:
            weighted = None
        aggregated[nid] = {
            "f1_macro_children": macro,
            "f1_weighted_children": weighted,
            "count": total_count,
            "children": [{"id": c_id, **aggregated[c_id]} for c_id in ch]
        }
        return aggregated[nid]

    visit(root.get('id'))
    return aggregated.get(root.get('id'), {})


@app.route('/api/ontology', methods=['GET'])
def ontology_endpoint():
    """Return ontology tree (versioned) with derived metadata (num_leaves)."""
    data = _ontology_tree_cached()
    if not data:
        return jsonify({"error": "ontology_unavailable"}), 503
    meta = ontology_metadata() if data else {}
    return jsonify({"ontology": data.get('root'), "version": data.get('version'), **meta})


@app.route('/api/metrics/hierarchical', methods=['GET'])
def hierarchical_metrics():
    """Aggregate per-class F1 metrics to internal ontology nodes.

    Response structure:
      {
        "root": { ... aggregated node ... },
        "version": <ontology_version>,
        "leaf_count": <num_leaves>
      }
    """
    data = _ontology_tree_cached()
    if not data:
        return jsonify({"error": "ontology_unavailable"}), 503
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    metrics_path = os.path.join(models_dir, 'metrics.json')
    metadata_path = os.path.join(models_dir, 'metadata.json')
    per_class_f1: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                mj = json.load(f) or {}
            per_class_f1 = (mj.get('final_model') or {}).get('per_class_f1') or {}
    except Exception:
        per_class_f1 = {}
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                md = json.load(f) or {}
            class_counts = (md.get('class_distribution') or {})
    except Exception:
        class_counts = {}
    aggregated = _aggregate_hierarchical_f1(per_class_f1, class_counts)
    return jsonify({
        "root": aggregated,
        "version": data.get('version'),
        "leaf_count": len(flatten_leaves(data)),
    })


def _js_divergence(p: List[float], q: List[float]) -> float:
    import math
    # Ensure same length
    if len(p) != len(q):
        raise ValueError("Distribution length mismatch")
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kld(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps); bi = max(bi, eps)
            s += ai * math.log(ai / bi)
        return s
    return float((kld(p, m) + kld(q, m)) / 2.0)


@app.route("/api/drift/compare", methods=["POST"])
def drift_compare():
    auth = require_api_key()
    if auth:
        return auth
    metadata_path = os.path.join(PROJECT_ROOT, 'models', 'metadata.json')
    if not os.path.exists(metadata_path):
        return jsonify({"error": "baseline_unavailable"}), 503
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f) or {}
    except Exception as e:
        return jsonify({"error": f"failed_to_load_metadata: {e}"}), 500
    baseline_dist = meta.get('class_distribution') or {}
    if not baseline_dist:
        return jsonify({"error": "baseline_distribution_missing"}), 500
    classes = list(baseline_dist.keys())
    baseline_counts = [baseline_dist[c] for c in classes]
    baseline_total = sum(baseline_counts)
    if baseline_total <= 0:
        return jsonify({"error": "invalid_baseline_total"}), 500
    baseline_norm = [c / baseline_total for c in baseline_counts]

    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    try:
        payload = DriftCompareRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    incoming_norm = payload.normalized(classes)
    jsd = _js_divergence(baseline_norm, incoming_norm)

    warn_threshold = float(os.getenv('DRIFT_JSD_WARN', '0.10'))
    alert_threshold = float(os.getenv('DRIFT_JSD_ALERT', '0.20'))
    status = 'ok'
    if jsd >= alert_threshold:
        status = 'alert'
    elif jsd >= warn_threshold:
        status = 'warn'

    result = {
        "classes": classes,
        "baseline_distribution": baseline_norm,
        "incoming_distribution": incoming_norm,
        "jsd": jsd,
        "status": status,
        "warn_threshold": warn_threshold,
        "alert_threshold": alert_threshold,
    "evaluated_at": datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    }
    # Append drift history (best-effort, non-blocking)
    try:
        drift_log_dir = os.path.join(PROJECT_ROOT, 'logs')
        os.makedirs(drift_log_dir, exist_ok=True)
        drift_log_path = os.path.join(drift_log_dir, 'drift_history.log')
        with open(drift_log_path, 'a', encoding='utf-8') as df:
            df.write(json.dumps(result) + "\n")
    except Exception:
        pass
    return jsonify(result)


@app.route("/api/drift/history", methods=["GET"])
def drift_history():
    """Return the most recent drift comparison events (tail of log).
    Query params: limit (default 50, max 200)."""
    limit = 50
    try:
        if 'limit' in request.args:
            limit = int(request.args.get('limit', '50') or 50)
    except Exception:
        pass
    limit = max(1, min(limit, 200))
    drift_log_path = os.path.join(PROJECT_ROOT, 'logs', 'drift_history.log')
    if not os.path.exists(drift_log_path):
        return jsonify({"events": []})
    events: list[dict] = []
    try:
        with open(drift_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    except Exception as e:
        return jsonify({"error": f"failed_to_read_history: {e}"}), 500
    return jsonify({"events": events, "count": len(events)})


@app.route("/api/metrics/model", methods=["GET"])
def model_metrics():
    """Return consolidated model evaluation metrics (final_model section) and run metadata.
    Gracefully returns empty object if artifacts missing."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    metrics_path = os.path.join(models_dir, 'metrics.json')
    metadata_path = os.path.join(models_dir, 'metadata.json')
    metrics_data = {}
    meta_data = {}
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                full = json.load(f) or {}
                metrics_data = full.get('final_model', {})
    except Exception as e:
        metrics_data = {"error": f"failed_to_load_metrics: {e}"}
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f) or {}
    except Exception as e:
        meta_data = {"error": f"failed_to_load_metadata: {e}"}
    return jsonify({
        "metrics": metrics_data,
        "metadata": {k: meta_data.get(k) for k in [
            'run_id','previous_run','trained_at','dataset_rows','num_classes','duplicate_ratio','test_accuracy','test_macro_f1'
        ] if k in meta_data}
    })

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
        logger.info(__import__('json').dumps(log_obj, ensure_ascii=False))
    except Exception:
        pass
    # Metrics: count and latency
    try:
        ep = getattr(g, 'endpoint_for_metrics', request.path)
        REQUEST_COUNT.labels(request.method, ep, response.status_code).inc()
        REQUEST_LATENCY.labels(ep).observe(duration)
    except Exception:
        pass
    # Always attach request ID
    if getattr(g, 'request_id', None):
        response.headers["X-Request-ID"] = g.request_id
    return response

# Model artifact path (classical)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "legal_case_classifier.pkl"))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train it first.")

with open(MODEL_PATH, "rb") as f:
    saved = dill.load(f)
    classifier_model = saved["model"]
    label_encoder = saved["label_encoder"]
    preprocess_fn = saved.get("preprocessor")
    if preprocess_fn is None:
        # Fallback: build a preprocessor using the class method if available
        try:
            from ai_court.model.legal_case_classifier import LegalCaseClassifier  # type: ignore
            preprocess_fn = LegalCaseClassifier().preprocess_text
        except Exception:
            # Last resort: identity function
            def _identity(x: str) -> str:
                return x
            preprocess_fn = _identity
    # Force a robust safe preprocessor regardless of artifact contents to eliminate stale global refs
    _orig = preprocess_fn
    def _universal_safe_preprocess(text: str):  # type: ignore
        try:
            if _orig is not None:
                try:
                    return _orig(text)  # type: ignore
                except NameError:
                    pass  # fall through to fallback cleaning
        except Exception:
            pass
        import re as _re
        if not isinstance(text, str):
            return ""
        t = text.lower()
        t = _re.sub(r"[^a-z\s]", " ", t)
        t = _re.sub(r"\s+", " ", t).strip()
        return t
    preprocess_fn = _universal_safe_preprocess
    print(f"[api] Loaded model from {MODEL_PATH}")

# Optional multi-axis promoted model load for shadow inference & agreement tracking
multi_axis_bundle = None
multi_axis_lock = threading.Lock()
_agreement_stats = {
    'total_compared': 0,
    'agreements': 0,
    'last_samples': []  # last discrepancy samples
}

def _persist_agreement():
    try:
        out = {
            'total_compared': _agreement_stats['total_compared'],
            'agreements': _agreement_stats['agreements'],
            'rate': (_agreement_stats['agreements']/ _agreement_stats['total_compared']) if _agreement_stats['total_compared'] else None,
            'last_samples': _agreement_stats['last_samples']
        }
        with open('agreement_stats.json','w',encoding='utf-8') as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

def _record_agreement(classical_label: str, shadow_axes: dict):
    rel = shadow_axes.get('relief_label') if isinstance(shadow_axes, dict) else None
    subst = shadow_axes.get('substantive_label') if isinstance(shadow_axes, dict) else None
    proc = shadow_axes.get('procedural_label') if isinstance(shadow_axes, dict) else None
    candidate = rel or subst or proc
    if candidate is None:
        return
    match = (candidate == classical_label)
    _agreement_stats['total_compared'] += 1
    if match:
        _agreement_stats['agreements'] += 1
    else:
        if len(_agreement_stats['last_samples']) > 20:
            _agreement_stats['last_samples'].pop(0)
        _agreement_stats['last_samples'].append({'classical': classical_label, 'multi_axis': candidate})
    # Persist periodically every 10 comparisons
    if _agreement_stats['total_compared'] % 10 == 0:
        _persist_agreement()
def _load_multi_axis():
    base_dir = os.path.join(PROJECT_ROOT,'models','multi_axis')
    prom_path = os.path.join(base_dir,'promoted.json')
    ckpt_path = os.path.join(base_dir,'multi_axis.pt')
    if not os.path.exists(prom_path) or not os.path.exists(ckpt_path):
        return None
    try:
        import torch
        from transformers import AutoTokenizer
        with open(prom_path,'r',encoding='utf-8') as f:
            pm = json.load(f) or {}
        run_id = pm.get('run_id')
        bundle = torch.load(ckpt_path, map_location='cpu')
        from ai_court.model.multi_axis_transformer import MultiAxisModel  # type: ignore
        model = MultiAxisModel(bundle['backbone'], {ax: len(m) for ax,m in bundle['label_maps'].items()})
        model.load_state_dict(bundle['model_state'])
        model.eval()
        tok = AutoTokenizer.from_pretrained(bundle['backbone'])
        return {'model': model, 'tokenizer': tok, 'label_maps': bundle['label_maps'], 'backbone': bundle['backbone'], 'run_id': run_id}
    except Exception as e:  # pragma: no cover
        print(f"[api] multi-axis load failed: {e}")
        return None

if os.getenv('ENABLE_MULTI_AXIS_SHADOW','1') == '1':
    multi_axis_bundle = _load_multi_axis()
    if multi_axis_bundle:
        print(f"[api] Loaded multi-axis promoted model (run {multi_axis_bundle.get('run_id')}) for shadow inference")

def _multi_axis_predict_single(text: str):
    b = multi_axis_bundle
    if not b:
        return None
    import torch
    max_len = int(os.getenv('MULTI_AXIS_INFER_MAX_LEN','384'))
    tok = b['tokenizer']
    model = b['model']
    enc = tok([text], truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
    inv = {ax: {v:k for k,v in b['label_maps'][ax].items()} for ax in b['label_maps']}
    pred_axes = {ax: inv[ax][int(logits[ax].argmax(dim=1).item())] for ax in b['label_maps']}
    main_judgment = pred_axes.get('relief_label') or pred_axes.get('substantive_label') or pred_axes.get('procedural_label')
    return {'axes': pred_axes, 'main_judgment': main_judgment}

def _record_agreement(classical_label: str, shadow_axes: dict | None):
    if not shadow_axes or not isinstance(shadow_axes, dict):
        return
    unified_candidate = shadow_axes.get('relief_label') or shadow_axes.get('substantive_label') or shadow_axes.get('procedural_label')
    if unified_candidate is None:
        return
    _agreement_stats['total_compared'] += 1
    if classical_label == unified_candidate:
        _agreement_stats['agreements'] += 1
    else:
        if len(_agreement_stats['last_samples']) >= 10:
            _agreement_stats['last_samples'].pop(0)
        _agreement_stats['last_samples'].append({
            'classical': classical_label,
            'multi_axis_relief': shadow_axes.get('relief_label'),
            'multi_axis_substantive': shadow_axes.get('substantive_label'),
            'multi_axis_procedural': shadow_axes.get('procedural_label')
        })

LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"
# Optional semantic search index (can be disabled to reduce memory)
SEARCH_INDEX_PATH = os.getenv("SEARCH_INDEX_PATH", os.path.join(PROJECT_ROOT, "models", "search_index.pkl"))
DISABLE_SEARCH_INDEX = os.getenv("DISABLE_SEARCH_INDEX", "1" if LOW_MEMORY else "0") == "1"
search_index: Optional[Dict[str, Any]] = None
if not DISABLE_SEARCH_INDEX and os.path.exists(SEARCH_INDEX_PATH):
    try:
        with open(SEARCH_INDEX_PATH, "rb") as f:
            search_index = dill.load(f)
        print(f"[api] Loaded search index from {SEARCH_INDEX_PATH} with {len(search_index.get('meta', []))} docs")
    except Exception as e:
        print(f"[api] Warning: failed to load search index: {e}")
else:
    if DISABLE_SEARCH_INDEX:
        print("[api] Skipping search index load (DISABLE_SEARCH_INDEX=1 or LOW_MEMORY=1)")

# Optional semantic (dense) search index (can be disabled to reduce memory)
SEMANTIC_INDEX_PATH = os.getenv("SEMANTIC_INDEX_PATH", os.path.join(PROJECT_ROOT, "models", "semantic_index.pkl"))
DISABLE_SEMANTIC_INDEX = os.getenv("DISABLE_SEMANTIC_INDEX", "1" if LOW_MEMORY else "0") == "1"
semantic_index: Optional[Dict[str, Any]] = None
if not DISABLE_SEMANTIC_INDEX and os.path.exists(SEMANTIC_INDEX_PATH):
    try:
        with open(SEMANTIC_INDEX_PATH, 'rb') as f:
            semantic_index = dill.load(f)
        emb_count = getattr(semantic_index.get('embeddings', []), 'shape', [len(semantic_index.get('meta', []))])[0]
        print(f"[api] Loaded semantic index from {SEMANTIC_INDEX_PATH} with {emb_count} embeddings")
    except Exception as e:
        print(f"[api] Warning: failed to load semantic index: {e}")
else:
    if DISABLE_SEMANTIC_INDEX:
        print("[api] Skipping semantic index load (DISABLE_SEMANTIC_INDEX=1 or LOW_MEMORY=1)")

CASE_TYPES = {
    "initial": [
        {"id": "case_type", "question": "What type of case is this?", "options": ["Criminal", "Civil", "Family", "Labor"]}
    ],
    "Criminal": [
        {"id": "parties", "question": "Who are the parties involved in the case?"},
        {"id": "violence_level", "question": "What was the level of violence involved?",
         "options": ["None", "Threat only", "Minor injury", "Serious injury", "Death"]},
        {"id": "weapon", "question": "Was any weapon used?", "options": ["Yes", "No"]},
        {"id": "police_report", "question": "Has a police report been filed?", "options": ["Yes", "No"]},
        {"id": "witnesses", "question": "Are there any witnesses?", "options": ["Yes", "No"]},
        {"id": "premeditation", "question": "Was the act premeditated or spontaneous?",
         "options": ["Premeditated", "Spontaneous", "Unclear"]}
    ],
    "Family": [
        {"id": "parties", "question": "Who are the parties involved in the dispute?"},
        {"id": "marriage_duration", "question": "How long have the parties been married? (if applicable)"},
        {"id": "children", "question": "Are there children involved?", "options": ["Yes", "No"]},
        {"id": "property", "question": "Is there shared property in dispute?", "options": ["Yes", "No"]},
        {"id": "previous_agreements", "question": "Are there any previous agreements between parties?", "options": ["Yes", "No"]},
        {"id": "violence", "question": "Has there been any domestic violence?", "options": ["Yes", "No"]}
    ],
    "Civil": [
        {"id": "parties", "question": "Who are the parties involved in the case?"},
        {"id": "dispute_type", "question": "What is the nature of the dispute?",
         "options": ["Contract", "Property", "Debt", "Damages", "Other"]},
        {"id": "document_evidence", "question": "Is there documentary evidence?", "options": ["Yes", "No"]},
        {"id": "monetary_value", "question": "What is the approximate monetary value of the dispute?"},
        {"id": "prior_relationship", "question": "What was the prior relationship between parties?"},
        {"id": "attempts_resolution", "question": "Have there been previous attempts at resolution?", "options": ["Yes", "No"]}
    ],
    "Labor": [
        {"id": "parties", "question": "Who are the involved parties? (employer/employee)"},
        {"id": "employment_duration", "question": "How long was the employment period?"},
        {"id": "contract_type", "question": "What type of employment contract was in place?",
         "options": ["Full-time", "Part-time", "Contract", "None", "Other"]},
        {"id": "dispute_reason", "question": "What is the main reason for the dispute?",
         "options": ["Termination", "Wages", "Working conditions", "Discrimination", "Other"]},
        {"id": "union_involvement", "question": "Is there union involvement?", "options": ["Yes", "No"]},
        {"id": "previous_complaints", "question": "Were there previous formal complaints?", "options": ["Yes", "No"]}
    ]
}


def _semantic_query_embeddings(query: str):
    """Encode query using the same sentence-transformer model as index if available."""
    if semantic_index is None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    model_name = semantic_index.get('model_name') or 'all-MiniLM-L6-v2'
    try:
        model = SentenceTransformer(model_name)
        vec = model.encode([preprocess_fn(query)], normalize_embeddings=True)
        return vec
    except Exception:
        return None


@app.route("/api/questions", methods=["GET"])
def get_questions():
    return jsonify(CASE_TYPES)


@app.route("/api/health", methods=["GET"])
def health():
    try:
        # minimal prediction to ensure model and encoder are responsive
        _ = classifier_model.predict([preprocess_fn("healthcheck")])
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/api/questions/<case_type>", methods=["GET"])
def get_questions_by_type(case_type):
    if case_type in CASE_TYPES:
        return jsonify(CASE_TYPES[case_type])
    return jsonify({"error": "Case type not found"}), 404


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("30/minute")
@swag_from({
    'tags': ['analyze'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'case_type': {'type': 'string', 'example': 'Criminal'},
                'parties': {'type': 'string'},
                'violence_level': {'type': 'string'},
                'weapon': {'type': 'string'},
                'police_report': {'type': 'string'},
                'witnesses': {'type': 'string'},
                'premeditation': {'type': 'string'},
            }
        }
    }],
    'responses': {200: {'description': 'OK'}}
})
def analyze_case():
    # API key check
    auth = require_api_key()
    if auth:
        return auth
    raw = request.json
    if raw is None:
        return jsonify({"error": "Expected application/json body"}), 400
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = AnalyzeRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    payload_str = __import__('json').dumps(raw)
    if len(payload_str) > 100_000:
        return jsonify({"error": "Payload too large"}), 413
    processed = preprocess_fn(parsed.combined_text())
    # Prediction with probability (classical model)
    proba = None
    try:
        proba = classifier_model.predict_proba([processed])[0]
        pred = int(proba.argmax())
        confidence = float(proba.max())
    except Exception:
        pred = int(classifier_model.predict([processed])[0])
        confidence = None
    judgment = label_encoder.inverse_transform([pred])[0]
    try:
        PREDICTIONS_TOTAL.labels('analyze', judgment).inc()
    except Exception:
        pass

    inferred_case_type = parsed.case_type or "Unknown"
    data = raw
    if inferred_case_type == "Unknown":
        if raw.get("violence_level", "None") != "None" or raw.get("police_report") == "Yes":
            inferred_case_type = "Criminal"
        elif raw.get("employment_duration"):
            inferred_case_type = "Labor"
        elif raw.get("children") or raw.get("marriage_duration"):
            inferred_case_type = "Family"
        else:
            inferred_case_type = "Civil"

    use_primary_multi = os.getenv('USE_MULTI_AXIS_PRIMARY','0') == '1'
    shadow = None
    if multi_axis_bundle:
        try:
            shadow = _multi_axis_predict_single(processed)
        except Exception:
            shadow = {'error':'shadow_inference_failed'}
    primary_judgment = judgment
    if use_primary_multi and isinstance(shadow, dict):
        primary_judgment = (shadow.get('main_judgment') if isinstance(shadow.get('main_judgment'), str) else None) or judgment
    # Record agreement stats (always compare classical vs relief promotion candidate)
    if shadow and isinstance(shadow, dict) and 'axes' in shadow:
        try:
            _record_agreement(judgment, shadow['axes'])
        except Exception:
            pass
    agreement_rate = None
    if _agreement_stats['total_compared'] > 0:
        agreement_rate = _agreement_stats['agreements'] / _agreement_stats['total_compared']
    return jsonify({
        "case_type": inferred_case_type,
        "judgment": primary_judgment,
        "judgment_classical": judgment,
        "judgment_source": 'multi_axis' if use_primary_multi and shadow else 'classical',
        "answers": data,
        "confidence": confidence,
        "shadow_multi_axis": shadow,
        "agreement_rate": agreement_rate
    })

@app.route('/api/metrics/agreement', methods=['GET'])
def agreement_metrics():
    rate = (_agreement_stats['agreements']/ _agreement_stats['total_compared']) if _agreement_stats['total_compared'] else None
    return jsonify({
        'total_compared': _agreement_stats['total_compared'],
        'agreements': _agreement_stats['agreements'],
        'rate': rate,
        'last_samples': _agreement_stats['last_samples']
    })


@app.route("/api/search", methods=["POST"])
@limiter.limit("60/minute")
@swag_from({
    'tags': ['search'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'k': {'type': 'integer'}}}
    }],
    'responses': {200: {'description': 'OK'}}
})
def search_cases():
    auth = require_api_key()
    if auth:
        return auth
    if search_index is None and semantic_index is None:
        return jsonify({"error": "No search index available. Build TF-IDF or semantic index."}), 503

    raw = request.json or {}
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = SearchRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    query = parsed.query.strip()
    k = parsed.k

    results = []
    semantic_results = []
    lexical_results = []
    processed = preprocess_fn(query)
    # Prefer semantic index if present
    if semantic_index is not None:
        try:
            dense = semantic_index.get('embeddings')
            meta_sem = semantic_index.get('meta', [])
            qv = _semantic_query_embeddings(processed)
            if qv is not None and hasattr(dense, 'shape'):
                # Cosine similarity (embeddings are normalized already)
                sims = np.dot(dense, qv[0])  # shape [N]
                order = np.argsort(-sims)[:k]
                for idx in order:
                    if idx < len(meta_sem):
                        m = meta_sem[idx]
                        semantic_results.append({
                            'title': m.get('title','Unknown'),
                            'url': m.get('url'),
                            'outcome': m.get('outcome'),
                            'snippet': m.get('snippet'),
                            'score': float(sims[idx])
                        })
        except Exception:
            pass
    # Lexical TF-IDF retrieval
    if search_index is not None:
        vect = search_index["vectorizer"]
        matrix = search_index["matrix"]  # sparse matrix
        meta = search_index.get("meta", [])
        qv = vect.transform([processed])
        scores = (matrix @ qv.T).toarray().ravel()
        if scores.size > 0:
            top_idx = np.argsort(-scores)[:k]
            for idx in top_idx:
                if idx < len(meta):
                    m = meta[idx]
                    lexical_results.append({
                        "title": m.get("title", "Unknown"),
                        "url": m.get("url"),
                        "outcome": m.get("outcome"),
                        "snippet": m.get("snippet"),
                        "score": float(scores[idx])
                    })
    # If both present perform reciprocal rank fusion (RRF)
    fused: List[Dict[str, Any]] = []
    if semantic_results and lexical_results:
        # Build ranking position maps
        k_sem = { (r.get('url'), r.get('title')): i for i,r in enumerate(semantic_results) }
        k_lex = { (r.get('url'), r.get('title')): i for i,r in enumerate(lexical_results) }
        all_keys = set(k_sem.keys()) | set(k_lex.keys())
        K = 60  # constant to dampen
        for key in all_keys:
            sem_rank = k_sem.get(key, 10**6)
            lex_rank = k_lex.get(key, 10**6)
            sem_score = 1.0 / (K + sem_rank)
            lex_score = 1.0 / (K + lex_rank)
            # Retrieve one of the records (prefer semantic)
            base = None
            for r in semantic_results:
                if (r.get('url'), r.get('title')) == key:
                    base = r; break
            if base is None:
                for r in lexical_results:
                    if (r.get('url'), r.get('title')) == key:
                        base = r; break
            if base is None:
                continue
            fused.append({**base, 'fusion_score': sem_score + lex_score})
        fused.sort(key=lambda x: -x['fusion_score'])
        results = fused[:k]
    else:
        results = (semantic_results or lexical_results)[:k]
    return jsonify({"results": results})


# ----------------------------- Active Learning Queue (In-Memory) -----------------------------
AL_QUEUE: List[Dict[str, Any]] = []

@app.route('/api/active_learning/queue', methods=['GET'])
def al_queue_get():
    return jsonify({"pending": AL_QUEUE, "size": len(AL_QUEUE)})


@app.route('/api/active_learning/queue', methods=['POST'])
def al_queue_add():
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    text = raw.get('text')
    if not text:
        return jsonify({"error": "missing_text"}), 400
    item = {
        'id': str(uuid.uuid4()),
        'text': text,
        'added_at': datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    }
    AL_QUEUE.append(item)
    return jsonify(item), 201


@app.route('/api/active_learning/queue/<item_id>', methods=['POST'])
def al_queue_label(item_id: str):
    raw = request.json or {}
    label = raw.get('label') if isinstance(raw, dict) else None
    for i, it in enumerate(AL_QUEUE):
        if it['id'] == item_id:
            it['label'] = label
            it['labeled_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
            return jsonify(it)
    return jsonify({"error": "not_found"}), 404


# ----------------------------- RAG Query Stub -----------------------------
@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    question = raw.get('question') or ''
    if not question:
        return jsonify({"error": "missing_question"}), 400
    # Minimal retrieval: reuse search endpoint logic for top docs (lexical only)
    docs: List[Dict[str, Any]] = []
    if search_index is not None:
        vect = search_index['vectorizer']
        matrix = search_index['matrix']
        meta = search_index.get('meta', [])
        qv = vect.transform([preprocess_fn(question)])
        scores = (matrix @ qv.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:3]
        for idx in top_idx:
            if idx < len(meta):
                m = meta[idx]
                docs.append({
                    'title': m.get('title','Unknown'),
                    'url': m.get('url'),
                    'outcome': m.get('outcome'),
                    'snippet': m.get('snippet'),
                    'score': float(scores[idx])
                })
    # Placeholder answer (no generation model integrated yet)
    answer = "RAG pipeline not yet fully implemented. Retrieved top documents only."
    return jsonify({
        'question': question,
        'answer': answer,
        'documents': docs,
        'num_documents': len(docs)
    })


@app.route("/api/analyze_and_search", methods=["POST"])
@limiter.limit("20/minute")
@swag_from({
    'tags': ['analyze','search'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {'type': 'object'}
    }],
    'responses': {200: {'description': 'OK'}}
})
def analyze_and_search():
    auth = require_api_key()
    if auth:
        return auth
    raw = request.json or {}
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = AnalyzeRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    data = raw
    processed = preprocess_fn(parsed.combined_text())
    try:
        proba = classifier_model.predict_proba([processed])[0]
        pred = int(proba.argmax())
        confidence = float(proba.max())
    except Exception:
        pred = int(classifier_model.predict([processed])[0])
        confidence = None
    judgment = label_encoder.inverse_transform([pred])[0]
    try:
        PREDICTIONS_TOTAL.labels('analyze_and_search', judgment).inc()
    except Exception:
        pass

    inferred_case_type = parsed.case_type or "Unknown"
    if inferred_case_type == "Unknown":
        if raw.get("violence_level", "None") != "None" or raw.get("police_report") == "Yes":
            inferred_case_type = "Criminal"
        elif raw.get("employment_duration"):
            inferred_case_type = "Labor"
        elif raw.get("children") or raw.get("marriage_duration"):
            inferred_case_type = "Family"
        else:
            inferred_case_type = "Civil"

    # Search similar
    search_results = []
    if search_index is not None:
        vect = search_index["vectorizer"]
        matrix = search_index["matrix"]
        meta = search_index.get("meta", [])
        qv = vect.transform([processed])
        scores = (matrix @ qv.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:5]
        for idx in top_idx:
            if idx < len(meta):
                m = meta[idx]
                search_results.append({
                    "title": m.get("title", "Unknown"),
                    "url": m.get("url"),
                    "outcome": m.get("outcome"),
                    "snippet": m.get("snippet"),
                    "score": float(scores[idx])
                })

    return jsonify({
        "case_type": inferred_case_type,
        "judgment": judgment,
        "answers": data,
        "similar": search_results,
        "confidence": confidence,
    })


# ensure single metrics endpoint is defined once above


if __name__ == "__main__":
    app.run(debug=True, port=5002, host="0.0.0.0")

import os
import json
import platform
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, Response
from pydantic import ValidationError

from ai_court.api import config, state, dependencies, models
from ai_court.ontology import ontology_metadata, flatten_leaves

monitoring_bp = Blueprint('monitoring', __name__)

@monitoring_bp.route("/metrics", methods=["GET"])
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@monitoring_bp.route("/version", methods=["GET"])
def version():
    metadata_path = os.path.join(config.PROJECT_ROOT, 'models', 'metadata.json')
    model_meta = None
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_meta = json.load(f)
    except Exception:
        model_meta = None
    return jsonify({
        "version": config.APP_VERSION,
        "env": config.APP_ENV,
        "commit": os.getenv("GIT_COMMIT"),
        "python": platform.python_version(),
        "model": model_meta,
        "ontology": ontology_metadata(),
    })

@monitoring_bp.route("/api/version", methods=["GET"])
def api_version():
    return version()

@monitoring_bp.route("/api/health", methods=["GET"])
def health():
    try:
        _ = state.classifier_model.predict([state.preprocess_fn("healthcheck")])
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@monitoring_bp.route("/api/health/ready", methods=["GET"])
def health_ready():
    """Readiness probe - checks if model is loaded and ready to serve."""
    checks = {
        'model_loaded': state.classifier is not None,
        'preprocessor_ready': state.preprocess_fn is not None,
    }
    all_ready = all(checks.values())
    return jsonify({
        "ready": all_ready,
        "checks": checks
    }), 200 if all_ready else 503

@monitoring_bp.route("/api/health/live", methods=["GET"])
def health_live():
    """Liveness probe - minimal check that service is running."""
    return jsonify({"alive": True}), 200

@monitoring_bp.route("/api/stats/memory", methods=["GET"])
def memory_stats():
    """Return current memory usage statistics."""
    memory_info = state.get_memory_usage()
    return jsonify({
        "memory": memory_info,
        "search_index_loaded": state.search_index is not None,
        "semantic_index_loaded": state.semantic_index is not None,
        "multi_axis_loaded": state.multi_axis_bundle is not None,
    })

@monitoring_bp.route("/api/stats/predictions", methods=["GET"])
def prediction_stats():
    """Return prediction statistics for monitoring."""
    stats = state.prediction_stats.copy()
    # Remove internal tracking fields
    stats.pop('_confidence_sum', None)
    
    # Add AL queue info
    stats['al_queue_size'] = len(state.AL_QUEUE)
    stats['agreement_stats'] = {
        'total_compared': state.agreement_stats['total_compared'],
        'agreements': state.agreement_stats['agreements'],
        'rate': (state.agreement_stats['agreements'] / state.agreement_stats['total_compared']) 
                if state.agreement_stats['total_compared'] > 0 else None
    }
    
    return jsonify(stats)

@monitoring_bp.route('/api/reload_multi_axis', methods=['POST'])
def reload_multi_axis():
    """Reload the promoted multi-axis model bundle without restarting the server."""
    try:
        dependencies.load_multi_axis()
        if state.multi_axis_bundle:
            return jsonify({'status':'ok','run_id':state.multi_axis_bundle.get('run_id')}), 200
        return jsonify({'status':'error','detail':'No promoted model found'}), 404
    except Exception as e:
        return jsonify({'status':'error','detail':str(e)}), 500

@monitoring_bp.route('/api/governance/status', methods=['GET'])
def governance_status():
    """Return consolidated governance status if generated (best-effort)."""
    path = config.GOVERNANCE_STATUS_PATH
    if not os.path.exists(path):
        return jsonify({'error':'status_unavailable'}), 404
    try:
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error':'load_failed','detail':str(e)}), 500

@monitoring_bp.route('/api/governance/refresh', methods=['POST'])
def governance_refresh():
    """Force a refresh of governance_status.json by executing the script inline."""
    try:
        import subprocess
        import sys
        cmd = [sys.executable, 'scripts/governance_status.py']
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            return jsonify({'status':'error','detail': rc.stderr.strip()}), 500
        return jsonify({'status':'ok'}), 200
    except Exception as e:
        return jsonify({'status':'error','detail':str(e)}), 500

@monitoring_bp.route("/api/drift/baseline", methods=["GET"])
def drift_baseline():
    metadata_path = os.path.join(config.PROJECT_ROOT, 'models', 'metadata.json')
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

@monitoring_bp.route('/api/ontology', methods=['GET'])
def ontology_endpoint():
    data = dependencies.ontology_tree_cached()
    if not data:
        return jsonify({"error": "ontology_unavailable"}), 503
    meta = ontology_metadata() if data else {}
    return jsonify({"ontology": data.get('root'), "version": data.get('version'), **meta})

@monitoring_bp.route('/api/metrics/hierarchical', methods=['GET'])
def hierarchical_metrics():
    data = dependencies.ontology_tree_cached()
    if not data:
        return jsonify({"error": "ontology_unavailable"}), 503
    models_dir = os.path.join(config.PROJECT_ROOT, 'models')
    metrics_path = os.path.join(models_dir, 'metrics.json')
    metadata_path = os.path.join(models_dir, 'metadata.json')
    per_class_f1: dict = {}
    class_counts: dict = {}
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
    aggregated = dependencies.aggregate_hierarchical_f1(per_class_f1, class_counts)
    return jsonify({
        "root": aggregated,
        "version": data.get('version'),
        "leaf_count": len(flatten_leaves(data)),
    })

@monitoring_bp.route("/api/drift/compare", methods=["POST"])
def drift_compare():
    auth = dependencies.require_api_key()
    if auth:
        return auth
    metadata_path = os.path.join(config.PROJECT_ROOT, 'models', 'metadata.json')
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
        payload = models.DriftCompareRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    incoming_norm = payload.normalized(classes)
    jsd = dependencies.js_divergence(baseline_norm, incoming_norm)

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
    try:
        drift_log_dir = os.path.join(config.PROJECT_ROOT, 'logs')
        os.makedirs(drift_log_dir, exist_ok=True)
        drift_log_path = os.path.join(drift_log_dir, 'drift_history.log')
        with open(drift_log_path, 'a', encoding='utf-8') as df:
            df.write(json.dumps(result) + "\n")
    except Exception:
        pass
    return jsonify(result)

@monitoring_bp.route("/api/drift/history", methods=["GET"])
def drift_history():
    limit = 50
    try:
        if 'limit' in request.args:
            limit = int(request.args.get('limit', '50') or 50)
    except Exception:
        pass
    limit = max(1, min(limit, 200))
    drift_log_path = os.path.join(config.PROJECT_ROOT, 'logs', 'drift_history.log')
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

@monitoring_bp.route("/api/metrics/model", methods=["GET"])
def model_metrics():
    models_dir = os.path.join(config.PROJECT_ROOT, 'models')
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

@monitoring_bp.route('/api/metrics/agreement', methods=['GET'])
def agreement_metrics():
    rate = (state.agreement_stats['agreements']/ state.agreement_stats['total_compared']) if state.agreement_stats['total_compared'] else None
    return jsonify({
        'total_compared': state.agreement_stats['total_compared'],
        'agreements': state.agreement_stats['agreements'],
        'rate': rate,
        'last_samples': state.agreement_stats['last_samples']
    })

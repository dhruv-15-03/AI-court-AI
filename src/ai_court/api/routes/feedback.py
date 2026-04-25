import uuid
import json
import logging
import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from ai_court.api import state, config

feedback_bp = Blueprint('feedback', __name__)
logger = logging.getLogger(__name__)

AL_QUEUE_FILE = os.path.join(config.PROJECT_ROOT, "data", "al_queue.json")
LABELS_FILE = os.path.join(config.PROJECT_ROOT, "data", "active_learning_labels.jsonl")

# Lazy-init label store and retrain engine
_label_store = None
_retrain_engine = None


def _get_label_store():
    global _label_store
    if _label_store is None:
        from ai_court.active_learning.loop import LabeledDataStore
        _label_store = LabeledDataStore(store_path=LABELS_FILE)
    return _label_store


def _get_retrain_engine():
    global _retrain_engine
    if _retrain_engine is None:
        from ai_court.active_learning.loop import RetrainingEngine
        _retrain_engine = RetrainingEngine(
            label_store=_get_label_store(),
            retrain_threshold=int(os.getenv("RETRAIN_THRESHOLD", "50")),
            model_dir=os.path.join(config.PROJECT_ROOT, "models"),
        )
    return _retrain_engine


def load_queue():
    if os.path.exists(AL_QUEUE_FILE):
        try:
            with open(AL_QUEUE_FILE, 'r', encoding='utf-8') as f:
                state.AL_QUEUE = json.load(f)
        except Exception:
            state.AL_QUEUE = []

def save_queue():
    """Atomically persist AL queue to disk (write-to-temp + rename)."""
    try:
        os.makedirs(os.path.dirname(AL_QUEUE_FILE), exist_ok=True)
        tmp_path = AL_QUEUE_FILE + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(state.AL_QUEUE, f, indent=2)
        os.replace(tmp_path, AL_QUEUE_FILE)  # atomic on POSIX / near-atomic on Windows
    except Exception as exc:
        logger.warning("Failed to persist AL queue: %s", exc)

# Load on startup
load_queue()

@feedback_bp.route('/api/active_learning/queue', methods=['GET'])
def al_queue_get():
    return jsonify({"pending": state.AL_QUEUE, "size": len(state.AL_QUEUE)})

@feedback_bp.route('/api/active_learning/queue', methods=['POST'])
def al_queue_add():
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    text = raw.get('text')
    if not text:
        return jsonify({"error": "missing_text"}), 400

    # Compute proper uncertainty if probabilities provided
    probabilities = raw.get('probabilities', {})
    uncertainty_method = raw.get('uncertainty_method', 'entropy')
    uncertainty = 1.0
    if probabilities:
        from ai_court.active_learning.loop import compute_uncertainty
        uncertainty = compute_uncertainty(probabilities, method=uncertainty_method)

    item = {
        'id': str(uuid.uuid4()),
        'text': text[:2000],
        'uncertainty': round(uncertainty, 4),
        'uncertainty_method': uncertainty_method,
        'predicted_label': raw.get('predicted_label'),
        'probabilities': probabilities,
        'case_type': raw.get('case_type'),
        'added_at': datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    }
    state.AL_QUEUE.append(item)
    save_queue()
    return jsonify(item), 201

@feedback_bp.route('/api/active_learning/queue/<item_id>', methods=['POST'])
def al_queue_label(item_id: str):
    """Label a queued item and store it for retraining."""
    raw = request.json or {}
    label = raw.get('label') if isinstance(raw, dict) else None
    if not label:
        return jsonify({"error": "missing_label"}), 400

    for i, it in enumerate(state.AL_QUEUE):
        if it['id'] == item_id:
            it['label'] = label
            it['labeled_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
            save_queue()

            # Store the labeled data for retraining
            store = _get_label_store()
            store.add_label(
                text=it.get('text', ''),
                label=label,
                case_type=it.get('case_type', ''),
                source='human',
                metadata={'queue_item_id': item_id, 'predicted_label': it.get('predicted_label')},
            )

            # Check if retraining is needed
            engine = _get_retrain_engine()
            needs_retrain = engine.should_retrain()

            return jsonify({
                **it,
                "labels_stored": store.count(),
                "retrain_ready": needs_retrain,
            })
    return jsonify({"error": "not_found"}), 404


@feedback_bp.route('/api/active_learning/suggest_label', methods=['POST'])
def al_suggest_label():
    """Use LLM to suggest a label for an uncertain prediction."""
    if state.llm_client is None:
        return jsonify({"error": "llm_unavailable"}), 503
    raw = request.json or {}
    text = raw.get('text', '')
    if not text:
        return jsonify({"error": "missing_text"}), 400

    # Get valid labels from the model's label encoder
    valid_labels = []
    if state.classifier and hasattr(state.classifier, 'label_encoder'):
        valid_labels = list(state.classifier.label_encoder.classes_)
    if not valid_labels:
        valid_labels = [
            "Relief Granted/Convicted",
            "Relief Denied/Dismissed",
            "Acquittal/Conviction Overturned",
        ]

    from ai_court.active_learning.loop import suggest_label_with_llm
    result = suggest_label_with_llm(text, state.llm_client, valid_labels)
    if result is None:
        return jsonify({"error": "suggestion_failed"}), 500
    return jsonify(result)


@feedback_bp.route('/api/active_learning/retrain', methods=['POST'])
def al_retrain():
    """Trigger model retraining with accumulated labels."""
    engine = _get_retrain_engine()
    store = _get_label_store()

    label_count = store.count()
    if label_count == 0:
        return jsonify({"error": "no_labels", "message": "No labeled data available for retraining."}), 400

    try:
        result = engine.retrain()

        # Reload the model in state
        from ai_court.api.dependencies import load_model
        load_model()

        # Update agent pipeline with new classifier
        if state.agent_pipeline is not None and state.classifier is not None:
            state.agent_pipeline.classifier = state.classifier

        return jsonify(result)
    except Exception as exc:
        logger.error("Retraining failed: %s", exc, exc_info=True)
        return jsonify({"error": "retrain_failed", "message": str(exc)}), 500


@feedback_bp.route('/api/active_learning/stats', methods=['GET'])
def al_stats():
    """Get active learning statistics."""
    store = _get_label_store()
    engine = _get_retrain_engine()
    return jsonify({
        "queue_size": len(state.AL_QUEUE),
        "labels_stored": store.count(),
        "retrain_threshold": engine.retrain_threshold,
        "retrain_ready": engine.should_retrain(),
    })


@feedback_bp.route('/api/active_learning/sync_outcomes', methods=['POST'])
def al_sync_outcomes():
    """Pull recorded case outcomes from the Java backend and add them as labels.

    Body (all optional):
        java_url:         override JAVA_BACKEND_URL
        internal_key:     override INTERNAL_API_KEY
        auto_retrain:     bool, default True
    """
    from pathlib import Path
    try:
        from scripts.sync_case_outcomes import sync as _sync  # type: ignore
    except ModuleNotFoundError:
        import sys as _sys
        _sys.path.insert(0, config.PROJECT_ROOT)
        from scripts.sync_case_outcomes import sync as _sync  # type: ignore

    body = request.json or {}
    java_url = body.get("java_url") or os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")
    internal_key = body.get("internal_key") or os.getenv("INTERNAL_API_KEY")
    auto_retrain = bool(body.get("auto_retrain", True))
    try:
        result = _sync(
            java_url=java_url,
            internal_key=internal_key,
            labels_path=Path(LABELS_FILE),
            state_path=Path(config.PROJECT_ROOT) / "data" / "outcome_sync.json",
            auto_retrain=auto_retrain,
            retrain_threshold=int(os.getenv("RETRAIN_THRESHOLD", "50")),
        )
        # If retraining happened, hot-swap the classifier in the running pipeline.
        if result.get("retrained"):
            try:
                from ai_court.model.legal_case_classifier import LegalCaseClassifier
                clf = LegalCaseClassifier()
                clf.load_model(os.path.join(config.PROJECT_ROOT, "models",
                                            "legal_case_classifier.pkl"))
                state.classifier = clf
                if state.agent_pipeline is not None:
                    state.agent_pipeline.classifier = clf
            except Exception as exc:
                logger.warning("Hot-reload of retrained model failed: %s", exc)
        return jsonify(result)
    except Exception as exc:
        logger.error("sync_outcomes failed: %s", exc, exc_info=True)
        return jsonify({"error": "sync_failed", "message": str(exc)}), 500

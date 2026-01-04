import uuid
import json
import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from ai_court.api import state, config

feedback_bp = Blueprint('feedback', __name__)

AL_QUEUE_FILE = os.path.join(config.PROJECT_ROOT, "data", "al_queue.json")

def load_queue():
    if os.path.exists(AL_QUEUE_FILE):
        try:
            with open(AL_QUEUE_FILE, 'r', encoding='utf-8') as f:
                state.AL_QUEUE = json.load(f)
        except Exception:
            state.AL_QUEUE = []

def save_queue():
    try:
        os.makedirs(os.path.dirname(AL_QUEUE_FILE), exist_ok=True)
        with open(AL_QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state.AL_QUEUE, f, indent=2)
    except Exception:
        pass

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
    item = {
        'id': str(uuid.uuid4()),
        'text': text,
        'added_at': datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
    }
    state.AL_QUEUE.append(item)
    save_queue()
    return jsonify(item), 201

@feedback_bp.route('/api/active_learning/queue/<item_id>', methods=['POST'])
def al_queue_label(item_id: str):
    raw = request.json or {}
    label = raw.get('label') if isinstance(raw, dict) else None
    for i, it in enumerate(state.AL_QUEUE):
        if it['id'] == item_id:
            it['label'] = label
            it['labeled_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
            save_queue()
            return jsonify(it)
    return jsonify({"error": "not_found"}), 404

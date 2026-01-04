import os
import json
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError
import numpy as np

from ai_court.api import state, dependencies, models, constants
from ai_court.api.extensions import limiter

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route("/api/questions", methods=["GET"])
def get_questions():
    return jsonify(constants.CASE_TYPES)

@analysis_bp.route("/api/questions/<case_type>", methods=["GET"])
def get_questions_by_type(case_type):
    if case_type in constants.CASE_TYPES:
        return jsonify(constants.CASE_TYPES[case_type])
    return jsonify({"error": "Case type not found"}), 404

@analysis_bp.route("/api/analyze", methods=["POST"])
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
    auth = dependencies.require_api_key()
    if auth:
        return auth
    raw = request.json
    if raw is None:
        return jsonify({"error": "Expected application/json body"}), 400
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = models.AnalyzeRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    payload_str = json.dumps(raw)
    if len(payload_str) > 100_000:
        return jsonify({"error": "Payload too large"}), 413
    
    try:
        if state.classifier:
            # synthesize_text_from_answers includes case_type at the start
            full_text = dependencies.synthesize_text_from_answers(raw)
            # We pass empty case_type because full_text already has it
            result = state.classifier.predict(full_text, "")
            judgment = result["judgment"]
            confidence = result["confidence"]
            processed = result["processed_text"]
        else:
            raise RuntimeError("Model not initialized")
    except Exception as e:
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500
    try:
        state.PREDICTIONS_TOTAL.labels('analyze', judgment).inc()
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
    if state.multi_axis_bundle:
        try:
            shadow = dependencies.multi_axis_predict_single(processed)
        except Exception:
            shadow = {'error':'shadow_inference_failed'}
    primary_judgment = judgment
    if use_primary_multi and isinstance(shadow, dict):
        primary_judgment = (shadow.get('main_judgment') if isinstance(shadow.get('main_judgment'), str) else None) or judgment
    
    if shadow and isinstance(shadow, dict) and 'axes' in shadow:
        try:
            dependencies.record_agreement(judgment, shadow['axes'])
        except Exception:
            pass
    agreement_rate = None
    if state.agreement_stats['total_compared'] > 0:
        agreement_rate = state.agreement_stats['agreements'] / state.agreement_stats['total_compared']
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

@analysis_bp.route('/api/rag/query', methods=['POST'])
def rag_query():
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    question = raw.get('question') or ''
    if not question:
        return jsonify({"error": "missing_question"}), 400
    docs: list = []
    if state.search_index is not None:
        vect = state.search_index['vectorizer']
        matrix = state.search_index['matrix']
        meta = state.search_index.get('meta', [])
        qv = vect.transform([state.preprocess_fn(question)])
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
    answer = "RAG pipeline not yet fully implemented. Retrieved top documents only."
    return jsonify({
        'question': question,
        'answer': answer,
        'documents': docs,
        'num_documents': len(docs)
    })

@analysis_bp.route("/api/analyze_and_search", methods=["POST"])
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
    auth = dependencies.require_api_key()
    if auth:
        return auth
    raw = request.json or {}
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = models.AnalyzeRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    data = raw
    
    try:
        if state.classifier:
            full_text = dependencies.synthesize_text_from_answers(raw)
            result = state.classifier.predict(full_text, "")
            judgment = result["judgment"]
            confidence = result["confidence"]
            processed = result["processed_text"]
        else:
            raise RuntimeError("Model not initialized")
    except Exception as e:
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500

    try:
        state.PREDICTIONS_TOTAL.labels('analyze_and_search', judgment).inc()
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

    search_results = []
    if state.search_index is not None:
        vect = state.search_index["vectorizer"]
        matrix = state.search_index["matrix"]
        meta = state.search_index.get("meta", [])
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

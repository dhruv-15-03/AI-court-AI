import os
import json
import uuid
import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError
import numpy as np

from ai_court.api import state, dependencies, models, constants, config
from ai_court.api.extensions import limiter

try:
    from ai_court.utils.explainability import extract_top_features, format_explanation
except ImportError:
    extract_top_features = None
    format_explanation = None

logger = logging.getLogger(__name__)
analysis_bp = Blueprint('analysis', __name__)


def _auto_queue_prediction(
    text: str,
    judgment: str,
    confidence: float,
    reason: str,
    raw_input: dict
) -> None:
    """Add low-confidence or uncertain predictions to the active learning queue."""
    if not config.AUTO_QUEUE_LOW_CONFIDENCE:
        return
    
    try:
        item = {
            'id': str(uuid.uuid4()),
            'text': text[:1000],  # Truncate for storage
            'predicted_label': judgment,
            'confidence': confidence,
            'reason': reason,
            'case_type': raw_input.get('case_type'),
            'added_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'source': 'auto_queue'
        }
        state.AL_QUEUE.append(item)
        
        # Persist queue (import the save function)
        from ai_court.api.routes.feedback import save_queue
        save_queue()
        
        logger.info(f"Auto-queued prediction for review: {reason}, confidence={confidence:.2f}")
    except Exception as e:
        logger.warning(f"Failed to auto-queue prediction: {e}")

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
            # Clean Logic: Pass components separately as expected by the model
            case_type = parsed.case_type or ""
            # Use body only, let predict handle concatenation
            case_body = dependencies.synthesize_body_from_answers(raw)
            
            result = state.classifier.predict(case_body, case_type)
            judgment = result["judgment"]
            confidence = result["confidence"]
            processed = result["processed_text"]
            pred_idx = None
            
            # Get prediction index for explainability
            if hasattr(state.classifier, 'label_encoder') and state.classifier.label_encoder:
                try:
                    pred_idx = int(state.classifier.label_encoder.transform([judgment])[0])
                except Exception:
                    pred_idx = None
        else:
            raise RuntimeError("Model not initialized")
    except Exception as e:
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500
    
    # Confidence thresholding and abstention
    needs_review = False
    abstention_reason = None
    if confidence is not None and confidence < config.CONFIDENCE_THRESHOLD:
        needs_review = True
        abstention_reason = f"Low confidence ({confidence:.2f} < {config.CONFIDENCE_THRESHOLD})"
        # Auto-queue for active learning
        _auto_queue_prediction(
            text=processed,
            judgment=judgment,
            confidence=confidence,
            reason=abstention_reason,
            raw_input=raw
        )
    
    # Extract explainability features
    key_factors = []
    explanation = None
    if extract_top_features and state.classifier and state.classifier.model and pred_idx is not None:
        try:
            key_factors = extract_top_features(
                state.classifier.model,
                processed,
                pred_idx,
                top_k=config.EXPLAIN_TOP_K
            )
            if format_explanation and key_factors:
                explanation = format_explanation(key_factors, judgment, confidence)
        except Exception as e:
            logger.debug(f"Explainability extraction failed: {e}")
    
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
    
    response_data = {
        "case_type": inferred_case_type,
        "judgment": primary_judgment,
        "judgment_classical": judgment,
        "judgment_source": 'multi_axis' if use_primary_multi and shadow else 'classical',
        "answers": data,
        "confidence": confidence,
        "needs_review": needs_review,
        "shadow_multi_axis": shadow,
        "agreement_rate": agreement_rate,
    }
    
    # Add explainability if available
    if key_factors:
        response_data["key_factors"] = key_factors
    if explanation:
        response_data["explanation"] = explanation
    if abstention_reason:
        response_data["abstention_reason"] = abstention_reason
    
    return jsonify(response_data)

@analysis_bp.route('/api/rag/query', methods=['POST'])
@limiter.limit("30/minute")
def rag_query_endpoint():
    """RAG-style query endpoint using retrieval-only mode (no LLM, zero cost)."""
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    question = raw.get('question') or ''
    if not question:
        return jsonify({"error": "missing_question"}), 400
    
    k = min(raw.get('k', 5), config.MAX_SEARCH_RESULTS)
    
    # Use the new RAG pipeline
    try:
        from ai_court.rag.pipeline import rag_query as rag_pipeline
        
        response = rag_pipeline(
            question=question,
            search_index=state.search_index,
            preprocess_fn=state.preprocess_fn,
            k=k
        )
        return jsonify(response)
        
    except ImportError:
        # Fallback to legacy implementation
        docs: list = []
        if state.search_index is not None:
            vect = state.search_index['vectorizer']
            matrix = state.search_index['matrix']
            meta = state.search_index.get('meta', [])
            qv = vect.transform([state.preprocess_fn(question)])
            scores = (matrix @ qv.T).toarray().ravel()
            top_idx = np.argsort(-scores)[:k]
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
        
        return jsonify({
            'question': question,
            'answer': "Retrieved relevant cases. See documents for details.",
            'documents': docs,
            'num_documents': len(docs),
            'mode': 'retrieval_only'
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

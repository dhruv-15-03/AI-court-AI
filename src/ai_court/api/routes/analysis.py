import os
import json
import uuid
import logging
import time
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError
import numpy as np

from ai_court.api import state, dependencies, models, constants, config
from ai_court.api.extensions import limiter
from typing import Any, Callable, List, Dict, Optional

# Type aliases for explainability functions
ExtractTopFeaturesType = Callable[[Any, str, int, int], List[Dict[str, Any]]]
FormatExplanationType = Callable[[List[Dict[str, Any]], str, Optional[float]], str]

extract_top_features: Optional[ExtractTopFeaturesType] = None
format_explanation: Optional[FormatExplanationType] = None

try:
    from ai_court.utils.explainability import extract_top_features as _extract, format_explanation as _format
    extract_top_features = _extract
    format_explanation = _format
except ImportError:
    pass

# Import performance utilities
try:
    from ai_court.utils.performance import (
        get_confidence_language,
        get_outcome_description,
        format_minimal_response,
        format_full_response,
        format_detailed_response
    )
except ImportError:
    get_confidence_language = None  # type: ignore[assignment]
    get_outcome_description = None  # type: ignore[assignment]
    format_minimal_response = None  # type: ignore[assignment]
    format_full_response = None  # type: ignore[assignment]
    format_detailed_response = None  # type: ignore[assignment]

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


# =============================================================================
# Response Generation Helpers
# =============================================================================

def _generate_basic_explanation(judgment: str, confidence: Optional[float]) -> str:
    """Generate a basic explanation when key factors are unavailable."""
    conf_str = f"{int((confidence or 0) * 100)}%" if confidence else "unknown"
    
    explanations = {
        "Bail Granted": f"Based on the case details, bail appears likely to be granted. The model confidence is {conf_str}.",
        "Bail Denied": f"Based on the case factors, bail may be denied. Review the evidence and prior record. Confidence: {conf_str}.",
        "Conviction": f"The evidence patterns suggest a likely conviction. This prediction has {conf_str} confidence.",
        "Acquittal": f"The case factors indicate potential acquittal. Model confidence: {conf_str}.",
        "Charges Quashed": f"Based on procedural or evidentiary factors, charges may be quashed. Confidence: {conf_str}.",
        "Compensation Awarded": f"The case merits suggest compensation will be awarded. Confidence: {conf_str}.",
        "Petition Dismissed": f"The petition appears likely to be dismissed based on presented factors. Confidence: {conf_str}.",
        "Injunction Granted": f"An injunction is likely based on the urgency and merits. Confidence: {conf_str}.",
        "Maintenance Ordered": f"Maintenance order appears likely given the family circumstances. Confidence: {conf_str}.",
    }
    
    return explanations.get(judgment, f"Predicted outcome: {judgment} with {conf_str} confidence.")


def _get_similar_cases(
    processed_text: str,
    k: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve similar cases using the search index."""
    similar_cases = []
    
    try:
        if state.search_index is not None:
            vect = state.search_index['vectorizer']
            matrix = state.search_index['matrix']
            meta = state.search_index.get('meta', [])
            
            qv = vect.transform([processed_text])
            scores = (matrix @ qv.T).toarray().ravel()
            top_idx = np.argsort(-scores)[:k]
            
            for idx in top_idx:
                if idx < len(meta) and scores[idx] > 0.1:  # Minimum relevance threshold
                    m = meta[idx]
                    similar_cases.append({
                        "case_id": m.get('case_id', f"CASE_{idx}"),
                        "title": m.get('title', 'Unknown Case'),
                        "similarity_score": round(float(scores[idx]), 3),
                        "outcome": m.get('outcome', m.get('judgment', 'Unknown')),
                        "court": m.get('court', 'Unknown Court'),
                        "year": m.get('year', None),
                        "summary": m.get('snippet', m.get('summary', ''))[:300],
                        "url": m.get('url'),
                    })
    except Exception as e:
        logger.debug(f"Similar cases retrieval failed: {e}")
    
    return similar_cases


def _generate_legal_analysis(
    judgment: str,
    case_type: str,
    key_factors: List[Dict[str, Any]],
    confidence: Optional[float]
) -> Dict[str, Any]:
    """Generate legal analysis section for UNLIMITED plan."""
    
    # Applicable sections based on case type
    sections_map = {
        "Criminal": [
            {"section": "Section 437 CrPC", "relevance": "Bail in non-bailable offenses", "interpretation": "Courts consider flight risk, severity, and evidence"},
            {"section": "Section 439 CrPC", "relevance": "Special powers of High Court/Sessions Court", "interpretation": "Wider discretion for bail"},
        ],
        "Civil": [
            {"section": "Order 39 CPC", "relevance": "Temporary injunctions", "interpretation": "Prima facie case, balance of convenience, irreparable injury"},
            {"section": "Section 9 CPC", "relevance": "Civil court jurisdiction", "interpretation": "Subject matter jurisdiction analysis"},
        ],
        "Labor": [
            {"section": "Section 25F Industrial Disputes Act", "relevance": "Conditions for retrenchment", "interpretation": "Notice and compensation requirements"},
            {"section": "Section 33C(2) ID Act", "relevance": "Recovery of money due", "interpretation": "Calculation and execution of dues"},
        ],
        "Family": [
            {"section": "Section 125 CrPC", "relevance": "Maintenance of wives, children, parents", "interpretation": "Inability to maintain, reasonable provision"},
            {"section": "Section 13 Hindu Marriage Act", "relevance": "Grounds for divorce", "interpretation": "Cruelty, desertion, adultery analysis"},
        ],
    }
    
    # Determine precedent strength from confidence
    if confidence and confidence >= 0.85:
        precedent_strength = "STRONG"
    elif confidence and confidence >= 0.65:
        precedent_strength = "MODERATE"
    else:
        precedent_strength = "WEAK"
    
    # Extract risk and mitigating factors from key_factors
    risk_factors = [f.get('feature', '') for f in key_factors if f.get('direction') == 'negative'][:3]
    mitigating_factors = [f.get('feature', '') for f in key_factors if f.get('direction') == 'positive'][:3]
    
    return {
        "applicable_sections": sections_map.get(case_type, sections_map["Civil"]),
        "precedent_strength": precedent_strength,
        "jurisdiction_notes": f"Analysis based on {case_type} law precedents",
        "risk_factors": risk_factors if risk_factors else ["No significant risk factors identified"],
        "mitigating_factors": mitigating_factors if mitigating_factors else ["Standard case factors apply"],
    }


def _generate_full_report(
    judgment: str,
    case_type: str,
    key_factors: List[Dict[str, Any]],
    confidence: Optional[float],
    explanation: Optional[str]
) -> Dict[str, Any]:
    """Generate full report section for UNLIMITED plan."""
    
    conf_pct = int((confidence or 0) * 100)
    
    # Generate argument points from key factors
    args_for = [
        f"{f.get('feature', 'Factor')}: {f.get('description', 'Supports outcome')}"
        for f in key_factors if f.get('direction') == 'positive'
    ][:5]
    
    args_against = [
        f"{f.get('feature', 'Factor')}: {f.get('description', 'May affect outcome')}"
        for f in key_factors if f.get('direction') == 'negative'
    ][:5]
    
    # Alternative outcomes with estimated probabilities
    alternative_outcomes = _get_alternative_outcomes(judgment, confidence)
    
    return {
        "executive_summary": f"Based on analysis of the submitted {case_type} case, the predicted outcome "
                            f"is **{judgment}** with {conf_pct}% confidence. "
                            f"{explanation or 'Review key factors for detailed reasoning.'}",
        "detailed_analysis": f"## Case Analysis\n\n"
                            f"**Predicted Outcome:** {judgment}\n\n"
                            f"**Confidence Level:** {conf_pct}%\n\n"
                            f"**Case Type:** {case_type}\n\n"
                            f"### Key Factors\n\n"
                            + "\n".join([f"- {f.get('feature', 'N/A')}: {f.get('description', '')}" for f in key_factors[:5]]),
        "argument_points_for": args_for if args_for else ["Case facts support the predicted outcome"],
        "argument_points_against": args_against if args_against else ["No significant counter-arguments identified"],
        "recommended_strategy": _get_strategy_recommendation(judgment, case_type),
        "estimated_timeline": _get_timeline_estimate(case_type),
        "alternative_outcomes": alternative_outcomes,
    }


def _get_alternative_outcomes(judgment: str, confidence: Optional[float]) -> List[Dict[str, Any]]:
    """Generate alternative outcome possibilities."""
    conf = confidence or 0.5
    remaining = 1 - conf
    
    alternatives_map = {
        "Bail Granted": [
            {"outcome": "Conditional Bail", "probability": round(remaining * 0.6, 2), "conditions": "Surety, travel restrictions, regular reporting"},
            {"outcome": "Bail Denied", "probability": round(remaining * 0.4, 2), "conditions": "Flight risk or evidence tampering concerns"},
        ],
        "Bail Denied": [
            {"outcome": "Conditional Bail", "probability": round(remaining * 0.5, 2), "conditions": "With strict conditions on appeal"},
            {"outcome": "Bail Granted", "probability": round(remaining * 0.5, 2), "conditions": "If additional evidence or circumstances presented"},
        ],
        "Conviction": [
            {"outcome": "Reduced Sentence", "probability": round(remaining * 0.5, 2), "conditions": "Mitigating factors considered"},
            {"outcome": "Acquittal on Appeal", "probability": round(remaining * 0.5, 2), "conditions": "If evidence challenged successfully"},
        ],
        "Acquittal": [
            {"outcome": "Conviction on Appeal", "probability": round(remaining * 0.6, 2), "conditions": "If prosecution appeals successfully"},
            {"outcome": "Partial Conviction", "probability": round(remaining * 0.4, 2), "conditions": "Lesser charges may apply"},
        ],
    }
    
    return alternatives_map.get(judgment, [
        {"outcome": "Appeal/Review", "probability": round(remaining, 2), "conditions": "Standard appellate process"}
    ])


def _get_strategy_recommendation(judgment: str, case_type: str) -> str:
    """Get recommended legal strategy."""
    strategies = {
        "Bail Granted": "Ensure compliance with all bail conditions. Prepare for trial proceedings.",
        "Bail Denied": "Consider appeal to higher court. Strengthen grounds for bail application.",
        "Conviction": "Evaluate grounds for appeal. Consider sentencing mitigation arguments.",
        "Acquittal": "Monitor for prosecution appeal. Maintain documentation of proceedings.",
        "Charges Quashed": "Document the quashing order. Monitor for any new proceedings.",
        "Compensation Awarded": "Pursue execution of decree. Consider enhancement if applicable.",
        "Petition Dismissed": "Analyze grounds for rejection. Consider appeal or fresh petition.",
        "Injunction Granted": "Ensure compliance by opposite party. Prepare for main suit.",
        "Maintenance Ordered": "Set up payment mechanism. Document compliance.",
    }
    return strategies.get(judgment, "Consult with legal counsel for case-specific strategy.")


def _get_timeline_estimate(case_type: str) -> str:
    """Estimate typical timeline for case type."""
    timelines = {
        "Criminal": "6-18 months (trial), 1-3 years (with appeals)",
        "Civil": "2-5 years (typical civil suit)",
        "Labor": "1-3 years (tribunal proceedings)",
        "Family": "6 months - 3 years (depending on complexity)",
    }
    return timelines.get(case_type, "Variable based on court and case complexity")


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
    """Analyze a legal case and predict outcome."""
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
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
    
    # Get plan from request (default to 'basic' for backwards compatibility)
    plan = raw.get('_plan', 'basic').lower()  # free, basic, pro, unlimited
    
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
    key_factors: List[Dict[str, Any]] = []
    explanation: Optional[str] = None
    if extract_top_features is not None and state.classifier and state.classifier.model and pred_idx is not None:
        try:
            key_factors = extract_top_features(
                state.classifier.model,
                processed,
                pred_idx,
                top_k=config.EXPLAIN_TOP_K
            )
            if format_explanation is not None and key_factors:
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
    
    # Determine judgment source
    judgment_source = 'ML_MODEL'
    if use_primary_multi and shadow:
        judgment_source = 'HYBRID'
    
    # Calculate processing time
    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    # ==================== BUILD RESPONSE ====================
    # Core fields (All Plans)
    response_data: Dict[str, Any] = {
        "judgment": primary_judgment,
        "confidence": round(confidence or 0, 4),
        "case_type": inferred_case_type,
        "needs_review": needs_review,
        "abstention_reason": abstention_reason,
    }
    
    # Basic Analysis (FREE + BASIC Plans)
    if plan in ('free', 'basic', 'pro', 'unlimited'):
        response_data["explanation"] = explanation or _generate_basic_explanation(primary_judgment, confidence)
        response_data["judgment_source"] = judgment_source
    
    # Key Factors (PRO+ Plans)
    if plan in ('pro', 'unlimited') and key_factors:
        # Remove internal _raw field for API response
        response_data["key_factors"] = [
            {k: v for k, v in factor.items() if not k.startswith('_')}
            for factor in key_factors
        ]
    
    # Similar Cases (PRO+ Plans)
    if plan in ('pro', 'unlimited') and config.INCLUDE_SIMILAR_CASES:
        similar_cases = _get_similar_cases(processed, config.SIMILAR_CASES_COUNT)
        if similar_cases:
            response_data["similar_cases"] = similar_cases
    
    # Legal Analysis (UNLIMITED Plan)
    if plan == 'unlimited':
        response_data["legal_analysis"] = _generate_legal_analysis(
            judgment=primary_judgment,
            case_type=inferred_case_type,
            key_factors=key_factors,
            confidence=confidence
        )
    
    # Full Report (UNLIMITED Plan)
    if plan == 'unlimited':
        response_data["full_report"] = _generate_full_report(
            judgment=primary_judgment,
            case_type=inferred_case_type,
            key_factors=key_factors,
            confidence=confidence,
            explanation=explanation
        )
    
    # Metadata (Always returned)
    response_data["metadata"] = {
        "model_version": config.APP_VERSION,
        "request_id": request_id,
        "processed_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "processing_time_ms": processing_time_ms,
        "rag_sources_used": len(response_data.get("similar_cases", [])),
        "tokens_used": len(processed.split()) if processed else 0,
    }
    
    # Input Echo (For DB storage)
    response_data["answers"] = data
    
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


# =============================================================================
# Batch Prediction Endpoint (P1 Optimization)
# =============================================================================

@analysis_bp.route("/api/analyze/batch", methods=["POST"])
@limiter.limit("10/minute")
@swag_from({
    'tags': ['analyze'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'cases': {
                    'type': 'array',
                    'items': {'type': 'object'},
                    'description': 'Array of case objects to analyze'
                },
                'format': {
                    'type': 'string',
                    'enum': ['minimal', 'full', 'detailed'],
                    'description': 'Response format level'
                }
            },
            'required': ['cases']
        }
    }],
    'responses': {
        200: {'description': 'Batch analysis results'},
        400: {'description': 'Invalid request'},
        413: {'description': 'Too many cases in batch'}
    }
})
def analyze_batch():
    """
    Analyze multiple cases in a single request.
    
    More efficient than making individual requests for bulk processing.
    Returns predictions for all cases with timing information.
    """
    auth = dependencies.require_api_key()
    if auth:
        return auth
    
    raw = request.json
    if raw is None or not isinstance(raw, dict):
        return jsonify({"error": "Expected JSON object with 'cases' array"}), 400
    
    cases = raw.get('cases', [])
    if not isinstance(cases, list):
        return jsonify({"error": "'cases' must be an array"}), 400
    
    if len(cases) > config.BATCH_SIZE_LIMIT:
        return jsonify({
            "error": f"Batch size exceeds limit of {config.BATCH_SIZE_LIMIT}",
            "submitted": len(cases),
            "limit": config.BATCH_SIZE_LIMIT
        }), 413
    
    if len(cases) == 0:
        return jsonify({
            "results": [],
            "count": 0,
            "timing_ms": 0
        })
    
    response_format = raw.get('format', 'minimal')
    if response_format not in ('minimal', 'full', 'detailed'):
        response_format = 'minimal'
    
    start_time = time.perf_counter()
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append({"index": idx, "error": "Case must be a JSON object"})
            continue
        
        try:
            # Validate input
            parsed = models.AnalyzeRequest(**case)
            case_type = parsed.case_type or ""
            case_body = dependencies.synthesize_body_from_answers(case)
            
            if state.classifier:
                result = state.classifier.predict(case_body, case_type)
                judgment = result["judgment"]
                confidence = result["confidence"]
                
                # Infer case type if not provided
                inferred_type = case_type or _infer_case_type(case)
                
                # Build response based on format
                if response_format == 'minimal':
                    results.append({
                        "index": idx,
                        "judgment": judgment,
                        "confidence": round(confidence or 0, 3),
                        "case_type": inferred_type
                    })
                elif response_format == 'full' and get_confidence_language is not None:
                    results.append({
                        "index": idx,
                        "judgment": judgment,
                        "confidence": round(confidence or 0, 3),
                        "case_type": inferred_type,
                        "confidence_info": get_confidence_language(confidence),
                        "needs_review": (confidence or 0) < config.CONFIDENCE_THRESHOLD
                    })
                else:  # detailed or fallback
                    results.append({
                        "index": idx,
                        "judgment": judgment,
                        "confidence": round(confidence or 0, 3),
                        "case_type": inferred_type,
                        "confidence_info": get_confidence_language(confidence) if get_confidence_language is not None else None,
                        "needs_review": (confidence or 0) < config.CONFIDENCE_THRESHOLD,
                        "answers": case
                    })
            else:
                errors.append({"index": idx, "error": "Model not initialized"})
                
        except ValidationError as ve:
            errors.append({"index": idx, "error": "validation_failed", "details": ve.errors()})
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Update metrics
    try:
        for r in results:
            state.PREDICTIONS_TOTAL.labels('batch', r.get('judgment', 'Unknown')).inc()
    except Exception:
        pass
    
    return jsonify({
        "results": results,
        "errors": errors,
        "count": len(results),
        "error_count": len(errors),
        "timing_ms": round(elapsed_ms, 2),
        "avg_ms_per_case": round(elapsed_ms / len(cases), 2) if cases else 0
    })


def _infer_case_type(raw: Dict[str, Any]) -> str:
    """Infer case type from input fields."""
    if raw.get("violence_level", "None") != "None" or raw.get("police_report") == "Yes":
        return "Criminal"
    elif raw.get("employment_duration"):
        return "Labor"
    elif raw.get("children") or raw.get("marriage_duration"):
        return "Family"
    else:
        return "Civil"


# =============================================================================
# Quick Analyze Endpoint (Minimal Response)
# =============================================================================

@analysis_bp.route("/api/analyze/quick", methods=["POST"])
@limiter.limit("60/minute")
@swag_from({
    'tags': ['analyze'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'Case text to analyze'},
                'case_type': {'type': 'string', 'description': 'Optional case type hint'}
            },
            'required': ['text']
        }
    }],
    'responses': {200: {'description': 'Quick prediction result'}}
})
def analyze_quick():
    """
    Fast prediction endpoint with minimal processing.
    
    Use this for high-throughput scenarios where speed is critical.
    Returns only: judgment, confidence, case_type.
    """
    auth = dependencies.require_api_key()
    if auth:
        return auth
    
    raw = request.json
    if raw is None or not isinstance(raw, dict):
        return jsonify({"error": "Expected JSON body"}), 400
    
    text = raw.get('text', '')
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    case_type = raw.get('case_type', '')
    
    start_time = time.perf_counter()
    
    try:
        if state.classifier:
            result = state.classifier.predict(text, case_type)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            return jsonify({
                "judgment": result["judgment"],
                "confidence": round(result["confidence"] or 0, 3),
                "case_type": case_type or "Unknown",
                "timing_ms": round(elapsed_ms, 2)
            })
        else:
            return jsonify({"error": "Model not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Outcome Info Endpoint
# =============================================================================

@analysis_bp.route("/api/outcomes", methods=["GET"])
def get_outcomes():
    """Get list of all possible outcomes with descriptions."""
    if get_outcome_description is not None:
        from ai_court.utils.performance import OUTCOME_DESCRIPTIONS
        return jsonify({
            "outcomes": list(OUTCOME_DESCRIPTIONS.keys()),
            "descriptions": OUTCOME_DESCRIPTIONS
        })
    else:
        # Fallback: return just the outcome labels
        if state.classifier and hasattr(state.classifier, 'label_encoder'):
            labels = list(state.classifier.label_encoder.classes_)
            return jsonify({"outcomes": labels})
        return jsonify({"outcomes": []})


@analysis_bp.route("/api/outcomes/<outcome>", methods=["GET"])
def get_outcome_info(outcome: str):
    """Get detailed information about a specific outcome."""
    if get_outcome_description is not None:
        info = get_outcome_description(outcome)
        return jsonify({
            "outcome": outcome,
            **info
        })
    return jsonify({"outcome": outcome, "meaning": "Information not available"})

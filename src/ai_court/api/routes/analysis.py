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
        OUTCOME_DESCRIPTIONS,
        format_minimal_response,
        format_full_response,
        format_detailed_response
    )
except ImportError:
    get_confidence_language = None  # type: ignore[assignment]
    get_outcome_description = None  # type: ignore[assignment]
    OUTCOME_DESCRIPTIONS = {}  # type: ignore[assignment]
    format_minimal_response = None  # type: ignore[assignment]
    format_full_response = None  # type: ignore[assignment]
    format_detailed_response = None  # type: ignore[assignment]

# Import extractive summary utilities
try:
    from ai_court.scraper.extractive_summary import (
        get_outcome_indicators,
        extract_key_holdings,
        extract_citations,
        extract_parties,
    )
except ImportError:
    get_outcome_indicators = None  # type: ignore[assignment]
    extract_key_holdings = None  # type: ignore[assignment]
    extract_citations = None  # type: ignore[assignment]
    extract_parties = None  # type: ignore[assignment]

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
        # Compute entropy-based uncertainty if probabilities available
        probabilities = raw_input.get('_all_probabilities', {})
        uncertainty = 1.0 - confidence  # fallback
        uncertainty_method = 'confidence'
        if probabilities:
            from ai_court.active_learning.loop import compute_uncertainty
            uncertainty = compute_uncertainty(probabilities, method='entropy')
            uncertainty_method = 'entropy'

        item = {
            'id': str(uuid.uuid4()),
            'text': text[:2000],  # Truncate for storage
            'predicted_label': judgment,
            'confidence': confidence,
            'uncertainty': round(uncertainty, 4),
            'uncertainty_method': uncertainty_method,
            'probabilities': probabilities,
            'reason': reason,
            'case_type': raw_input.get('case_type'),
            'added_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'source': 'auto_queue'
        }
        state.AL_QUEUE.append(item)
        
        # Persist queue (import the save function)
        from ai_court.api.routes.feedback import save_queue
        save_queue()
        
        logger.info(f"Auto-queued prediction for review: {reason}, uncertainty={uncertainty:.3f}")
    except Exception as e:
        logger.warning(f"Failed to auto-queue prediction: {e}")


# =============================================================================
# Response Generation Helpers
# =============================================================================


def _generate_case_summary(
    raw_input: Dict[str, Any],
    case_type: str,
    judgment: str,
    confidence: Optional[float],
    key_factors: List[Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a brief structured analysis summary from the submitted case facts.

    Returned in PRO+ plans alongside the model prediction so the user sees a
    concise narrative of *what was submitted* plus how the model interpreted it.
    """
    # --- 1. Narrative summary of submitted facts ---
    parts: List[str] = []
    ct = case_type or raw_input.get('case_type') or 'Unknown'
    parties = raw_input.get('parties', '')
    if parties:
        parts.append(f"{ct} matter involving {parties}.")
    else:
        parts.append(f"{ct} case submitted for analysis.")

    summary_text = raw_input.get('summary')
    if summary_text:
        # Take first 300 chars of the user-supplied summary
        snippet = str(summary_text).strip()
        if len(snippet) > 300:
            snippet = snippet[:297] + '...'
        parts.append(snippet)

    # Pick up additional contextual fields
    field_labels = {
        'relief_requested': 'Relief sought',
        'sections': 'Relevant sections',
        'evidence_type': 'Evidence type',
        'injury_severity': 'Injury severity',
        'weapon_used': 'Weapon involved',
        'dispute_type': 'Dispute type',
        'violence_level': 'Violence level',
        'police_report': 'Police report filed',
        'witnesses': 'Witnesses',
        'mitigating_factors': 'Mitigating factors',
        'children': 'Children involved',
        'marriage_duration': 'Marriage duration',
        'employment_duration': 'Employment duration',
    }
    detail_parts: List[str] = []
    for field, label in field_labels.items():
        val = raw_input.get(field)
        if val and str(val).strip() and str(val).strip().lower() not in ('none', 'unknown', 'n/a', ''):
            detail_parts.append(f"{label}: {val}")
    if detail_parts:
        parts.append(' | '.join(detail_parts) + '.')

    narrative = ' '.join(parts)

    # --- 2. Key factors digest (top-3 one-liners) ---
    factors_digest: List[str] = []
    for f in key_factors[:3]:
        direction_icon = '+' if f.get('direction') == 'positive' else '-'
        factors_digest.append(
            f"[{direction_icon}] {f.get('feature', 'Unknown')}: {f.get('description', '')}"
        )

    # --- 3. Precedent snapshot ---
    precedent_snapshot: Optional[Dict[str, Any]] = None
    if similar_cases:
        outcomes_in_similar = [sc.get('outcome') for sc in similar_cases if sc.get('outcome')]
        matching = sum(1 for o in outcomes_in_similar if o == judgment)
        precedent_snapshot = {
            'cases_reviewed': len(similar_cases),
            'matching_outcome': matching,
            'top_match': {
                'title': similar_cases[0].get('title'),
                'similarity': similar_cases[0].get('similarity_score'),
                'outcome': similar_cases[0].get('outcome'),
            } if similar_cases else None,
        }

    conf_pct = int((confidence or 0) * 100)
    return {
        'narrative': narrative,
        'predicted_outcome': judgment,
        'confidence_pct': conf_pct,
        'key_factors_digest': factors_digest,
        'precedent_snapshot': precedent_snapshot,
    }


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
    # Ensure lazy-loaded search index is available
    dependencies.ensure_search_index()
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
    """Generate legal analysis section for UNLIMITED plan.
    
    Uses LLM when available, falls back to template-based analysis.
    """
    # Try LLM-powered analysis first
    if state.llm_client is not None:
        try:
            factors_text = "\n".join(
                f"- {f.get('feature', 'Unknown')} ({f.get('direction', 'neutral')}): {f.get('description', '')}"
                for f in key_factors[:5]
            )
            prompt = (
                f"Provide a concise legal analysis for an Indian {case_type} case.\n\n"
                f"Predicted outcome: {judgment} (confidence: {int((confidence or 0) * 100)}%)\n\n"
                f"Key factors:\n{factors_text}\n\n"
                f"Return a JSON object with these keys:\n"
                f'- "applicable_sections": list of {{"section": "...", "relevance": "...", "interpretation": "..."}}\n'
                f'- "precedent_strength": "STRONG"|"MODERATE"|"WEAK"\n'
                f'- "jurisdiction_notes": string\n'
                f'- "risk_factors": list of strings\n'
                f'- "mitigating_factors": list of strings\n\n'
                f"Return ONLY valid JSON, no markdown fences."
            )
            from ai_court.llm.prompts import SYSTEM_PROMPT_LEGAL_AGENT
            raw = state.llm_client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
                 {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=2048,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            import json as _json
            return _json.loads(raw.strip())
        except Exception as exc:
            logger.warning("LLM legal analysis failed, using template: %s", exc)
    
    # Fallback: template-based analysis
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
    
    if confidence and confidence >= 0.85:
        precedent_strength = "STRONG"
    elif confidence and confidence >= 0.65:
        precedent_strength = "MODERATE"
    else:
        precedent_strength = "WEAK"
    
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
    """Generate full report section for UNLIMITED plan.

    Uses LLM when available for a comprehensive court-ready report.
    Falls back to template-based generation otherwise.
    """
    conf_pct = int((confidence or 0) * 100)

    # --- LLM-powered full report ---
    if state.llm_client is not None:
        try:
            factors_text = "\n".join(
                f"- {f.get('feature', 'Unknown')} ({f.get('direction', 'neutral')}): "
                f"{f.get('description', '')}"
                for f in key_factors[:8]
            )
            prompt = (
                f"You are an expert Indian legal analyst preparing a comprehensive court-ready "
                f"report for a {case_type} case.\n\n"
                f"Predicted outcome: {judgment} (confidence: {conf_pct}%)\n"
                f"Explanation: {explanation or 'N/A'}\n\n"
                f"Key factors from ML model:\n{factors_text}\n\n"
                f"Generate a JSON report with these exact keys:\n"
                f'- "executive_summary": 2-3 sentence professional summary of the case assessment\n'
                f'- "detailed_analysis": Markdown-formatted detailed analysis covering: '
                f'case strength, evidence considerations, procedural aspects, and judicial tendencies\n'
                f'- "argument_points_for": list of 3-5 strong arguments supporting the client\'s position, '
                f'each citing specific Indian law sections\n'
                f'- "argument_points_against": list of 2-4 arguments the opposing side may raise\n'
                f'- "recommended_strategy": detailed strategy paragraph with specific next steps\n'
                f'- "estimated_timeline": realistic timeline for this type of case in Indian courts\n'
                f'- "alternative_outcomes": list of {{"outcome": "...", "probability": 0.X, '
                f'"conditions": "when this could happen"}}\n\n'
                f"Return ONLY valid JSON. No markdown fences."
            )
            from ai_court.llm.prompts import SYSTEM_PROMPT_LEGAL_AGENT
            raw = state.llm_client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
                 {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=3000,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            import json as _json
            return _json.loads(raw.strip())
        except Exception as exc:
            logger.warning("LLM full report failed, using template: %s", exc)

    # --- Template fallback ---
    args_for = [
        f"{f.get('feature', 'Factor')}: {f.get('description', 'Supports outcome')}"
        for f in key_factors if f.get('direction') == 'positive'
    ][:5]

    args_against = [
        f"{f.get('feature', 'Factor')}: {f.get('description', 'May affect outcome')}"
        for f in key_factors if f.get('direction') == 'negative'
    ][:5]

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


def _get_risk_level(judgment: str, confidence: Optional[float]) -> str:
    """Derive risk level from outcome class and confidence."""
    conf = confidence or 0.0
    high_risk_outcomes = {"Acquittal/Conviction Overturned", "Bail Denied"}
    if conf < 0.60 or judgment in high_risk_outcomes:
        return "HIGH"
    elif conf < 0.80:
        return "MEDIUM"
    return "LOW"


def _get_confidence_band(confidence: Optional[float]) -> str:
    """Convert raw confidence float to a readable band label."""
    conf = (confidence or 0.0) * 100
    if conf >= 85:
        return "Very High (≥85%)"
    elif conf >= 70:
        return "High (70–84%)"
    elif conf >= 50:
        return "Moderate (50–69%)"
    return "Low (<50%)"


def _get_outcome_description(judgment: str) -> str:
    """Plain-English legal meaning of each prediction class."""
    descriptions = {
        "Relief Granted/Convicted": (
            "The court is likely to rule in favour of the petitioner / convict the accused. "
            "In civil matters this means the relief sought (injunction, compensation, etc.) is likely granted. "
            "In criminal matters it indicates a likely conviction."
        ),
        "Relief Denied/Dismissed": (
            "The court is likely to dismiss the petition, appeal or application. "
            "The opposing party's position is upheld and the relief sought is denied."
        ),
        "Acquittal/Conviction Overturned": (
            "The accused is likely to be acquitted, or an existing conviction is likely to be set aside on appeal. "
            "This is the least common outcome class and carries the highest uncertainty."
        ),
        "Bail Granted": (
            "The court is likely to grant bail. Conditions such as surety, travel restrictions, "
            "or regular reporting to police may be imposed."
        ),
        "Bail Denied": (
            "Bail is likely to be refused. Grounds may include flight risk, evidence tampering concerns, "
            "or severity of the alleged offence."
        ),
    }
    return descriptions.get(
        judgment,
        f"Predicted outcome: {judgment}. Consult legal counsel for case-specific interpretation."
    )


_APPLICABLE_STATUTES: Dict[str, List[Dict[str, str]]] = {
    "Criminal": [
        {"section": "Section 302 IPC", "subject": "Punishment for murder", "note": "Life imprisonment or death penalty"},
        {"section": "Section 307 IPC", "subject": "Attempt to murder", "note": "Up to 10 years or life if hurt caused"},
        {"section": "Section 376 IPC", "subject": "Punishment for rape", "note": "Minimum 7 years, up to life"},
        {"section": "Section 437 CrPC", "subject": "Bail in non-bailable offences", "note": "Sessions/Magistrate court bail powers"},
        {"section": "Section 439 CrPC", "subject": "Special bail powers of High Court/Sessions Court", "note": "Wider discretion than §437"},
        {"section": "Section 173 CrPC", "subject": "Police report / charge-sheet", "note": "Filed after completion of investigation"},
    ],
    "Civil": [
        {"section": "Order 39 Rule 1 & 2 CPC", "subject": "Temporary injunction", "note": "Prima facie case + balance of convenience + irreparable injury"},
        {"section": "Section 9 CPC", "subject": "Civil court jurisdiction", "note": "Subject-matter jurisdiction analysis"},
        {"section": "Section 34 CPC", "subject": "Interest on decree", "note": "Pre-suit, pendente lite, and post-decree interest"},
        {"section": "Section 89 CPC", "subject": "Alternative dispute resolution", "note": "Reference to mediation/arbitration"},
    ],
    "Family": [
        {"section": "Section 125 CrPC", "subject": "Maintenance of wives, children, parents", "note": "Inability to maintain; reasonable provision"},
        {"section": "Section 13 Hindu Marriage Act 1955", "subject": "Grounds for divorce", "note": "Cruelty, desertion, adultery, unsound mind"},
        {"section": "Section 6 Hindu Minority and Guardianship Act", "subject": "Custody of child below 5", "note": "Natural guardian is father; mother for child below 5"},
        {"section": "Section 24 Hindu Marriage Act", "subject": "Maintenance pendente lite", "note": "Interim maintenance during divorce proceedings"},
    ],
    "Labor": [
        {"section": "Section 25F Industrial Disputes Act 1947", "subject": "Conditions precedent to retrenchment", "note": "Notice + compensation requirements"},
        {"section": "Section 33C(2) ID Act", "subject": "Recovery of money due from employer", "note": "Calculation and execution of dues"},
        {"section": "Section 10 ID Act", "subject": "Reference of disputes to tribunals", "note": "State/Central Government reference powers"},
        {"section": "Article 21 Constitution", "subject": "Right to livelihood", "note": "Wrongful termination may engage fundamental rights"},
    ],
    "Property": [
        {"section": "Section 54 Transfer of Property Act", "subject": "Sale of immoveable property", "note": "Essential elements: parties, price, conveyance"},
        {"section": "Section 6 Specific Relief Act", "subject": "Recovery of possession", "note": "Summary suit for dispossession"},
        {"section": "Section 10 Specific Relief Act", "subject": "Specific performance of contract", "note": "Contracts for immoveable property are ordinarily specifically enforceable"},
    ],
    "Constitutional": [
        {"section": "Article 32 Constitution", "subject": "Writ jurisdiction of Supreme Court", "note": "Habeas corpus, mandamus, certiorari, prohibition, quo warranto"},
        {"section": "Article 226 Constitution", "subject": "Writ jurisdiction of High Court", "note": "Wider than Art 32 — extends to any person within jurisdiction"},
        {"section": "Article 14 Constitution", "subject": "Equality before law", "note": "Arbitrary executive/legislative action can be challenged"},
    ],
}


def _get_applicable_statutes(case_type: str) -> List[Dict[str, str]]:
    """Return applicable legal sections based on case type."""
    return _APPLICABLE_STATUTES.get(case_type, _APPLICABLE_STATUTES["Civil"])


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
            all_probabilities = result.get("all_probabilities", {})
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
        "class_probabilities": all_probabilities,
        "risk_level": _get_risk_level(primary_judgment, confidence),
        "confidence_band": _get_confidence_band(confidence),
        "case_type": inferred_case_type,
        "needs_review": needs_review,
        "abstention_reason": abstention_reason,
    }

    # Confidence info — rich structured block (All Plans)
    if get_confidence_language is not None:
        response_data["confidence_info"] = get_confidence_language(confidence)

    # Basic Analysis (FREE + BASIC Plans)
    if plan in ('free', 'basic', 'pro', 'unlimited'):
        response_data["explanation"] = explanation or _generate_basic_explanation(primary_judgment, confidence)
        response_data["judgment_source"] = judgment_source

        # Rich outcome description from OUTCOME_DESCRIPTIONS registry
        if OUTCOME_DESCRIPTIONS:
            rich_desc = OUTCOME_DESCRIPTIONS.get(primary_judgment)
            if rich_desc:
                response_data["outcome_description"] = rich_desc
            else:
                response_data["outcome_description"] = {
                    "meaning": _get_outcome_description(primary_judgment),
                    "implications": "Review the specific judgment for details.",
                    "next_steps": "Consult with a legal professional for specific guidance.",
                }
        else:
            response_data["outcome_description"] = _get_outcome_description(primary_judgment)

    # Outcome indicators — phrases from text that signal the outcome (FREE+)
    if plan in ('free', 'basic', 'pro', 'unlimited') and get_outcome_indicators is not None:
        try:
            indicators = get_outcome_indicators(processed or '')
            if indicators:
                response_data["outcome_indicators"] = indicators
        except Exception:
            pass

    # Key Factors (PRO+ Plans) — always present so frontend can rely on it
    if plan in ('pro', 'unlimited'):
        response_data["key_factors"] = [
            {k: v for k, v in factor.items() if not k.startswith('_')}
            for factor in key_factors
        ]

    # Applicable statutes (PRO+ Plans)
    if plan in ('pro', 'unlimited'):
        response_data["applicable_statutes"] = _get_applicable_statutes(inferred_case_type)

    # Similar Cases (PRO+ Plans) with outcome_distribution
    similar_cases: List[Dict[str, Any]] = []
    if plan in ('pro', 'unlimited') and config.INCLUDE_SIMILAR_CASES:
        similar_cases = _get_similar_cases(processed, config.SIMILAR_CASES_COUNT)
        response_data["similar_cases"] = similar_cases

        # Outcome distribution across similar cases
        if similar_cases:
            outcome_counts: Dict[str, int] = {}
            for sc in similar_cases:
                oc = sc.get('outcome', 'Unknown')
                outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
            total_sc = len(similar_cases)
            response_data["outcome_distribution"] = {
                "total_similar": total_sc,
                "distribution": {
                    k: {"count": v, "pct": round(v / total_sc * 100, 1)}
                    for k, v in outcome_counts.items()
                },
                "alignment": primary_judgment in outcome_counts,
            }

    # Text analysis — citations and parties extracted from input (PRO+)
    if plan in ('pro', 'unlimited'):
        text_analysis: Dict[str, Any] = {}
        if extract_citations is not None:
            try:
                cites = extract_citations(processed or '')
                if cites:
                    text_analysis["citations_found"] = cites
            except Exception:
                pass
        if extract_parties is not None:
            try:
                appellant, respondent = extract_parties(processed or '')
                if appellant or respondent:
                    text_analysis["parties"] = {
                        "appellant": appellant,
                        "respondent": respondent,
                    }
            except Exception:
                pass
        if extract_key_holdings is not None:
            try:
                holdings = extract_key_holdings(processed or '', max_holdings=3)
                if holdings:
                    text_analysis["key_holdings"] = holdings
            except Exception:
                pass
        if text_analysis:
            response_data["text_analysis"] = text_analysis

    # Case Summary / Brief Analysis (PRO+ Plans)
    if plan in ('pro', 'unlimited'):
        response_data["case_summary"] = _generate_case_summary(
            raw_input=raw,
            case_type=inferred_case_type,
            judgment=primary_judgment,
            confidence=confidence,
            key_factors=key_factors,
            similar_cases=similar_cases or None,
        )

    # Shadow multi-axis model breakdown (PRO+ Plans)
    if plan in ('pro', 'unlimited') and shadow and isinstance(shadow, dict):
        shadow_info: Dict[str, Any] = {}
        axes = shadow.get('axes')
        if isinstance(axes, dict):
            shadow_info["axes"] = axes
            shadow_info["main_judgment"] = shadow.get('main_judgment')
            shadow_info["agrees_with_classical"] = (
                shadow.get('main_judgment') == judgment
            )
        if agreement_rate is not None:
            shadow_info["agreement_rate"] = round(agreement_rate, 3)
        if shadow_info:
            response_data["multi_axis_analysis"] = shadow_info

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
    model_run_id = None
    if state.classifier and hasattr(state.classifier, 'run_id'):
        model_run_id = state.classifier.run_id
    response_data["metadata"] = {
        "model_version": config.APP_VERSION,
        "model_run_id": model_run_id,
        "model_type": "TF-IDF + RandomForest",
        "request_id": request_id,
        "processed_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "processing_time_ms": processing_time_ms,
        "rag_sources_used": len(response_data.get("similar_cases", [])),
        "tokens_used": len(processed.split()) if processed else 0,
        "plan": plan,
        "shadow_model_available": state.multi_axis_bundle is not None,
    }

    # Input Echo (For DB storage)
    response_data["answers"] = data

    return jsonify(response_data)

@analysis_bp.route('/api/rag/query', methods=['POST'])
@limiter.limit("30/minute")
def rag_query_endpoint():
    """RAG-style query endpoint — uses LLM when available, falls back to retrieval-only."""
    # Ensure lazy-loaded search index is available
    dependencies.ensure_search_index()
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400
    question = raw.get('question') or ''
    if not question:
        return jsonify({"error": "missing_question"}), 400
    
    k = min(raw.get('k', 5), config.MAX_SEARCH_RESULTS)
    
    # Use the RAG pipeline with LLM support
    try:
        from ai_court.rag.pipeline import rag_query as rag_pipeline
        
        response = rag_pipeline(
            question=question,
            search_index=state.search_index,
            preprocess_fn=state.preprocess_fn,
            k=k,
            llm_client=state.llm_client,
            statute_corpus=state.statute_corpus,
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

            if format_minimal_response is not None:
                resp = format_minimal_response(
                    result["judgment"],
                    result["confidence"],
                    case_type or "Unknown",
                )
            else:
                resp = {
                    "judgment": result["judgment"],
                    "confidence": round(result["confidence"] or 0, 3),
                    "case_type": case_type or "Unknown",
                }
            resp["timing_ms"] = round(elapsed_ms, 2)

            return jsonify(resp)
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

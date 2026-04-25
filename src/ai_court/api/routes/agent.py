"""Agent API Routes — endpoints for the AI Legal Agent.

Endpoints:
  POST /api/agent/analyze            Full case analysis (ML + LLM + Search + Statutes)
  POST /api/agent/analyze-with-docs  Full analysis with uploaded documents
  POST /api/agent/chat               Follow-up chat within a case session
  POST /api/agent/rag                RAG-powered quick legal Q&A
  POST /api/agent/stream             SSE streaming: Query → Law → Cases → LLM tokens
  POST /api/agent/upload-documents   Upload PDFs/images/docs for AI reading
  POST /api/agent/generate-document  Generate court-ready documents (bail apps, appeals, etc.)
  GET  /api/agent/document-types     List available document types
  GET  /api/agent/session/<id>       Get session info
  GET  /api/agent/sessions           List active sessions for a user
  GET  /api/agent/health             Agent health check

Tier system:
  free     - Basic ML prediction only (no LLM)
  pro      - Full analysis + 5 follow-up questions
  unlimited- Full analysis + unlimited follow-ups + doc reading + doc generation
  court    - Everything + full audit trail + court-presentable documents
"""
from __future__ import annotations

import logging
import re
import time
from flask import Blueprint, request, jsonify, Response, stream_with_context

from ai_court.api import state, config
from ai_court.api.extensions import limiter

logger = logging.getLogger(__name__)
agent_bp = Blueprint("agent", __name__)

# ── Legal Disclaimer (appended to EVERY AI response) ──────────────────
LEGAL_DISCLAIMER = (
    "DISCLAIMER: This AI-generated legal analysis is produced by an automated "
    "system and does NOT constitute legal advice. It is intended as a supportive "
    "research tool only. No attorney-client relationship is created by using this "
    "service. Always consult a qualified legal professional before making any legal "
    "decisions or taking action based on this output. The developers and operators "
    "of this platform accept no liability for actions taken based on AI predictions."
)

# ── Input Validation Constants ─────────────────────────────────────────
MAX_QUERY_LENGTH = 15000       # ~3000 words max
MAX_MESSAGE_LENGTH = 5000
MIN_QUERY_LENGTH = 10

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(above|prior|previous)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"new\s+system\s+prompt",
    r"override\s+system\s+prompt",
    r"forget\s+(all\s+)?(your|previous)\s+instructions",
    r"\bsystem:\s*",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _validate_query(text: str, max_len: int = MAX_QUERY_LENGTH) -> str | None:
    """Return an error message if the query is invalid, else None."""
    if not text or len(text.strip()) < MIN_QUERY_LENGTH:
        return f"Query must be at least {MIN_QUERY_LENGTH} characters."
    if len(text) > max_len:
        return f"Query exceeds maximum length of {max_len} characters."
    if _INJECTION_RE.search(text):
        return "Query contains disallowed instructions. Please rephrase."
    return None


def _add_disclaimer(response: dict) -> dict:
    """Inject legal disclaimer into every AI response dict."""
    response["legal_disclaimer"] = LEGAL_DISCLAIMER
    return response

# Follow-up limits per tier
TIER_FOLLOWUP_LIMITS = {
    "free": 0,
    "pro": 5,
    "unlimited": 999,
    "court": 999,
}


def _get_agent():
    """Get the agent pipeline, raising 503 if not initialized."""
    if state.agent_pipeline is None:
        return None
    return state.agent_pipeline


def _get_session_manager():
    """Get session manager."""
    return state.session_manager


# =========================================================================
# POST /api/agent/analyze — Full case analysis
# =========================================================================
@agent_bp.route("/api/agent/analyze", methods=["POST"])
@limiter.limit("10/minute")
def agent_analyze():
    """Full AI legal agent analysis.

    Request JSON:
      {
        "query": "My client was charged with murder under IPC 302...",
        "tier": "pro",           // free|pro|unlimited|court
        "user_id": "user_123",   // optional
        "k_cases": 5,            // optional, default 5
        "include_strategy": true  // optional, default true
      }

    Response: Full analysis with prediction, cases, statutes, LLM analysis.
    """
    agent = _get_agent()
    raw = request.json or {}
    if not isinstance(raw, dict):
        return jsonify({"error": "invalid_body"}), 400

    query = (raw.get("query") or "").strip()
    if not query:
        return jsonify({"error": "missing_query", "message": "Provide a 'query' field."}), 400
    err = _validate_query(query)
    if err:
        return jsonify({"error": "invalid_query", "message": err}), 400

    tier = raw.get("tier", "free")
    user_id = raw.get("user_id")
    k_cases = min(int(raw.get("k_cases", 5)), 10)
    include_strategy = raw.get("include_strategy", True)

    # ---------- Free tier: ML-only (no LLM needed) ----------
    if tier == "free" or agent is None:
        result = _free_tier_analysis(query, k_cases)
        if agent is None and tier != "free":
            result["warning"] = "LLM agent not available. Showing ML-only results."
        result["tier"] = tier
        return jsonify(_add_disclaimer(result))

    # ---------- Pro / Unlimited / Court: Full LLM agent ----------
    try:
        analysis = agent.analyze(
            query,
            k_cases=k_cases,
            include_strategy=include_strategy,
        )
    except Exception as exc:
        logger.error("Agent analysis failed: %s", exc)
        # Fallback to ML-only
        result = _free_tier_analysis(query, k_cases)
        result["warning"] = f"LLM analysis failed ({type(exc).__name__}). Showing ML-only results."
        result["tier"] = tier
        return jsonify(_add_disclaimer(result))

    # Create a session for follow-up chat
    sm = _get_session_manager()
    session_id = None
    if sm is not None:
        session = sm.create_session(
            user_id=user_id,
            case_query=query,
            case_type=analysis.get("understanding", {}).get("case_type", ""),
            analysis_result=analysis,
            tier=tier,
        )
        session_id = session.session_id

    # ---------- Build tiered response ----------
    response = {
        "tier": tier,
        "session_id": session_id,
        "query": query,
        "prediction": analysis.get("prediction"),
        "understanding": analysis.get("understanding"),
    }

    # Pro and above: full analysis
    if tier in ("pro", "unlimited", "court"):
        response["analysis"] = analysis.get("analysis")
        response["similar_cases"] = analysis.get("similar_cases")
        response["statute_context"] = analysis.get("statute_context")
        response["strategy"] = analysis.get("strategy")
        response["metadata"] = analysis.get("metadata")
        response["followup_limit"] = TIER_FOLLOWUP_LIMITS.get(tier, 0)

    # Court tier: add audit trail
    if tier == "court":
        response["audit_trail"] = {
            "timestamp": time.time(),
            "model_version": config.APP_VERSION,
            "llm_model": analysis.get("metadata", {}).get("model_used", "unknown"),
            "cases_referenced": len(analysis.get("similar_cases", [])),
            "processing_time": analysis.get("metadata", {}).get("processing_time_seconds"),
            "disclaimer": (
                "This AI-generated legal analysis is produced by an automated system "
                "and should be used as a supportive tool only. It does not constitute "
                "legal advice and should be reviewed by a qualified legal professional "
                "before being presented in any court of law."
            ),
        }
        response["court_document"] = _build_court_document(analysis)

    return jsonify(_add_disclaimer(response))


# =========================================================================
# POST /api/agent/chat — Follow-up chat in a case session
# =========================================================================
@agent_bp.route("/api/agent/chat", methods=["POST"])
@limiter.limit("30/minute")
def agent_chat():
    """Follow-up question within an existing case session.

    Request JSON:
      {
        "session_id": "uuid",
        "message": "Can I also file a counter-FIR?"
      }

    Response: LLM answer with full case context.
    """
    agent = _get_agent()
    if agent is None:
        return jsonify({"error": "agent_unavailable", "message": "AI agent not initialized."}), 503

    sm = _get_session_manager()
    if sm is None:
        return jsonify({"error": "sessions_unavailable"}), 503

    raw = request.json or {}
    session_id = raw.get("session_id", "")
    message = (raw.get("message") or "").strip()

    if not session_id or not message:
        return jsonify({"error": "missing_fields", "message": "Provide session_id and message."}), 400
    err = _validate_query(message, MAX_MESSAGE_LENGTH)
    if err:
        return jsonify({"error": "invalid_message", "message": err}), 400

    session = sm.get_session(session_id)
    if session is None:
        return jsonify({"error": "session_not_found", "message": "Session expired or not found."}), 404

    # Check follow-up limits
    followup_limit = TIER_FOLLOWUP_LIMITS.get(session.tier, 0)
    # Only count user messages after the initial one
    user_msgs = sum(1 for m in session.conversation_history if m["role"] == "user") - 1
    if user_msgs >= followup_limit:
        return jsonify({
            "error": "followup_limit_reached",
            "message": f"Your {session.tier} plan allows {followup_limit} follow-up questions. Upgrade for more.",
            "tier": session.tier,
            "used": user_msgs,
            "limit": followup_limit,
        }), 403

    # Add user message
    session.add_message("user", message)

    # Get LLM response with conversation history
    try:
        history = session.get_history_for_llm()
        answer = agent.follow_up(message, history)
        session.add_message("assistant", answer)
    except Exception as exc:
        logger.error("Agent follow-up failed: %s", exc)
        return jsonify({"error": "agent_error", "message": str(exc)}), 500

    return jsonify(_add_disclaimer({
        "session_id": session_id,
        "answer": answer,
        "followups_used": user_msgs + 1,
        "followups_remaining": max(0, followup_limit - user_msgs - 1),
    }))


# =========================================================================
# POST /api/agent/rag — Quick RAG Q&A (no session)
# =========================================================================
@agent_bp.route("/api/agent/rag", methods=["POST"])
@limiter.limit("20/minute")
def agent_rag():
    """RAG-powered quick legal Q&A.

    Request JSON:
      {"question": "Can bail be granted in NDPS cases?", "k": 5}

    Response: LLM-generated answer with case citations.
    """
    agent = _get_agent()
    raw = request.json or {}
    question = (raw.get("question") or "").strip()
    if not question:
        return jsonify({"error": "missing_question"}), 400
    err = _validate_query(question)
    if err:
        return jsonify({"error": "invalid_question", "message": err}), 400

    k = min(int(raw.get("k", 5)), 10)

    # If agent available, use full RAG-LLM
    if agent is not None:
        try:
            result = agent.rag_answer(question, k=k)
            return jsonify(_add_disclaimer(result))
        except Exception as exc:
            logger.warning("Agent RAG failed: %s", exc)
            # Fall through to retrieval-only

    # Fallback: retrieval-only
    from ai_court.rag.pipeline import rag_query
    result = rag_query(
        question=question,
        search_index=state.search_index,
        preprocess_fn=state.preprocess_fn,
        k=k,
        llm_client=state.llm_client,
        statute_corpus=state.statute_corpus,
    )
    return jsonify(_add_disclaimer(result))


# =========================================================================
# GET /api/agent/session/<id> — Session info
# =========================================================================
@agent_bp.route("/api/agent/session/<session_id>", methods=["GET"])
def agent_session_info(session_id: str):
    sm = _get_session_manager()
    if sm is None:
        return jsonify({"error": "sessions_unavailable"}), 503
    session = sm.get_session(session_id)
    if session is None:
        return jsonify({"error": "session_not_found"}), 404
    return jsonify(session.to_dict())


# =========================================================================
# GET /api/agent/sessions — List sessions
# =========================================================================
@agent_bp.route("/api/agent/sessions", methods=["GET"])
def agent_sessions():
    sm = _get_session_manager()
    if sm is None:
        return jsonify({"error": "sessions_unavailable"}), 503
    user_id = request.args.get("user_id")
    return jsonify({"sessions": sm.list_sessions(user_id=user_id)})


# =========================================================================
# GET /api/agent/health — Agent health
# =========================================================================
@agent_bp.route("/api/agent/health", methods=["GET"])
def agent_health():
    agent = _get_agent()
    sm = _get_session_manager()
    return jsonify({
        "agent_ready": agent is not None,
        "llm_client_ready": state.llm_client is not None,
        "statute_corpus_loaded": (
            state.statute_corpus is not None
            and state.statute_corpus.loaded
        ),
        "session_manager_ready": sm is not None,
        "active_sessions": sm.active_count if sm else 0,
        "classifier_ready": state.classifier is not None,
        "search_index_ready": state.search_index is not None,
    })


# =========================================================================
# Helpers
# =========================================================================
def _free_tier_analysis(query: str, k_cases: int = 3) -> dict:
    """ML-only analysis for free tier (no LLM calls)."""
    result: dict = {"tier": "free", "query": query}

    # ML prediction
    if state.classifier is not None:
        try:
            pred = state.classifier.predict(query, "")
            result["prediction"] = {
                "judgment": pred.get("judgment"),
                "confidence": pred.get("confidence"),
                "all_probabilities": pred.get("all_probabilities", {}),
            }
        except Exception:
            result["prediction"] = {"judgment": "Unknown", "confidence": 0.0}
    else:
        result["prediction"] = {"judgment": "Unknown", "confidence": 0.0}

    # Similar cases (no LLM needed)
    from ai_court.api.routes.analysis import _get_similar_cases

    processed = state.preprocess_fn(query) if state.preprocess_fn else query.lower()
    result["similar_cases"] = _get_similar_cases(processed, k=k_cases)

    return result


def _build_court_document(analysis: dict) -> dict:
    """Build a structured document suitable for court presentation."""
    understanding = analysis.get("understanding", {})
    prediction = analysis.get("prediction", {})

    return {
        "title": "AI-Assisted Legal Analysis Report",
        "case_type": understanding.get("case_type", "Unknown"),
        "legal_issues": understanding.get("legal_issues", []),
        "applicable_law": understanding.get("relevant_acts", []),
        "sections_cited": understanding.get("relevant_sections", []),
        "ml_prediction": {
            "outcome": prediction.get("judgment"),
            "confidence": prediction.get("confidence"),
            "model_type": "TF-IDF + RandomForest (300 estimators)",
            "training_data": "10,838 Indian court cases from Indian Kanoon",
        },
        "precedent_cases_cited": [
            {
                "title": c.get("title"),
                "url": c.get("url"),
                "outcome": c.get("outcome"),
                "relevance_score": c.get("score"),
            }
            for c in analysis.get("similar_cases", [])
        ],
        "statutory_provisions": analysis.get("statute_context", ""),
        "ai_analysis": analysis.get("analysis", ""),
        "strategy_recommendations": analysis.get("strategy", ""),
        "disclaimer": (
            "This document has been generated by an AI legal analysis system. "
            "All case citations, statutory references, and legal analysis should be "
            "independently verified by a qualified advocate before being relied upon "
            "in any court proceedings. This tool is intended as a legal research aid "
            "and does not constitute legal advice."
        ),
    }


# =========================================================================
# POST /api/agent/upload-documents — Upload case documents for AI reading
# =========================================================================
@agent_bp.route("/api/agent/upload-documents", methods=["POST"])
@limiter.limit("10/minute")
def agent_upload_documents():
    """Upload case documents (PDF, images, DOCX) for the agent to read.

    Form-data:
        files: One or more files (multipart/form-data)
        session_id: Optional session to attach documents to

    Response: Extracted text, entities, document types for each file.
    """
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"error": "no_files", "message": "Upload at least one file using 'files' or 'file' field."}), 400

    files = request.files.getlist("files") or [request.files.get("file")]
    files = [f for f in files if f and f.filename]

    if not files:
        return jsonify({"error": "no_valid_files"}), 400
    if len(files) > 20:
        return jsonify({"error": "too_many_files", "message": "Maximum 20 files per upload."}), 400

    session_id = request.form.get("session_id", "")

    # Process documents
    try:
        from ai_court.documents.processor import DocumentProcessor, format_documents_for_llm

        processor = DocumentProcessor(llm_client=state.llm_client)
        file_data = []
        for f in files:
            data = f.read()
            if len(data) > 50 * 1024 * 1024:  # 50MB limit per file
                continue
            file_data.append({
                "file_bytes": data,
                "filename": f.filename,
                "content_type": f.content_type,
            })

        results = processor.process_multiple(file_data)

        # Attach to session if provided
        if session_id:
            sm = _get_session_manager()
            if sm:
                session = sm.get_session(session_id)
                if session:
                    docs_context = format_documents_for_llm(results)
                    session.add_message("system", f"[User uploaded {len(results)} document(s)]\n{docs_context}")

        response_docs = []
        for doc in results:
            response_docs.append({
                "filename": doc.filename,
                "file_type": doc.file_type,
                "doc_type": doc.doc_type_guess,
                "text_length": len(doc.text),
                "text_preview": doc.text[:500] + ("..." if len(doc.text) > 500 else ""),
                "sections_mentioned": doc.sections_mentioned[:15],
                "citations_found": doc.citations_found[:10],
                "parties_mentioned": doc.parties_mentioned,
                "dates_found": doc.dates_found[:10],
                "evidence_description": doc.evidence_description,
                "confidence": doc.confidence,
            })

        # Build combined context for agent use
        combined_context = format_documents_for_llm(results)

        return jsonify({
            "documents": response_docs,
            "total_files": len(results),
            "combined_text_length": len(combined_context),
            "documents_context": combined_context,  # For passing to /agent/analyze
        })

    except Exception as exc:
        logger.error("Document processing failed: %s", exc, exc_info=True)
        return jsonify({"error": "processing_failed", "message": str(exc)}), 500


# =========================================================================
# POST /api/agent/analyze-with-docs — Full analysis with uploaded documents
# =========================================================================
@agent_bp.route("/api/agent/analyze-with-docs", methods=["POST"])
@limiter.limit("5/minute")
def agent_analyze_with_docs():
    """Full agent analysis with uploaded document context.

    Form-data:
        query: Case description text
        files: One or more document files (optional)
        tier: free|pro|unlimited|court
        user_id: Optional user identifier
        documents_context: Pre-processed document text (from /upload-documents)

    Response: Full analysis incorporating document contents.
    """
    agent = _get_agent()

    # Handle both JSON and form-data
    if request.is_json:
        raw = request.json or {}
        query = (raw.get("query") or "").strip()
        tier = raw.get("tier", "pro")
        user_id = raw.get("user_id")
        documents_context = raw.get("documents_context", "")
    else:
        query = request.form.get("query", "").strip()
        tier = request.form.get("tier", "pro")
        user_id = request.form.get("user_id")
        documents_context = request.form.get("documents_context", "")

    if not query:
        return jsonify({"error": "missing_query"}), 400

    # Process any new uploaded files
    if request.files:
        try:
            from ai_court.documents.processor import DocumentProcessor, format_documents_for_llm
            processor = DocumentProcessor(llm_client=state.llm_client)
            file_data = []
            for key in request.files:
                for f in request.files.getlist(key):
                    if f and f.filename:
                        data = f.read()
                        if len(data) <= 50 * 1024 * 1024:
                            file_data.append({
                                "file_bytes": data,
                                "filename": f.filename,
                                "content_type": f.content_type,
                            })
            if file_data:
                results = processor.process_multiple(file_data)
                new_context = format_documents_for_llm(results)
                documents_context = (documents_context + "\n\n" + new_context).strip() if documents_context else new_context
        except Exception as exc:
            logger.warning("Document processing in analyze: %s", exc)

    # Free tier: ML only
    if tier == "free" or agent is None:
        result = _free_tier_analysis(query)
        if documents_context:
            result["documents_processed"] = True
        result["tier"] = tier
        return jsonify(result)

    # Full analysis with documents
    try:
        analysis = agent.analyze(
            query,
            include_strategy=True,
            documents_context=documents_context,
        )
    except Exception as exc:
        logger.error("Agent analysis with docs failed: %s", exc)
        result = _free_tier_analysis(query)
        result["warning"] = "LLM analysis failed. Showing ML-only results."
        result["tier"] = tier
        return jsonify(result)

    # Create session
    sm = _get_session_manager()
    session_id = None
    if sm:
        session = sm.create_session(
            user_id=user_id,
            case_query=query,
            case_type=analysis.get("understanding", {}).get("case_type", ""),
            analysis_result=analysis,
            tier=tier,
        )
        session_id = session.session_id
        if documents_context:
            session.add_message("system", f"[Documents uploaded and analyzed]\n{documents_context[:2000]}")

    response = {
        "tier": tier,
        "session_id": session_id,
        "query": query,
        "prediction": analysis.get("prediction"),
        "understanding": analysis.get("understanding"),
        "analysis": analysis.get("analysis"),
        "similar_cases": analysis.get("similar_cases"),
        "statute_context": analysis.get("statute_context"),
        "strategy": analysis.get("strategy"),
        "metadata": analysis.get("metadata"),
        "documents_processed": bool(documents_context),
        "followup_limit": TIER_FOLLOWUP_LIMITS.get(tier, 0),
    }

    if tier == "court":
        response["audit_trail"] = {
            "timestamp": time.time(),
            "documents_included": bool(documents_context),
        }
        response["court_document"] = _build_court_document(analysis)

    return jsonify(response)


# =========================================================================
# POST /api/agent/generate-document — Generate court-ready legal documents
# =========================================================================
@agent_bp.route("/api/agent/generate-document", methods=["POST"])
@limiter.limit("5/minute")
def agent_generate_document():
    """Generate a court-ready legal document.

    Request JSON:
      {
        "doc_type": "bail_application",  // See /api/agent/document-types
        "case_info": "My client was arrested under IPC 302...",
        "session_id": "uuid",            // Optional: use session's analysis
        "documents_context": "...",       // Optional: uploaded doc text
        "user_instructions": "Focus on lack of evidence"
      }

    Response: Generated document content.
    """
    if state.llm_client is None:
        return jsonify({"error": "llm_unavailable", "message": "LLM not configured."}), 503

    raw = request.json or {}
    doc_type = raw.get("doc_type", "case_summary")
    case_info = raw.get("case_info", "")
    session_id = raw.get("session_id", "")
    documents_context = raw.get("documents_context", "")
    user_instructions = raw.get("user_instructions", "")

    if not case_info and not session_id:
        return jsonify({"error": "missing_info", "message": "Provide case_info or session_id."}), 400

    # Pull context from session if available
    analysis_summary = ""
    statute_context = ""
    precedents = ""
    if session_id:
        sm = _get_session_manager()
        if sm:
            session = sm.get_session(session_id)
            if session and session.analysis_result:
                a = session.analysis_result
                analysis_summary = a.get("analysis", "")
                statute_context = a.get("statute_context", "")
                precedents = "\n".join([
                    f"- {c.get('title', 'Case')}: {c.get('outcome', '')}"
                    for c in a.get("similar_cases", [])
                ])
                if not case_info:
                    case_info = session.case_query

    try:
        from ai_court.documents.court_docs import CourtDocumentGenerator
        generator = CourtDocumentGenerator(state.llm_client)
        doc = generator.generate(
            doc_type=doc_type,
            case_info=case_info,
            analysis_summary=analysis_summary,
            statute_context=statute_context,
            precedents=precedents,
            document_context=documents_context,
            user_instructions=user_instructions,
        )
        return jsonify({
            "doc_type": doc.doc_type,
            "title": doc.title,
            "content": doc.content,
            "generated_at": doc.generated_at,
            "metadata": doc.metadata,
        })
    except ValueError as e:
        return jsonify({"error": "invalid_doc_type", "message": str(e)}), 400
    except Exception as exc:
        logger.error("Document generation failed: %s", exc, exc_info=True)
        return jsonify({"error": "generation_failed", "message": str(exc)}), 500


# =========================================================================
# GET /api/agent/document-types — List available document types
# =========================================================================
@agent_bp.route("/api/agent/document-types", methods=["GET"])
def agent_document_types():
    """List all court document types the agent can generate."""
    from ai_court.documents.court_docs import DOCUMENT_TYPES
    types = [
        {"id": k, "title": v["title"], "sections": v["sections"]}
        for k, v in DOCUMENT_TYPES.items()
    ]
    return jsonify({"document_types": types})


# =========================================================================
# POST /api/agent/stream — SSE streaming chat/RAG (Phase 2 deliverable)
# =========================================================================
def _sse_event(event: str, data: str) -> str:
    """Format a Server-Sent Event payload."""
    # Escape newlines per SSE spec
    data_lines = "\n".join(f"data: {line}" for line in data.splitlines() or [""])
    return f"event: {event}\n{data_lines}\n\n"


@agent_bp.route("/api/agent/stream", methods=["POST"])
@limiter.limit("20/minute")
def agent_stream():
    """Streaming endpoint: Query → Law Search → Case Search → LLM (token-by-token).

    Request JSON:
      {
        "query": "Can I get bail in NDPS case?",
        "session_id": null,        // optional — continues existing session context
        "k_cases": 5,
        "k_statutes": 5
      }

    Response: ``text/event-stream`` with events:
      - ``status``   : progress updates (understanding, searching, generating)
      - ``citations``: JSON list of retrieved cases + statutes
      - ``token``    : individual LLM tokens
      - ``done``     : final marker with metadata
      - ``error``    : on failure
    """
    raw = request.json or {}
    query = (raw.get("query") or "").strip()
    if not query:
        return jsonify({"error": "missing_query"}), 400
    err = _validate_query(query)
    if err:
        return jsonify({"error": "invalid_query", "message": err}), 400

    k_cases = min(int(raw.get("k_cases", 5)), 10)
    k_statutes = min(int(raw.get("k_statutes", 5)), 10)
    session_id = raw.get("session_id")
    user_id = raw.get("user_id")
    tier = raw.get("tier", "pro")

    if state.llm_client is None:
        return jsonify({"error": "llm_unavailable"}), 503

    agent = _get_agent()
    sm = _get_session_manager()

    # Auto-create a session when one is not supplied. This enables multi-turn
    # chat: the client then echoes the returned session_id on subsequent calls.
    if not session_id and sm is not None:
        try:
            sess = sm.create_session(
                user_id=user_id,
                case_query=query,
                case_type="",
                tier=tier,
            )
            session_id = sess.session_id
        except Exception as exc:
            logger.warning("stream: auto session create failed: %s", exc)
            session_id = None

    @stream_with_context
    def generate():
        import json as _json

        t0 = time.perf_counter()
        try:
            yield _sse_event("status", _json.dumps({"step": "understanding", "message": "Analyzing your query…"}))

            # --- Retrieval -------------------------------------------------
            cases: list = []
            statute_ctx = ""
            if agent is not None:
                yield _sse_event("status", _json.dumps({"step": "case_search", "message": "Searching case law…"}))
                try:
                    cases = agent.find_similar_cases(query, k=k_cases)
                except Exception as exc:
                    logger.warning("stream: case search failed: %s", exc)

                yield _sse_event("status", _json.dumps({"step": "statute_search", "message": "Searching statutes…"}))
                try:
                    statute_ctx = agent.find_relevant_statutes(query, k=k_statutes)
                except Exception as exc:
                    logger.warning("stream: statute search failed: %s", exc)
            elif state.search_index is not None:
                # Fallback: pure retrieval using rag pipeline
                from ai_court.rag.pipeline import retrieve
                cases = retrieve(query, state.search_index, state.preprocess_fn, k=k_cases)
                if state.statute_corpus is not None and state.statute_corpus.loaded:
                    secs = state.statute_corpus.search_sections(query, k=k_statutes)
                    statute_ctx = state.statute_corpus.format_for_context(secs, max_length=3000)

            citations_payload = {
                "cases": [
                    {
                        "title": c.get("title"),
                        "url": c.get("url"),
                        "outcome": c.get("outcome"),
                        "score": c.get("score"),
                    }
                    for c in cases
                ],
                "statutes_excerpt": statute_ctx[:1500] if statute_ctx else "",
            }
            yield _sse_event("citations", _json.dumps(citations_payload))

            # --- Generation (token stream) --------------------------------
            yield _sse_event("status", _json.dumps({"step": "generating", "message": "Drafting response…"}))

            from ai_court.llm.prompts import SYSTEM_PROMPT_LEGAL_AGENT

            history_msgs: list = []
            if session_id:
                sm = _get_session_manager()
                if sm:
                    sess = sm.get_session(session_id)
                    if sess:
                        history_msgs = sess.get_history_for_llm(max_messages=10)

            case_context_lines = []
            for i, c in enumerate(cases[:5], 1):
                case_context_lines.append(
                    f"[{i}] {c.get('title', 'Case')} — Outcome: {c.get('outcome', 'Unknown')}\n"
                    f"    Excerpt: {(c.get('snippet') or '')[:300]}"
                )
            case_context = "\n".join(case_context_lines) or "No precedent matches found."

            user_prompt = (
                f"USER QUERY: {query}\n\n"
                f"RELEVANT STATUTES:\n{statute_ctx or 'None retrieved.'}\n\n"
                f"RELEVANT CASES:\n{case_context}\n\n"
                f"Provide a well-cited, structured legal response grounded in these sources."
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT}]
            messages.extend(history_msgs)
            messages.append({"role": "user", "content": user_prompt})

            full_answer_parts: list = []
            try:
                for chunk in state.llm_client.chat(messages, stream=True):
                    if not chunk:
                        continue
                    full_answer_parts.append(chunk)
                    yield _sse_event("token", _json.dumps({"text": chunk}))
            except Exception as exc:
                logger.error("stream: LLM error: %s", exc)
                yield _sse_event("error", _json.dumps({"message": str(exc)}))
                return

            # Persist to session
            if session_id:
                sm = _get_session_manager()
                if sm:
                    sess = sm.get_session(session_id)
                    if sess:
                        sess.add_message("user", query)
                        sess.add_message("assistant", "".join(full_answer_parts))

            yield _sse_event("done", _json.dumps({
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "model": state.llm_client.model,
                "provider": getattr(state.llm_client, "provider", "unknown"),
                "citation_count": len(cases),
                "session_id": session_id,
                "legal_disclaimer": LEGAL_DISCLAIMER,
            }))
        except Exception as exc:
            logger.error("stream: unhandled: %s", exc, exc_info=True)
            yield _sse_event("error", _json.dumps({"message": str(exc)}))

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )

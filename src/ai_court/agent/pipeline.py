"""Legal Agent Pipeline — the orchestrator that connects all AI components.

Flow:
  1. User query → LLM extracts structured understanding (case_type, issues, sections)
  2. ML classifier predicts outcome + confidence + probabilities
  3. Statute corpus searched for relevant provisions
  4. TF-IDF/hybrid search finds similar cases
  5. Evidence admissibility checked for red flags
  6. Cross-reference old ↔ new codes (IPC→BNS, CrPC→BNSS)
  7. Procedural checklist attached for case type
  8. Judge pattern analyzed (if judge name known)
  9. All context assembled → LLM generates comprehensive legal analysis
  10. Strategy synthesis for actionable advice
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ai_court.llm.client import LLMClient
from ai_court.llm.prompts import (
    CASE_ANALYSIS_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
    STRATEGY_SYNTHESIS_PROMPT,
    SYSTEM_PROMPT_LEGAL_AGENT,
)

logger = logging.getLogger(__name__)


class LegalAgentPipeline:
    """Orchestrates the full AI legal agent: ML + Search + Statutes + LLM + Intelligence."""

    def __init__(
        self,
        llm_client: LLMClient,
        classifier: Any = None,
        search_index: Any = None,
        statute_corpus: Any = None,
        preprocess_fn: Optional[Callable[[str], str]] = None,
    ):
        self.llm = llm_client
        self.classifier = classifier
        self.search_index = search_index
        self.statute_corpus = statute_corpus
        self.preprocess_fn = preprocess_fn or (lambda x: x.lower())

        # Elite intelligence modules (lazy-initialized)
        self._evidence_checker = None
        self._checklist_engine = None
        self._cross_ref = None
        self._judge_analyzer = None
        self._init_intelligence()

    def _init_intelligence(self):
        """Initialize elite intelligence modules."""
        try:
            from ai_court.intelligence import (
                EvidenceAdmissibilityChecker,
                ProceduralChecklistEngine,
                CrossReferenceEngine,
                JudgePatternAnalyzer,
            )
            self._evidence_checker = EvidenceAdmissibilityChecker()
            self._checklist_engine = ProceduralChecklistEngine()
            self._cross_ref = CrossReferenceEngine()
            self._cross_ref.load()
            self._judge_analyzer = JudgePatternAnalyzer(search_index=self.search_index)
            logger.info("[agent] Elite intelligence modules loaded")
        except Exception as e:
            logger.warning("[agent] Intelligence modules partially loaded: %s", e)

    # ------------------------------------------------------------------
    # Step 1: Understand the query
    # ------------------------------------------------------------------
    def understand_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to extract structured information from the user's query."""
        prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self.llm.chat(messages, temperature=0.1, max_tokens=1024)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
            understanding = json.loads(raw)
            return understanding
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Query understanding LLM parse failed: %s", exc)
            return {
                "case_type": "Unknown",
                "legal_issues": [],
                "relevant_acts": [],
                "relevant_sections": [],
                "party_role": "unknown",
                "relief_sought": "unknown",
                "key_facts": [],
            }

    # ------------------------------------------------------------------
    # Step 2: ML prediction
    # ------------------------------------------------------------------
    def predict_outcome(self, query: str, case_type: str) -> Dict[str, Any]:
        """Get ML classifier prediction."""
        if self.classifier is None:
            return {"judgment": "Unknown", "confidence": 0.0, "all_probabilities": {}}
        try:
            result = self.classifier.predict(query, case_type)
            return result
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            return {"judgment": "Unknown", "confidence": 0.0, "all_probabilities": {}}

    # ------------------------------------------------------------------
    # Step 3: Search for similar cases
    # ------------------------------------------------------------------
    def find_similar_cases(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar cases using TF-IDF search index."""
        if self.search_index is None:
            return []
        try:
            import numpy as np

            vectorizer = self.search_index.get("vectorizer")
            matrix = self.search_index.get("matrix")
            meta = self.search_index.get("meta", [])

            if vectorizer is None or matrix is None:
                return []

            processed = self.preprocess_fn(query)
            qv = vectorizer.transform([processed])
            scores = (matrix @ qv.T).toarray().ravel()
            top_idx = np.argsort(-scores)[:k]

            cases = []
            for idx in top_idx:
                if idx < len(meta) and scores[idx] > 0.05:
                    m = meta[idx]
                    cases.append(
                        {
                            "title": m.get("title", "Unknown Case"),
                            "url": m.get("url", ""),
                            "outcome": m.get("outcome", "Unknown"),
                            "snippet": m.get("snippet", "")[:500],
                            "score": float(scores[idx]),
                        }
                    )
            return cases
        except Exception as exc:
            logger.warning("Similar case search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 4: Find relevant statutes
    # ------------------------------------------------------------------
    def find_relevant_statutes(
        self, query: str, acts: Optional[List[str]] = None, k: int = 10
    ) -> str:
        """Search statute corpus for relevant provisions."""
        if self.statute_corpus is None or not self.statute_corpus.loaded:
            return "No statutory provisions available."
        try:
            search_text = query
            if acts:
                search_text += " " + " ".join(acts)
            sections = self.statute_corpus.search_sections(search_text, k=k)
            return self.statute_corpus.format_for_context(sections, max_length=4000)
        except Exception as exc:
            logger.warning("Statute search failed: %s", exc)
            return "Statute lookup failed."

    # ------------------------------------------------------------------
    # Step 5: Full case analysis via LLM
    # ------------------------------------------------------------------
    def _format_case_context(self, cases: List[Dict[str, Any]]) -> str:
        """Format similar cases for LLM prompt."""
        if not cases:
            return "No directly relevant precedent cases found in database."
        parts = []
        for i, c in enumerate(cases, 1):
            entry = (
                f"[{i}] {c.get('title', 'Unknown')}\n"
                f"    Outcome: {c.get('outcome', 'Unknown')}\n"
                f"    Relevance Score: {c.get('score', 0):.2f}\n"
                f"    Summary: {c.get('snippet', 'N/A')}\n"
            )
            url = c.get("url")
            if url:
                entry += f"    Source: {url}\n"
            parts.append(entry)
        return "\n".join(parts)

    def _format_ml_prediction(self, prediction: Dict[str, Any]) -> str:
        """Format ML prediction for LLM prompt."""
        judgment = prediction.get("judgment", "Unknown")
        confidence = prediction.get("confidence")
        probas = prediction.get("all_probabilities", {})

        lines = [f"Predicted Outcome: {judgment}"]
        if confidence is not None:
            lines.append(f"Confidence: {confidence:.1%}")
        if probas:
            lines.append("Probability Distribution:")
            for label, prob in sorted(probas.items(), key=lambda x: -x[1]):
                lines.append(f"  - {label}: {prob:.1%}")
        return "\n".join(lines)

    def analyze(
        self,
        query: str,
        *,
        k_cases: int = 5,
        k_statutes: int = 10,
        include_strategy: bool = True,
        documents_context: str = "",
    ) -> Dict[str, Any]:
        """Run the FULL agent pipeline: understand → predict → search → analyze.

        Args:
            query: User's legal query / case description.
            k_cases: Number of similar cases to retrieve.
            k_statutes: Number of statute sections to retrieve.
            include_strategy: Whether to generate strategy synthesis.
            documents_context: Formatted text from uploaded case documents.

        Returns a comprehensive legal analysis dict.
        """
        t0 = time.perf_counter()

        # 1. Understand
        understanding = self.understand_query(query)
        case_type = understanding.get("case_type", "Unknown")
        legal_issues = understanding.get("legal_issues", [])
        relevant_acts = understanding.get("relevant_acts", [])
        party_role = understanding.get("party_role", "unknown")
        relief_sought = understanding.get("relief_sought", "unknown")

        # 2. Predict
        prediction = self.predict_outcome(query, case_type)

        # 3. Search similar cases
        similar_cases = self.find_similar_cases(query, k=k_cases)

        # 4. Find statutes
        statute_query = query + " " + " ".join(legal_issues)
        statute_context = self.find_relevant_statutes(
            statute_query, acts=relevant_acts, k=k_statutes
        )

        # 5. LLM analysis
        case_context = self._format_case_context(similar_cases)
        ml_prediction = self._format_ml_prediction(prediction)

        analysis_prompt = CASE_ANALYSIS_PROMPT.format(
            query=query,
            case_type=case_type,
            legal_issues=", ".join(legal_issues) if legal_issues else "To be determined",
            party_role=party_role,
            relief_sought=relief_sought,
            statute_context=statute_context,
            case_context=case_context,
            ml_prediction=ml_prediction,
        )

        # Inject uploaded documents context if present
        if documents_context:
            analysis_prompt += (
                f"\n\nUPLOADED CASE DOCUMENTS:\n"
                f"The user has uploaded the following documents related to their case. "
                f"Consider these as primary evidence and facts:\n\n"
                f"{documents_context}"
            )

        # ── Elite Intelligence Injections ────────────────────────────────
        # 5a. Evidence admissibility check
        evidence_context = ""
        if self._evidence_checker:
            full_text = query + " " + documents_context
            evidence_issues = self._evidence_checker.check(full_text)
            if evidence_issues:
                evidence_context = self._evidence_checker.format_for_context(evidence_issues)
                analysis_prompt += f"\n\n{evidence_context}"

        # 5b. Cross-reference old ↔ new codes
        cross_ref_context = ""
        sections_found = understanding.get("relevant_sections", [])
        if self._cross_ref and sections_found:
            # Build section text list for cross-reference
            section_texts = [f"Section {s} IPC" for s in sections_found if s.isdigit()]
            cross_ref_context = self._cross_ref.format_for_context(section_texts)
            if cross_ref_context:
                analysis_prompt += f"\n\n{cross_ref_context}"

        # 5c. Procedural checklist
        checklist_context = ""
        if self._checklist_engine:
            checklist_context = self._checklist_engine.format_for_context(case_type)
            if checklist_context:
                analysis_prompt += f"\n\nPROCEDURAL GUIDANCE:\n{checklist_context}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
            {"role": "user", "content": analysis_prompt},
        ]
        analysis_text = self.llm.chat(messages)

        # 6. Strategy (optional)
        strategy_text = None
        if include_strategy:
            strategy_prompt = STRATEGY_SYNTHESIS_PROMPT.format(analysis=analysis_text)
            strategy_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
                {"role": "user", "content": strategy_prompt},
            ]
            strategy_text = self.llm.chat(strategy_messages)

        elapsed = time.perf_counter() - t0

        return {
            "query": query,
            "understanding": understanding,
            "prediction": {
                "judgment": prediction.get("judgment"),
                "confidence": prediction.get("confidence"),
                "all_probabilities": prediction.get("all_probabilities", {}),
            },
            "similar_cases": similar_cases,
            "statute_context": statute_context,
            "analysis": analysis_text,
            "strategy": strategy_text,
            "evidence_alerts": evidence_context,
            "cross_references": cross_ref_context,
            "procedural_checklist": checklist_context,
            "analysis": analysis_text,
            "strategy": strategy_text,
            "metadata": {
                "model_used": self.llm.model,
                "cases_searched": len(similar_cases),
                "processing_time_seconds": round(elapsed, 2),
            },
        }

    # ------------------------------------------------------------------
    # RAG-powered answer (lighter than full analysis)
    # ------------------------------------------------------------------
    def rag_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Retrieval-Augmented Generation: search cases → LLM generates answer.

        Lighter-weight than full analyze() — no ML prediction or strategy synthesis.
        Good for quick legal Q&A.
        """
        # Retrieve
        cases = self.find_similar_cases(question, k=k)
        case_context = self._format_case_context(cases)

        # Also search statutes
        statute_context = self.find_relevant_statutes(question, k=5)

        prompt = (
            f"Answer the following legal question based on the provided Indian case law "
            f"and statutory provisions.\n\n"
            f"QUESTION: {question}\n\n"
            f"RELEVANT CASES:\n{case_context}\n\n"
            f"RELEVANT STATUTES:\n{statute_context}\n\n"
            f"Provide a clear, well-cited answer. Reference specific cases and sections."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
            {"role": "user", "content": prompt},
        ]
        answer = self.llm.chat(messages)

        # Outcome distribution from retrieved cases
        outcome_counts: Dict[str, int] = {}
        for c in cases:
            o = c.get("outcome", "Unknown")
            outcome_counts[o] = outcome_counts.get(o, 0) + 1

        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "title": c.get("title"),
                    "url": c.get("url"),
                    "outcome": c.get("outcome"),
                    "relevance": c.get("score"),
                }
                for c in cases[:5]
            ],
            "outcome_distribution": outcome_counts,
            "num_documents": len(cases),
            "mode": "rag_llm",
        }

    # ------------------------------------------------------------------
    # Follow-up on existing analysis (per-case chat)
    # ------------------------------------------------------------------
    def follow_up(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """Answer a follow-up question given conversation history."""
        from ai_court.llm.prompts import FOLLOW_UP_PROMPT

        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in history[-10:]  # Keep last 10 messages for context
        )
        prompt = FOLLOW_UP_PROMPT.format(history=history_text, query=query)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat(messages)

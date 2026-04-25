"""
Court Document Generator — Draft court-ready legal documents.
==============================================================

Generates:
- Bail applications (regular, anticipatory)
- Written statements
- Appeals (criminal, civil)
- Counter-affidavits
- Legal notices
- Petitions (writ, criminal, civil)
- Case summaries for court presentation

Uses LLM with structured legal prompts to generate properly formatted
Indian court documents following established legal formatting conventions.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Document Templates ────────────────────────────────────────────────────

DOCUMENT_TYPES = {
    "bail_application": {
        "title": "Application for Bail",
        "sections": ["header", "court_details", "case_details", "grounds", "prayer", "verification"],
    },
    "anticipatory_bail": {
        "title": "Application for Anticipatory Bail under Section 438 CrPC / Section 482 BNSS",
        "sections": ["header", "court_details", "apprehension_details", "grounds", "prayer", "verification"],
    },
    "written_statement": {
        "title": "Written Statement",
        "sections": ["header", "preliminary_objections", "parawise_reply", "additional_pleas", "prayer"],
    },
    "criminal_appeal": {
        "title": "Criminal Appeal",
        "sections": ["header", "facts", "grounds_of_appeal", "prayer"],
    },
    "civil_appeal": {
        "title": "Civil Appeal",
        "sections": ["header", "facts", "grounds_of_appeal", "prayer"],
    },
    "writ_petition": {
        "title": "Writ Petition under Article 226/32 of the Constitution of India",
        "sections": ["header", "facts", "questions_of_law", "grounds", "prayer"],
    },
    "legal_notice": {
        "title": "Legal Notice",
        "sections": ["header", "facts", "legal_basis", "demand", "consequences"],
    },
    "counter_affidavit": {
        "title": "Counter Affidavit",
        "sections": ["header", "preliminary_submissions", "parawise_reply", "prayer", "verification"],
    },
    "case_summary": {
        "title": "Case Summary for Court Presentation",
        "sections": ["case_overview", "key_facts", "legal_issues", "applicable_law", "arguments", "precedents", "conclusion"],
    },
    "arguments_brief": {
        "title": "Brief of Arguments",
        "sections": ["introduction", "statement_of_facts", "issues", "arguments", "authorities_cited", "prayer"],
    },
}

COURT_DOC_SYSTEM_PROMPT = """You are an expert Indian legal document drafter with decades of experience 
in Indian courts. You draft court documents that are:

1. Properly formatted following Indian court conventions
2. Legally precise with correct section citations (IPC/BNS, CrPC/BNSS, CPC, Evidence Act/BSA)
3. Persuasive but factual — never misrepresent facts
4. Complete with all required sections (header, body, prayer, verification)
5. Use proper legal language and terminology
6. Include relevant case law citations where applicable

CRITICAL RULES:
- Use ONLY facts provided by the user. Never fabricate facts.
- Cite specific sections of law that apply.
- Include relevant precedents from the case analysis if provided.
- Follow the format conventions of Indian courts.
- Include proper party descriptions, court details, and case numbers.
- End with appropriate prayer/relief sought.
- Include verification/affirmation where required.

Output the document in clean, professional format with proper headings,
numbered paragraphs, and legal formatting."""

DRAFT_PROMPT = """Draft a {doc_type_title} based on the following case details:

CASE INFORMATION:
{case_info}

CASE ANALYSIS (from AI agent):
{analysis_summary}

APPLICABLE LAWS:
{statute_context}

RELEVANT PRECEDENTS:
{precedents}

UPLOADED DOCUMENTS CONTEXT:
{document_context}

USER INSTRUCTIONS:
{user_instructions}

Generate the complete {doc_type_title} with all required sections: {sections}.
Use proper Indian court formatting. Number all paragraphs. 
Include specific section citations and case law references.
The document should be ready to file in an Indian court."""


@dataclass
class GeneratedDocument:
    """A generated court document."""
    doc_type: str
    title: str
    content: str  # Full document text
    case_id: Optional[str] = None
    generated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"


class CourtDocumentGenerator:
    """
    Generates court-ready legal documents using LLM.
    
    Integrates case analysis, statute context, precedents, and uploaded
    document context to produce properly formatted Indian court documents.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM client for document generation.
        """
        self.llm_client = llm_client

    def list_document_types(self) -> List[Dict[str, str]]:
        """List available document types."""
        return [
            {"id": k, "title": v["title"], "sections": v["sections"]}
            for k, v in DOCUMENT_TYPES.items()
        ]

    def generate(
        self,
        doc_type: str,
        case_info: str,
        analysis_summary: str = "",
        statute_context: str = "",
        precedents: str = "",
        document_context: str = "",
        user_instructions: str = "",
        case_id: str = None,
    ) -> GeneratedDocument:
        """
        Generate a court document.
        
        Args:
            doc_type: One of DOCUMENT_TYPES keys.
            case_info: User's case facts and details.
            analysis_summary: Agent's case analysis (from pipeline).
            statute_context: Relevant statutes formatted for context.
            precedents: Similar cases and precedents.
            document_context: Text from uploaded documents.
            user_instructions: Any specific instructions from user.
            case_id: Associated case ID.
        
        Returns:
            GeneratedDocument with the drafted content.
        """
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type: {doc_type}. Available: {list(DOCUMENT_TYPES.keys())}")

        template = DOCUMENT_TYPES[doc_type]

        prompt = DRAFT_PROMPT.format(
            doc_type_title=template["title"],
            case_info=case_info or "[No case info provided]",
            analysis_summary=analysis_summary or "[No analysis available]",
            statute_context=statute_context or "[No statute context]",
            precedents=precedents or "[No precedents provided]",
            document_context=document_context or "[No uploaded documents]",
            user_instructions=user_instructions or "Generate a complete, court-ready document.",
            sections=", ".join(template["sections"]),
        )

        messages = [
            {"role": "system", "content": COURT_DOC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"Generating court document: {template['title']}")

        try:
            content = self.llm_client.chat(messages, temperature=0.3, max_tokens=4096)
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            raise RuntimeError(f"Failed to generate document: {e}")

        return GeneratedDocument(
            doc_type=doc_type,
            title=template["title"],
            content=content.strip(),
            case_id=case_id,
            metadata={
                "doc_type": doc_type,
                "sections": template["sections"],
                "has_analysis": bool(analysis_summary),
                "has_precedents": bool(precedents),
                "has_documents": bool(document_context),
            },
        )

    def generate_case_brief(
        self,
        case_info: str,
        analysis: Dict[str, Any],
        documents_context: str = "",
    ) -> GeneratedDocument:
        """
        Generate a complete case brief — the primary "AI Lawyer" output.
        
        This is the master document combining everything the agent knows
        about the case into a court-presentable brief.
        """
        # Build rich context from the analysis
        analysis_parts = []
        if analysis.get("prediction"):
            analysis_parts.append(f"Predicted Outcome: {analysis['prediction']} (Confidence: {analysis.get('confidence', 'N/A')})")
        if analysis.get("analysis"):
            analysis_parts.append(f"Legal Analysis:\n{analysis['analysis']}")
        if analysis.get("strategy"):
            analysis_parts.append(f"Strategy:\n{analysis['strategy']}")

        precedent_parts = []
        for case in analysis.get("similar_cases", []):
            precedent_parts.append(f"- {case.get('title', 'Case')}: {case.get('outcome', '')}")

        return self.generate(
            doc_type="arguments_brief",
            case_info=case_info,
            analysis_summary="\n".join(analysis_parts),
            statute_context=analysis.get("statute_context", ""),
            precedents="\n".join(precedent_parts) if precedent_parts else "",
            document_context=documents_context,
            user_instructions="Generate a comprehensive brief of arguments that can be presented in court. Include all relevant law, precedents, and arguments both for and against.",
        )

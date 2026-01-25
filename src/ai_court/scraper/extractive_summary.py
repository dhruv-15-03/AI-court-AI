"""Local extractive summarization utilities.

Provides HuggingFace API-free text summarization using rule-based extraction.
Designed for legal case documents from Indian Kanoon.

Zero API cost, works offline, memory-efficient.
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_judgment_section(text: str) -> str:
    """Extract the judgment/order section from legal text.
    
    Searches for common legal judgment markers and extracts relevant portions.
    
    Args:
        text: Full case text
        
    Returns:
        Extracted judgment section or last paragraphs as fallback
    """
    if not text or text == "Judgment text not found":
        return "No judgment text available"
    
    # Patterns for judgment sections (order of priority)
    judgment_patterns = [
        # Strong markers
        r'(?:ORDER|JUDGMENT|CONCLUSION|HELD|THEREFORE|RESULT)(.*?)(?:\n\n|$)',
        # Common legal phrases
        r'(?:we\s+hold|we\s+direct|it\s+is\s+ordered|it\s+is\s+hereby|accordingly|'
        r'the\s+appeal\s+is|the\s+petition\s+is|we\s+find\s+that|'
        r'for\s+the\s+reasons\s+stated|in\s+the\s+result)(.*?)(?:\n\n|$)',
        # Conclusion markers
        r'(?:for\s+the\s+foregoing\s+reasons|in\s+view\s+of\s+the\s+above|'
        r'in\s+the\s+light\s+of\s+the\s+above|considering\s+the\s+above)(.*?)(?:\n\n|$)',
        # Disposition markers
        r'(?:the\s+appeal\s+(?:is|stands)\s+(?:dismissed|allowed)|'
        r'the\s+petition\s+(?:is|stands)\s+(?:dismissed|allowed)|'
        r'the\s+writ\s+petition\s+is)(.*?)(?:\n\n|$)',
    ]
    
    judgment_texts = []
    for pattern in judgment_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            extracted = match.group(0).strip()
            if len(extracted) > 50:  # Avoid tiny fragments
                judgment_texts.append(extracted)
    
    if judgment_texts:
        # Return last 3 matches (usually the actual judgment is at the end)
        final_judgment = " ".join(judgment_texts[-3:])
        return final_judgment[:2000]  # Limit length
    
    # Fallback: last meaningful paragraphs
    lines = text.split("\n")
    last_lines = [line.strip() for line in lines[-30:] if line.strip()][-15:]
    return " ".join(last_lines)[:1500]


def extract_key_holdings(text: str, max_holdings: int = 5) -> List[str]:
    """Extract key legal holdings from the text.
    
    Args:
        text: Full case text
        max_holdings: Maximum number of holdings to extract
        
    Returns:
        List of key holding sentences
    """
    holdings = []
    
    # Patterns that typically precede key holdings
    holding_patterns = [
        r'(?:held\s+that|we\s+hold\s+that|it\s+is\s+held\s+that|'
        r'the\s+court\s+held|this\s+court\s+holds)(.*?[.!?])',
        r'(?:we\s+are\s+of\s+the\s+view|in\s+our\s+view|'
        r'we\s+are\s+of\s+the\s+opinion)(.*?[.!?])',
        r'(?:it\s+is\s+well\s+settled|the\s+settled\s+position)(.*?[.!?])',
    ]
    
    for pattern in holding_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            holding = match.strip() if isinstance(match, str) else match[0].strip()
            if 20 < len(holding) < 500:
                holdings.append(holding)
    
    # Deduplicate and limit
    seen = set()
    unique_holdings = []
    for h in holdings:
        h_normalized = h.lower()[:100]
        if h_normalized not in seen:
            seen.add(h_normalized)
            unique_holdings.append(h)
    
    return unique_holdings[:max_holdings]


def extract_parties(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract appellant/petitioner and respondent names.
    
    Args:
        text: Full case text
        
    Returns:
        Tuple of (appellant/petitioner, respondent) or (None, None)
    """
    appellant = None
    respondent = None
    
    # Common patterns
    appellant_patterns = [
        r'(?:appellant|petitioner)[:\s]+([A-Z][^,\n]+)',
        r'([A-Z][^,\n]+)\s+(?:\.\.\.\s*)?(?:Appellant|Petitioner)',
    ]
    respondent_patterns = [
        r'(?:respondent)[:\s]+([A-Z][^,\n]+)',
        r'([A-Z][^,\n]+)\s+(?:\.\.\.\s*)?(?:Respondent)',
    ]
    
    for pattern in appellant_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            appellant = match.group(1).strip()[:100]
            break
    
    for pattern in respondent_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            respondent = match.group(1).strip()[:100]
            break
    
    return appellant, respondent


def extract_citations(text: str, max_citations: int = 5) -> List[str]:
    """Extract legal citations from the text.
    
    Args:
        text: Full case text
        max_citations: Maximum number of citations to extract
        
    Returns:
        List of citation strings
    """
    citations = []
    
    # Common Indian case citation patterns
    patterns = [
        r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',  # (2020) 5 SCC 123
        r'AIR\s+\d{4}\s+SC\s+\d+',  # AIR 2020 SC 123
        r'\d{4}\s+\(\d+\)\s+SCR\s+\d+',  # 2020 (5) SCR 123
        r'\(\d{4}\)\s+\d+\s+SCR\s+\d+',  # (2020) 5 SCR 123
        r'[\w\s]+\s+v\.?\s*(?:State|Union)\s+of\s+\w+',  # X v. State of Y
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    
    # Deduplicate
    seen = set()
    unique = []
    for c in citations:
        c_norm = c.lower().strip()
        if c_norm not in seen:
            seen.add(c_norm)
            unique.append(c.strip())
    
    return unique[:max_citations]


def create_extractive_summary(
    text: str,
    target_length: int = 1500,
    include_holdings: bool = True
) -> str:
    """Create a comprehensive extractive summary of a legal case.
    
    This is the main entry point for local summarization, replacing
    HuggingFace API calls for zero-cost operation.
    
    Args:
        text: Full case text
        target_length: Approximate target length of summary
        include_holdings: Whether to include key holdings section
        
    Returns:
        Extractive summary string
    """
    if not text or len(text) < 100:
        return text or "No text available"
    
    parts = []
    
    # 1. Introduction (first few sentences)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    intro_sentences = sentences[:5]
    intro = ' '.join(intro_sentences)
    if intro:
        parts.append(intro)
    
    # 2. Key holdings (if requested)
    if include_holdings:
        holdings = extract_key_holdings(text, max_holdings=3)
        if holdings:
            holdings_text = "Key Holdings: " + "; ".join(holdings)
            parts.append(holdings_text)
    
    # 3. Judgment section
    judgment = extract_judgment_section(text)
    if judgment and judgment != "No judgment text available":
        parts.append(f"[...]\n\nJudgment: {judgment}")
    
    # 4. Combine and truncate
    summary = "\n\n".join(parts)
    
    if len(summary) > target_length:
        summary = summary[:target_length - 3] + "..."
    
    return summary


def get_outcome_indicators(text: str) -> List[str]:
    """Extract words/phrases that indicate case outcome.
    
    Useful for understanding why a particular outcome was predicted.
    
    Args:
        text: Case text (can be summary or full text)
        
    Returns:
        List of outcome indicator phrases found
    """
    indicators = []
    
    outcome_phrases = [
        # Positive for appellant
        "appeal allowed", "petition allowed", "writ allowed",
        "acquitted", "bail granted", "conviction set aside",
        "charges quashed", "relief granted",
        # Negative for appellant
        "appeal dismissed", "petition dismissed", "bail denied",
        "conviction upheld", "appeal fails", "no merit",
        "relief denied", "bail rejected",
        # Neutral/procedural
        "remanded", "sent back", "partly allowed",
        "directions issued", "modified",
    ]
    
    text_lower = text.lower()
    for phrase in outcome_phrases:
        if phrase in text_lower:
            indicators.append(phrase)
    
    return indicators


__all__ = [
    'extract_judgment_section',
    'extract_key_holdings', 
    'extract_parties',
    'extract_citations',
    'create_extractive_summary',
    'get_outcome_indicators',
]

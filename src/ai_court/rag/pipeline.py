"""RAG Pipeline for Legal Case Retrieval.

Provides retrieval-augmented responses using existing TF-IDF search index.
No LLM generation - retrieval-only mode for zero cost operation.

Contract:
- retrieve(query) -> List[Doc]: Find relevant case documents
- augment(query, docs) -> context: Format documents as context
- generate(query, context) -> response: Format final response (no LLM)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    search_index: Optional[Dict[str, Any]] = None,
    preprocess_fn: Optional[callable] = None,
    k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve relevant documents using TF-IDF search index.
    
    Args:
        query: User's search query
        search_index: Pre-loaded search index with vectorizer, matrix, and meta
        preprocess_fn: Text preprocessing function
        k: Number of documents to retrieve
        
    Returns:
        List of document dicts with title, url, snippet, outcome, and score
    """
    if search_index is None:
        logger.warning("No search index available for retrieval")
        return []
    
    try:
        vectorizer = search_index.get('vectorizer')
        matrix = search_index.get('matrix')
        meta = search_index.get('meta', [])
        
        if vectorizer is None or matrix is None:
            logger.warning("Search index missing vectorizer or matrix")
            return []
        
        # Preprocess query
        processed_query = preprocess_fn(query) if preprocess_fn else query.lower()
        
        # Transform and compute similarity
        query_vector = vectorizer.transform([processed_query])
        scores = (matrix @ query_vector.T).toarray().ravel()
        
        # Get top-k indices
        top_indices = np.argsort(-scores)[:k]
        
        documents = []
        for idx in top_indices:
            if idx < len(meta) and scores[idx] > 0:
                m = meta[idx]
                documents.append({
                    'title': m.get('title', 'Unknown'),
                    'url': m.get('url'),
                    'snippet': m.get('snippet', ''),
                    'outcome': m.get('outcome'),
                    'score': float(scores[idx]),
                    'rank': len(documents) + 1
                })
        
        return documents
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []


def augment(
    query: str,
    docs: List[Dict[str, Any]],
    max_context_length: int = 4000
) -> str:
    """Format retrieved documents as context for the response.
    
    Args:
        query: Original user query
        docs: Retrieved documents from retrieve()
        max_context_length: Maximum context length in characters
        
    Returns:
        Formatted context string
    """
    if not docs:
        return ""
    
    context_parts = []
    total_length = 0
    
    for i, doc in enumerate(docs, 1):
        title = doc.get('title', 'Unknown Case')
        outcome = doc.get('outcome', 'Unknown')
        snippet = doc.get('snippet', '')
        url = doc.get('url', '')
        score = doc.get('score', 0)
        
        # Format document entry
        entry = f"[{i}] {title}\n"
        entry += f"    Outcome: {outcome}\n"
        if url:
            entry += f"    Source: {url}\n"
        entry += f"    Relevance: {score:.2f}\n"
        entry += f"    Excerpt: {snippet[:300]}...\n"
        
        if total_length + len(entry) > max_context_length:
            break
            
        context_parts.append(entry)
        total_length += len(entry)
    
    header = f"Found {len(context_parts)} relevant cases for: \"{query}\"\n"
    header += "=" * 50 + "\n\n"
    
    return header + "\n".join(context_parts)


def generate(
    query: str,
    context: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate response based on retrieved context.
    
    Note: This is retrieval-only mode - no LLM generation.
    Returns formatted response with citations.
    
    Args:
        query: Original user query
        context: Formatted context from augment()
        documents: Original document list
        
    Returns:
        Response dict with answer, citations, and metadata
    """
    if not documents:
        return {
            "answer": "No relevant cases found for your query. Please try different search terms.",
            "confidence": 0.0,
            "citations": [],
            "num_documents": 0,
            "mode": "retrieval_only"
        }
    
    # Analyze retrieved documents for patterns
    outcomes = [d.get('outcome') for d in documents if d.get('outcome')]
    outcome_counts = {}
    for o in outcomes:
        outcome_counts[o] = outcome_counts.get(o, 0) + 1
    
    # Find most common outcome
    most_common_outcome = max(outcome_counts.items(), key=lambda x: x[1])[0] if outcome_counts else None
    
    # Build answer
    answer_parts = []
    answer_parts.append(f"Based on {len(documents)} similar cases found in the database:")
    
    if most_common_outcome:
        outcome_pct = (outcome_counts[most_common_outcome] / len(documents)) * 100
        answer_parts.append(f"\n• Most common outcome: {most_common_outcome} ({outcome_pct:.0f}% of similar cases)")
    
    answer_parts.append(f"\n• Top match: \"{documents[0].get('title', 'Unknown')}\" (relevance: {documents[0].get('score', 0):.2f})")
    
    # Add citations
    citations = []
    for doc in documents[:3]:
        citations.append({
            "title": doc.get('title'),
            "url": doc.get('url'),
            "outcome": doc.get('outcome'),
            "relevance": doc.get('score')
        })
    
    return {
        "answer": " ".join(answer_parts),
        "confidence": float(documents[0].get('score', 0)) if documents else 0.0,
        "citations": citations,
        "num_documents": len(documents),
        "outcome_distribution": outcome_counts,
        "mode": "retrieval_only",
        "context": context[:2000] if context else None  # Truncate for response size
    }


def rag_query(
    question: str,
    search_index: Optional[Dict[str, Any]] = None,
    preprocess_fn: Optional[callable] = None,
    k: int = 5
) -> Dict[str, Any]:
    """Complete RAG pipeline: retrieve, augment, generate.
    
    Convenience function that chains all pipeline steps.
    
    Args:
        question: User's question
        search_index: Pre-loaded search index
        preprocess_fn: Text preprocessing function
        k: Number of documents to retrieve
        
    Returns:
        Complete RAG response
    """
    # Step 1: Retrieve
    documents = retrieve(question, search_index, preprocess_fn, k)
    
    # Step 2: Augment
    context = augment(question, documents)
    
    # Step 3: Generate
    response = generate(question, context, documents)
    response['question'] = question
    
    return response


__all__ = ['retrieve', 'augment', 'generate', 'rag_query']

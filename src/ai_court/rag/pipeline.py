"""RAG Pipeline for Legal Case Retrieval-Augmented Generation.

Provides retrieval-augmented responses using TF-IDF search index + LLM generation.

Contract:
- retrieve(query) -> List[Doc]: Find relevant case documents
- augment(query, docs) -> context: Format documents as context
- generate(query, context, docs, llm_client) -> response: LLM-generated or statistical fallback
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ai_court.llm.faithfulness import verify_citations

logger = logging.getLogger(__name__)


def _lexical_retrieve(
    query: str,
    search_index: Optional[Dict[str, Any]],
    preprocess_fn: Optional[Callable[[str], str]],
    k: int,
) -> List[Dict[str, Any]]:
    """TF-IDF retrieval leg (the historical behaviour)."""
    if search_index is None:
        return []
    try:
        vectorizer = search_index.get('vectorizer')
        matrix = search_index.get('matrix')
        meta = search_index.get('meta', [])
        if vectorizer is None or matrix is None:
            logger.warning("Search index missing vectorizer or matrix")
            return []

        processed_query = preprocess_fn(query) if preprocess_fn else query.lower()
        query_vector = vectorizer.transform([processed_query])
        scores = (matrix @ query_vector.T).toarray().ravel()
        top_indices = np.argsort(-scores)[:k]

        documents: List[Dict[str, Any]] = []
        for idx in top_indices:
            if idx < len(meta) and scores[idx] > 0:
                m = meta[idx]
                documents.append({
                    'title': m.get('title', 'Unknown'),
                    'url': m.get('url'),
                    'snippet': m.get('snippet', ''),
                    'outcome': m.get('outcome'),
                    'score': float(scores[idx]),
                    'rank': len(documents) + 1,
                })
        return documents
    except Exception as e:
        logger.error(f"Lexical retrieval failed: {e}")
        return []


def _semantic_retrieve(
    query: str,
    semantic_index: Optional[Dict[str, Any]],
    query_embed_fn: Optional[Callable[[str], Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Dense/semantic retrieval leg using a pre-built embedding index.

    ``query_embed_fn`` embeds the raw query (it applies its own preprocessing)
    and returns an array whose first row is the query vector, matching the
    convention used by the /api/search endpoint.
    """
    if semantic_index is None or query_embed_fn is None:
        return []
    try:
        dense = semantic_index.get('embeddings')
        meta = semantic_index.get('meta', [])
        if dense is None or not hasattr(dense, 'shape'):
            return []
        qv = query_embed_fn(query)
        if qv is None or not hasattr(qv, '__getitem__'):
            return []
        sims = np.dot(dense, qv[0])
        top_indices = np.argsort(-sims)[:k]

        documents: List[Dict[str, Any]] = []
        for idx in top_indices:
            if idx < len(meta):
                m = meta[idx]
                documents.append({
                    'title': m.get('title', 'Unknown'),
                    'url': m.get('url'),
                    'snippet': m.get('snippet', ''),
                    'outcome': m.get('outcome'),
                    'score': float(sims[idx]),
                    'rank': len(documents) + 1,
                })
        return documents
    except Exception as e:
        logger.error(f"Semantic retrieval failed: {e}")
        return []


def retrieve(
    query: str,
    search_index: Optional[Dict[str, Any]] = None,
    preprocess_fn: Optional[Callable[[str], str]] = None,
    k: int = 5,
    semantic_index: Optional[Dict[str, Any]] = None,
    query_embed_fn: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant documents.

    Fuses dense semantic search with TF-IDF via reciprocal-rank fusion when a
    semantic index *and* an embedding function are supplied; otherwise falls back
    to whichever single retriever is available. The result shape is identical in
    all three cases, so downstream augment()/generate() are unaffected.

    Args:
        query: User's search query.
        search_index: Pre-loaded TF-IDF index (vectorizer, matrix, meta).
        preprocess_fn: Text preprocessing function for the lexical leg.
        k: Number of documents to retrieve.
        semantic_index: Optional dense index (embeddings, meta).
        query_embed_fn: Optional callable that embeds the query for the dense leg.

    Returns:
        List of document dicts with title, url, snippet, outcome, score (+rank).
    """
    lexical = _lexical_retrieve(query, search_index, preprocess_fn, k)
    semantic = _semantic_retrieve(query, semantic_index, query_embed_fn, k)

    if semantic and lexical:
        # Both legs available -> hybrid. Reuse the shared RRF utility. Each fused
        # doc keeps its original retriever 'score' (used downstream as a rough
        # confidence) and gains a 'fusion_score'; ordering is by fusion_score.
        from ai_court.retrieval.hybrid import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(semantic=semantic, lexical=lexical, k=k)
        for rank, doc in enumerate(fused, 1):
            doc['rank'] = rank
        return fused

    return (semantic or lexical)[:k]


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
    documents: List[Dict[str, Any]],
    llm_client: Any = None,
    statute_corpus: Any = None,
) -> Dict[str, Any]:
    """Generate response based on retrieved context.
    
    If llm_client is provided, uses LLM for intelligent generation.
    Otherwise falls back to statistical retrieval-only mode.
    
    Args:
        query: Original user query
        context: Formatted context from augment()
        documents: Original document list
        llm_client: Optional LLMClient instance for LLM generation
        statute_corpus: Optional StatuteCorpus for statutory provisions
        
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
    
    # Build citations
    citations = []
    for doc in documents[:5]:
        citations.append({
            "title": doc.get('title'),
            "url": doc.get('url'),
            "outcome": doc.get('outcome'),
            "relevance": doc.get('score')
        })
    
    # Outcome distribution
    outcomes: List[str] = [str(d.get('outcome')) for d in documents if d.get('outcome')]
    outcome_counts: Dict[str, int] = {}
    for o in outcomes:
        outcome_counts[o] = outcome_counts.get(o, 0) + 1
    
    # --- LLM Generation Mode ---
    if llm_client is not None:
        try:
            from ai_court.llm.prompts import SYSTEM_PROMPT_LEGAL_AGENT
            
            # Build statute context if available
            statute_text = ""
            if statute_corpus is not None and statute_corpus.loaded:
                try:
                    sections = statute_corpus.search_sections(query, k=5)
                    statute_text = statute_corpus.format_for_context(sections, max_length=2000)
                except Exception:
                    statute_text = ""
            
            prompt = (
                f"Answer the following legal question using the provided Indian case law"
                f"{' and statutory provisions' if statute_text else ''}.\n\n"
                f"QUESTION: {query}\n\n"
                f"RELEVANT CASES:\n{context}\n\n"
            )
            if statute_text:
                prompt += f"RELEVANT STATUTES:\n{statute_text}\n\n"
            prompt += (
                "Provide a clear, well-cited answer. Cite ONLY the case names listed "
                "under RELEVANT CASES above and the statutory sections provided — do "
                "NOT introduce any case that does not appear in that list. If the "
                "provided material is insufficient to answer, say so explicitly rather "
                "than citing cases from memory. Structure your response with clear sections."
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_LEGAL_AGENT},
                {"role": "user", "content": prompt},
            ]
            answer = llm_client.chat(messages)

            # Post-hoc citation faithfulness check: flag any case the model cited
            # that is NOT among the retrieved documents (legal AI's #1 liability).
            faith = verify_citations(answer, documents)
            if not faith["grounded"]:
                logger.warning(
                    "Ungrounded citation(s) in RAG answer: %s",
                    faith["unverified_citations"],
                )
                answer = (
                    answer
                    + "\n\n> \u26a0\ufe0f **Citation caution:** the following case "
                    "reference(s) could not be matched to the retrieved source "
                    "documents and may be inaccurate \u2014 verify before relying on "
                    "them: " + "; ".join(faith["unverified_citations"])
                )

            return {
                "answer": answer,
                "confidence": float(documents[0].get('score', 0)) if documents else 0.0,
                "citations": citations,
                "unverified_citations": faith["unverified_citations"],
                "grounded": faith["grounded"],
                "num_documents": len(documents),
                "outcome_distribution": outcome_counts,
                "mode": "rag_llm",
                "context": context[:2000] if context else None,
            }
        except Exception as exc:
            logger.warning("LLM generation failed, falling back to retrieval-only: %s", exc)
            # Fall through to statistical mode
    
    # --- Statistical Fallback Mode (no LLM) ---
    most_common_outcome = max(outcome_counts.items(), key=lambda x: x[1])[0] if outcome_counts else None
    
    answer_parts = []
    answer_parts.append(f"Based on {len(documents)} similar cases found in the database:")
    
    if most_common_outcome:
        outcome_pct = (outcome_counts[most_common_outcome] / len(documents)) * 100
        answer_parts.append(f"\n• Most common outcome: {most_common_outcome} ({outcome_pct:.0f}% of similar cases)")
    
    answer_parts.append(f"\n• Top match: \"{documents[0].get('title', 'Unknown')}\" (relevance: {documents[0].get('score', 0):.2f})")
    
    return {
        "answer": " ".join(answer_parts),
        "confidence": float(documents[0].get('score', 0)) if documents else 0.0,
        "citations": citations,
        "num_documents": len(documents),
        "outcome_distribution": outcome_counts,
        "mode": "retrieval_only",
        "context": context[:2000] if context else None,
    }


def rag_query(
    question: str,
    search_index: Optional[Dict[str, Any]] = None,
    preprocess_fn: Optional[Callable[[str], str]] = None,
    k: int = 5,
    llm_client: Any = None,
    statute_corpus: Any = None,
    semantic_index: Optional[Dict[str, Any]] = None,
    query_embed_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Complete RAG pipeline: retrieve, augment, generate.
    
    Convenience function that chains all pipeline steps.
    Uses LLM for generation when llm_client is provided, and hybrid
    (dense + TF-IDF) retrieval when a semantic index and embedding function
    are supplied.
    
    Args:
        question: User's question
        search_index: Pre-loaded TF-IDF search index
        preprocess_fn: Text preprocessing function
        k: Number of documents to retrieve
        llm_client: Optional LLMClient for LLM-powered answers
        statute_corpus: Optional StatuteCorpus for statutory provisions
        semantic_index: Optional dense index for hybrid retrieval
        query_embed_fn: Optional query embedding callable for hybrid retrieval
        
    Returns:
        Complete RAG response
    """
    # Step 1: Retrieve (hybrid when a semantic index is available)
    documents = retrieve(
        question,
        search_index,
        preprocess_fn,
        k,
        semantic_index=semantic_index,
        query_embed_fn=query_embed_fn,
    )
    
    # Step 2: Augment
    context = augment(question, documents)
    
    # Step 3: Generate (with LLM if available)
    response = generate(question, context, documents, llm_client=llm_client, statute_corpus=statute_corpus)
    response['question'] = question
    
    return response


__all__ = ['retrieve', 'augment', 'generate', 'rag_query']

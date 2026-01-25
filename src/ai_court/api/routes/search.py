import numpy as np
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError

from ai_court.api import state, dependencies, models
from ai_court.api.extensions import limiter

search_bp = Blueprint('search', __name__)

@search_bp.route("/api/search", methods=["POST"])
@limiter.limit("60/minute")
@swag_from({
    'tags': ['search'],
    'consumes': ['application/json'],
    'parameters': [{
        'name': 'body', 'in': 'body', 'required': True,
        'schema': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'k': {'type': 'integer'}}}
    }],
    'responses': {200: {'description': 'OK'}}
})
def search_cases():
    auth = dependencies.require_api_key()
    if auth:
        return auth
    if state.search_index is None and state.semantic_index is None:
        return jsonify({"error": "No search index available. Build TF-IDF or semantic index."}), 503

    raw = request.json or {}
    if not isinstance(raw, dict):
        raise BadRequest("Body must be a JSON object")
    try:
        parsed = models.SearchRequest(**raw)
    except ValidationError as ve:
        return jsonify({"error": "validation_failed", "details": ve.errors()}), 400
    query = parsed.query.strip()
    k = parsed.k

    results = []
    semantic_results = []
    lexical_results = []
    processed = state.preprocess_fn(query)
    
    if state.semantic_index is not None:
        try:
            dense = state.semantic_index.get('embeddings')
            meta_sem = state.semantic_index.get('meta', [])
            qv = dependencies.semantic_query_embeddings(processed)
            if qv is not None and hasattr(dense, 'shape'):
                sims = np.dot(dense, qv[0])
                order = np.argsort(-sims)[:k]
                for idx in order:
                    if idx < len(meta_sem):
                        m = meta_sem[idx]
                        semantic_results.append({
                            'title': m.get('title','Unknown'),
                            'url': m.get('url'),
                            'outcome': m.get('outcome'),
                            'snippet': m.get('snippet'),
                            'score': float(sims[idx])
                        })
        except Exception:
            pass
    
    if state.search_index is not None:
        vect = state.search_index["vectorizer"]
        matrix = state.search_index["matrix"]
        meta = state.search_index.get("meta", [])
        qv = vect.transform([processed])
        scores = (matrix @ qv.T).toarray().ravel()
        if scores.size > 0:
            top_idx = np.argsort(-scores)[:k]
            for idx in top_idx:
                if idx < len(meta):
                    m = meta[idx]
                    lexical_results.append({
                        "title": m.get("title", "Unknown"),
                        "url": m.get("url"),
                        "outcome": m.get("outcome"),
                        "snippet": m.get("snippet"),
                        "score": float(scores[idx])
                    })
    
    fused: list = []
    if semantic_results and lexical_results:
        k_sem = { (r.get('url'), r.get('title')): i for i,r in enumerate(semantic_results) }
        k_lex = { (r.get('url'), r.get('title')): i for i,r in enumerate(lexical_results) }
        all_keys = set(k_sem.keys()) | set(k_lex.keys())
        K = 60
        for key in all_keys:
            sem_rank = k_sem.get(key, 10**6)
            lex_rank = k_lex.get(key, 10**6)
            sem_score = 1.0 / (K + sem_rank)
            lex_score = 1.0 / (K + lex_rank)
            base = None
            for r in semantic_results:
                if (r.get('url'), r.get('title')) == key:
                    base = r
                    break
            if base is None:
                for r in lexical_results:
                    if (r.get('url'), r.get('title')) == key:
                        base = r
                        break
            if base is None:
                continue
            fused.append({**base, 'fusion_score': sem_score + lex_score})
        fused.sort(key=lambda x: -x['fusion_score'])
        results = fused[:k]
    else:
        results = (semantic_results or lexical_results)[:k]
    return jsonify({"results": results})

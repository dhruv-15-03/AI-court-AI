import numpy as np
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from werkzeug.exceptions import BadRequest
from pydantic import ValidationError

from ai_court.api import state, dependencies, models, config
from ai_court.api.extensions import limiter
from ai_court.retrieval.hybrid import reciprocal_rank_fusion
from ai_court.utils.cache import ResponseCache, make_key

search_bp = Blueprint('search', __name__)

# /api/search recomputed a full dense dot-product + full-array argsort over the
# entire corpus on every call, even for a repeated query — unlike /api/analyze,
# which already caches deterministic responses. Same cache, same invalidation
# story: keyed on a version tag so a reload/rebuild of the search indices (which
# replaces the `state.search_index`/`state.semantic_index` objects) never serves
# a stale result from before the rebuild.
search_response_cache = ResponseCache(
    enabled=config.RESPONSE_CACHE_ENABLED,
    ttl_seconds=config.RESPONSE_CACHE_TTL,
    max_size=config.RESPONSE_CACHE_MAX_SIZE,
    redis_url=config.RESPONSE_CACHE_REDIS_URL,
    key_prefix="aicache-search",
)


def _search_index_version() -> str:
    """Identify the loaded search/semantic indices so cache keys are scoped to them.

    Uses `id()` + row count of whichever indices are loaded: both are replaced
    (never mutated in place) by `ensure_search_index`, so a fresh object identity
    means a fresh index and therefore a fresh cache namespace.
    """
    parts = []
    lex = state.search_index
    if lex is not None:
        matrix = lex.get("matrix")
        rows = getattr(matrix, "shape", (None,))[0]
        parts.append(f"lex:{id(lex)}:{rows}")
    sem = state.semantic_index
    if sem is not None:
        dense = sem.get("embeddings")
        rows = getattr(dense, "shape", (None,))[0]
        parts.append(f"sem:{id(sem)}:{rows}")
    return "|".join(parts) or "none"


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-k scores, sorted descending.

    ``np.argsort(-scores)`` sorts the *entire* array (O(n log n)) just to keep
    the first k. For a corpus of any real size this dwarfs the O(n) dot-product
    that produced ``scores``. ``argpartition`` finds the top-k boundary in O(n),
    and only the k selected elements are then sorted (O(k log k)).
    """
    n = scores.shape[0]
    if k >= n:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


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
    # Lazy-load search index on first request if configured
    dependencies.ensure_search_index()
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
    outcome_filter = raw.get('outcome_filter')

    cache_key = make_key(
        "search",
        {"query": query, "k": k, "outcome_filter": outcome_filter},
        model_version=_search_index_version(),
    )
    cached = search_response_cache.get(cache_key)
    if cached is not None:
        payload = dict(cached)
        payload["cached"] = True
        return jsonify(payload)

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
                order = _top_k_indices(sims, k)
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
            top_idx = _top_k_indices(scores, k)
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
        fused = reciprocal_rank_fusion(
            semantic=semantic_results,
            lexical=lexical_results,
            k=k,
            outcome_filter=outcome_filter,
        )
        results = fused
    else:
        results = (semantic_results or lexical_results)[:k]
        if outcome_filter:
            needle = outcome_filter.lower()
            results = [r for r in results if needle in (r.get('outcome') or '').lower()]
    response_payload = {"results": results, "cached": False}
    search_response_cache.set(cache_key, response_payload)
    return jsonify(response_payload)


@search_bp.route("/api/search/cache/stats", methods=["GET"])
def search_cache_stats():
    """Diagnostics for the /api/search response cache (hits/misses/backend)."""
    auth = dependencies.require_api_key()
    if auth:
        return auth
    return jsonify(search_response_cache.stats())

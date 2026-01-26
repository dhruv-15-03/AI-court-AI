import os
import json
import dill
import logging
from typing import Dict, Any, Optional, List
from flask import request, jsonify
from ai_court.api import config, state

logger = logging.getLogger("api")

try:
    from ai_court.ontology import (
        ontology_metadata,
        load_ontology,
        flatten_leaves,
        map_coarse_label,
    )
except Exception:
    # Fallback stubs when ontology module unavailable
    def ontology_metadata() -> Dict[str, Any]:  # type: ignore[misc]
        return {}

    def load_ontology() -> Dict[str, Any]:  # type: ignore[misc]
        return {}

    def flatten_leaves(_data: Any) -> List[Any]:  # type: ignore[misc]
        return []

    def map_coarse_label(label: str) -> tuple[str, bool]:  # type: ignore[misc]
        return (label, False)

def load_model():
    _default_model_path = os.path.join(config.PROJECT_ROOT, "models", "legal_case_classifier.pkl")
    _env_model_path = config.MODEL_PATH
    
    model_path = _default_model_path
    if _env_model_path and os.path.exists(_env_model_path):
        model_path = _env_model_path
    elif os.path.exists(_default_model_path):
        model_path = _default_model_path
    else:
        _docker_path = "/app/models/legal_case_classifier.pkl"
        if os.path.exists(_docker_path):
            model_path = _docker_path
        else:
            logger.warning(f"Model file not found at {model_path} or {_docker_path}")
            # Don't raise here to allow app to start, but functionality will be broken
            return

    try:
        from ai_court.model.legal_case_classifier import LegalCaseClassifier
        state.classifier = LegalCaseClassifier()
        state.classifier.load_model(model_path)
        
        # Sync legacy state for backward compatibility
        state.classifier_model = state.classifier.model
        state.label_encoder = state.classifier.label_encoder
        state.preprocess_fn = state.classifier.preprocess_text
        
        logger.info(f"[api] Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

def load_search_index():
    low_memory = os.getenv("LOW_MEMORY", "0") == "1"
    disable_search = os.getenv("DISABLE_SEARCH_INDEX", "1" if low_memory else "0") == "1"
    
    if not disable_search and os.path.exists(config.SEARCH_INDEX_PATH):
        try:
            with open(config.SEARCH_INDEX_PATH, "rb") as f:
                state.search_index = dill.load(f)
            logger.info(f"[api] Loaded search index from {config.SEARCH_INDEX_PATH}")
        except Exception as e:
            logger.warning(f"[api] Failed to load search index: {e}")
    else:
        if disable_search:
            logger.info("[api] Skipping search index load")

def load_semantic_index():
    low_memory = os.getenv("LOW_MEMORY", "0") == "1"
    disable_semantic = os.getenv("DISABLE_SEMANTIC_INDEX", "1" if low_memory else "0") == "1"
    semantic_path = os.getenv("SEMANTIC_INDEX_PATH", os.path.join(config.PROJECT_ROOT, "models", "semantic_index.pkl"))

    if not disable_semantic and os.path.exists(semantic_path):
        try:
            with open(semantic_path, 'rb') as f:
                state.semantic_index = dill.load(f)
            logger.info(f"[api] Loaded semantic index from {semantic_path}")
        except Exception as e:
            logger.warning(f"[api] Failed to load semantic index: {e}")
    else:
        if disable_semantic:
            logger.info("[api] Skipping semantic index load")

def load_multi_axis():
    if os.getenv('ENABLE_MULTI_AXIS_SHADOW','1') != '1':
        return

    base_dir = os.path.join(config.PROJECT_ROOT,'models','multi_axis')
    prom_path = os.path.join(base_dir,'promoted.json')
    ckpt_path = os.path.join(base_dir,'multi_axis.pt')
    
    if not os.path.exists(prom_path) or not os.path.exists(ckpt_path):
        return

    try:
        import torch
        from transformers import AutoTokenizer
        with open(prom_path,'r',encoding='utf-8') as f:
            pm = json.load(f) or {}
        run_id = pm.get('run_id')
        bundle = torch.load(ckpt_path, map_location='cpu')
        from ai_court.model.multi_axis_transformer import MultiAxisModel
        model = MultiAxisModel(bundle['backbone'], {ax: len(m) for ax,m in bundle['label_maps'].items()})
        model.load_state_dict(bundle['model_state'])
        model.eval()
        tok = AutoTokenizer.from_pretrained(bundle['backbone'])
        state.multi_axis_bundle = {'model': model, 'tokenizer': tok, 'label_maps': bundle['label_maps'], 'backbone': bundle['backbone'], 'run_id': run_id}
        logger.info(f"[api] Loaded multi-axis promoted model (run {run_id})")
    except Exception as e:
        logger.warning(f"[api] multi-axis load failed: {e}")

def require_api_key():
    if config.API_KEY:
        key = request.headers.get("X-API-Key", "")
        if key != config.API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
    return None

def synthesize_body_from_answers(raw: Dict[str, Any]) -> str:
    """Extract case body text excluding case_type for cleaner API usage."""
    if not isinstance(raw, dict):
        return ""
    parts: List[str] = []
    summary = raw.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    for k, v in raw.items():
        if k in ("case_type", "summary"):
            continue
        if v is None:
            continue
        sv = str(v).strip()
        if not sv:
            continue
        kk = k.replace('_', ' ')
        parts.append(f"{kk}: {sv}")
    return ". ".join(parts).strip()

def synthesize_text_from_answers(raw: Dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        return ""
    ct = str(raw.get("case_type") or "").strip()
    body = synthesize_body_from_answers(raw)
    base = (ct + " ").strip()
    return (base + " " + body).strip()

def multi_axis_predict_single(text: str):
    b = state.multi_axis_bundle
    if not b:
        return None
    import torch
    max_len = int(os.getenv('MULTI_AXIS_INFER_MAX_LEN','384'))
    tok = b['tokenizer']
    model = b['model']
    enc = tok([text], truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'])
    inv = {ax: {v:k for k,v in b['label_maps'][ax].items()} for ax in b['label_maps']}
    pred_axes = {ax: inv[ax][int(logits[ax].argmax(dim=1).item())] for ax in b['label_maps']}
    main_judgment = pred_axes.get('relief_label') or pred_axes.get('substantive_label') or pred_axes.get('procedural_label')
    return {'axes': pred_axes, 'main_judgment': main_judgment}

def record_agreement(classical_label: str, shadow_axes: Optional[Dict[str, Any]]) -> None:
    if not shadow_axes or not isinstance(shadow_axes, dict):
        return
    unified_candidate = shadow_axes.get('relief_label') or shadow_axes.get('substantive_label') or shadow_axes.get('procedural_label')
    if unified_candidate is None:
        return
    _prev_total = state.agreement_stats.get('total_compared', 0)
    total_compared = (int(_prev_total) if isinstance(_prev_total, (int, float, str)) else 0) + 1
    state.agreement_stats['total_compared'] = total_compared
    if classical_label == unified_candidate:
        _prev_agree = state.agreement_stats.get('agreements', 0)
        state.agreement_stats['agreements'] = (int(_prev_agree) if isinstance(_prev_agree, (int, float, str)) else 0) + 1
    else:
        last_samples = state.agreement_stats.get('last_samples', [])
        if isinstance(last_samples, list) and len(last_samples) >= 10:
            last_samples.pop(0)
        if isinstance(last_samples, list):
            last_samples.append({
                'classical': classical_label,
                'multi_axis_relief': shadow_axes.get('relief_label'),
                'multi_axis_substantive': shadow_axes.get('substantive_label'),
                'multi_axis_procedural': shadow_axes.get('procedural_label')
            })
            state.agreement_stats['last_samples'] = last_samples
    
    if total_compared % 10 == 0:
        persist_agreement()

def persist_agreement():
    try:
        out = {
            'total_compared': state.agreement_stats['total_compared'],
            'agreements': state.agreement_stats['agreements'],
            'rate': (state.agreement_stats['agreements']/ state.agreement_stats['total_compared']) if state.agreement_stats['total_compared'] else None,
            'last_samples': state.agreement_stats['last_samples']
        }
        with open('agreement_stats.json','w',encoding='utf-8') as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

def semantic_query_embeddings(query: str):
    if state.semantic_index is None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    model_name = state.semantic_index.get('model_name') or 'all-MiniLM-L6-v2'
    try:
        model = SentenceTransformer(model_name)
        vec = model.encode([state.preprocess_fn(query)], normalize_embeddings=True)
        return vec
    except Exception:
        return None

# Ontology helpers
_ontology_cache = None

def ontology_tree_cached() -> Dict[str, Any]:
    global _ontology_cache
    if _ontology_cache is None:
        try:
            _ontology_cache = load_ontology() or {}
        except Exception:
            _ontology_cache = {}
    return _ontology_cache

def build_parent_map(node: Dict[str, Any], parent: Optional[str], acc: Dict[str, Optional[str]]):
    nid = node.get('id')
    if nid:
        acc[nid] = parent
    for ch in (node.get('children') or []):
        build_parent_map(ch, nid, acc)

def aggregate_hierarchical_f1(per_class_f1: Dict[str, float], class_counts: Dict[str, int]) -> Dict[str, Any]:
    ontology = ontology_tree_cached()
    root = ontology.get('root') or {}
    if not root:
        return {}
    parent_map: Dict[str, Optional[str]] = {}
    build_parent_map(root, None, parent_map)
    leaves = {leaf['id'] for leaf in flatten_leaves(ontology)} if ontology else set()

    leaf_f1: Dict[str, float] = {}
    for k, v in per_class_f1.items():
        leaf_id, _ = map_coarse_label(k)
        if leaf_id not in leaves and k in leaves:
            leaf_id = k
        leaf_f1[leaf_id] = float(v)

    children: Dict[str, List[str]] = {nid: [] for nid in parent_map}
    for nid, parent in parent_map.items():
        if parent is not None and parent in children:
            children[parent].append(nid)

    aggregated: Dict[str, Dict[str, Any]] = {}

    def visit(nid: str):
        ch = children.get(nid, [])
        if not ch:
            f1 = leaf_f1.get(nid)
            cnt = class_counts.get(nid) or 0
            aggregated[nid] = {"f1": f1, "count": cnt, "children": []}
            return aggregated[nid]
        child_nodes = [visit(c) for c in ch]
        child_f1_vals = [c.get('f1') for c in child_nodes if c.get('f1') is not None]
        macro = float(sum(child_f1_vals) / len(child_f1_vals)) if child_f1_vals else None
        total_count = sum(c.get('count', 0) for c in child_nodes)
        if total_count > 0:
            weighted_num = sum((c.get('f1') or 0.0) * c.get('count', 0) for c in child_nodes)
            weighted = float(weighted_num / total_count) if weighted_num else 0.0
        else:
            weighted = None
        aggregated[nid] = {
            "f1_macro_children": macro,
            "f1_weighted_children": weighted,
            "count": total_count,
            "children": [{"id": c_id, **aggregated[c_id]} for c_id in ch]
        }
        return aggregated[nid]

    root_id = root.get('id') or ''
    visit(root_id)
    return aggregated.get(root_id, {})

def js_divergence(p: List[float], q: List[float]) -> float:
    import math
    if len(p) != len(q):
        raise ValueError("Distribution length mismatch")
    eps = 1e-12
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kld(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps)
            bi = max(bi, eps)
            s += ai * math.log(ai / bi)
        return s
    return float((kld(p, m) + kld(q, m)) / 2.0)

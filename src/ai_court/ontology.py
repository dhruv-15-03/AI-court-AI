"""Ontology utilities for outcome hierarchy.

Loads YAML ontology (versioned) and provides mapping helpers:
 - load_ontology() -> dict structure
 - flatten_leaves(tree) -> list of leaf nodes
 - map_coarse_label(label) -> ontology leaf id (fallback unrecognized)

Future extensions: semantic diffing, migration scripts, hierarchical metrics aggregation.
"""
from __future__ import annotations

import os
import json
from functools import lru_cache
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ONTOLOGY_PATH = os.path.join(PROJECT_ROOT, 'ontology', 'outcomes_v1.yml')


class OntologyLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_ontology() -> Dict[str, Any]:
    """Load ontology YAML if available; otherwise return a safe fallback.

    The fallback keeps the system operational by defining a minimal flat ontology that
    maps known coarse labels to themselves and groups them under a synthetic root.
    This prevents training/evaluation code from failing when the external ontology file
    is absent or PyYAML is unavailable.
    """
    if yaml is None or not os.path.exists(ONTOLOGY_PATH):
        # Safe fallback ontology with a flat structure and identity mapping
        return _default_fallback_ontology()
    try:
        with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data or _default_fallback_ontology()
    except Exception:
        # Any YAML parse or IO error -> fallback
        return _default_fallback_ontology()


def _default_fallback_ontology() -> Dict[str, Any]:
    """Return a minimal ontology structure and mapping used as a fallback.

    The mapping covers the coarse outcomes produced by normalize_outcome() in
    ai_court.model.legal_case_classifier. New/unknown labels will be mapped to
    'unrecognized'.
    """
    coarse_classes = [
        'Acquittal/Conviction Overturned',
        'Conviction Upheld/Appeal Dismissed',
        'Charge Sheet Quashed',
        'Charges/Proceedings Quashed',
        'Sentence Reduced/Modified',
        'Bail Granted',
        'Bail Denied',
        'Relief Granted/Convicted',
        'Relief Denied/Dismissed',
        'Case Remanded/Sent Back',
        'Petition Withdrawn/Dismissed as Withdrawn',
        'Other',
    ]
    # Identity mapping: map coarse label -> same id
    mapping = {c: c for c in coarse_classes}
    # Add canonical sink for unmapped
    mapping['unrecognized'] = 'unrecognized'
    # Flat tree leaves
    children = [{"id": c, "name": c, "children": []} for c in mapping.values()]
    return {
        'version': 'fallback',
        'root': {
            'id': 'root',
            'name': 'Legal Outcomes (Fallback)',
            'children': children,
        },
        'mapping': mapping,
    }


def _walk(node: Dict[str, Any], path: List[str], leaves: List[Dict[str, Any]]):
    children = node.get('children') or []
    if not children:
        leaves.append({"id": node.get('id'), "name": node.get('name'), "path": path + [node.get('id')]})
    else:
        for ch in children:
            _walk(ch, path + [node.get('id')], leaves)


def flatten_leaves(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    root = data.get('root') or {}
    _walk(root, [], leaves)
    return leaves


@lru_cache(maxsize=128)
def map_coarse_label(coarse: str) -> Tuple[str, bool]:
    """Map old coarse label to ontology leaf id.
    Returns tuple (leaf_id, created_flag) where created_flag is False for existing mapping,
    True if fallback to unrecognized.
    """
    data = load_ontology()
    mapping = data.get('mapping', {})
    if coarse in mapping:
        return mapping[coarse], False
    return 'unrecognized', True


def ontology_metadata() -> Dict[str, Any]:
    data = load_ontology()
    leaves = flatten_leaves(data)
    return {
        'version': data.get('version', 'fallback'),
        'num_leaves': len(leaves),
        'leaves': [leaf['id'] for leaf in leaves],
    }


if __name__ == '__main__':  # manual quick check
    try:
        o = load_ontology()
        print(json.dumps(ontology_metadata(), indent=2))
    except Exception as e:
        print(f"Ontology load failed: {e}")

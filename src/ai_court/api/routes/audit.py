"""Lightweight audit log for Python AI service actions.

Records every significant AI operation (analysis, document generation, retrain,
etc.) to a JSONL file. Also provides a REST endpoint for the frontend audit
viewer to query.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from ai_court.api import config

audit_bp = Blueprint("audit", __name__)
logger = logging.getLogger(__name__)

AUDIT_FILE = os.path.join(config.PROJECT_ROOT, "data", "audit_log.jsonl")


def record(
    action: str,
    *,
    actor: str = "system",
    entity_type: str = "",
    entity_id: str = "",
    details: Optional[Dict[str, Any]] = None,
    ip_address: str = "",
) -> Dict[str, Any]:
    """Append an audit entry. Safe to call from any codepath."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "actor": actor,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "details": details or {},
        "ip_address": ip_address,
    }
    try:
        os.makedirs(os.path.dirname(AUDIT_FILE) or ".", exist_ok=True)
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("audit write failed: %s", exc)
    return entry


def _load_entries(limit: int = 200, entity_type: str = "", entity_id: str = "") -> List[dict]:
    if not os.path.exists(AUDIT_FILE):
        return []
    entries: List[dict] = []
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entity_type and e.get("entity_type") != entity_type:
                continue
            if entity_id and str(e.get("entity_id")) != str(entity_id):
                continue
            entries.append(e)
    # Newest first
    entries.sort(key=lambda x: x.get("epoch", 0), reverse=True)
    return entries[:limit]


@audit_bp.route("/api/audit/ai", methods=["GET"])
def ai_audit_log():
    """Return recent AI audit entries.

    Query params:
        limit        max rows (default 200, max 1000)
        entity_type  filter by type (e.g. 'case', 'document', 'model')
        entity_id    filter by id
    """
    limit = min(int(request.args.get("limit", 200)), 1000)
    entity_type = request.args.get("entity_type", "")
    entity_id = request.args.get("entity_id", "")
    entries = _load_entries(limit, entity_type, entity_id)
    return jsonify({"entries": entries, "count": len(entries)})

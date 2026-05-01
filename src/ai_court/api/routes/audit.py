"""Audit log for Python AI service actions — SQLite-backed.

Records every significant AI operation. Falls back to JSONL if SQLite fails.
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


def record(
    action: str,
    *,
    actor: str = "system",
    entity_type: str = "",
    entity_id: str = "",
    details: Optional[Dict[str, Any]] = None,
    ip_address: str = "",
) -> Dict[str, Any]:
    """Append an audit entry — SQLite primary, JSONL fallback."""
    try:
        from ai_court.storage import audit_record
        return audit_record(action, actor=actor, entity_type=entity_type,
                            entity_id=entity_id, details=details, ip_address=ip_address)
    except Exception as exc:
        logger.warning("SQLite audit write failed, using JSONL fallback: %s", exc)
    # JSONL fallback
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
        p = os.path.join(config.PROJECT_ROOT, "data", "audit_log.jsonl")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return entry


@audit_bp.route("/api/audit/ai", methods=["GET"])
def ai_audit_log():
    """Return recent AI audit entries from SQLite."""
    limit = min(int(request.args.get("limit", 200)), 1000)
    entity_type = request.args.get("entity_type", "")
    entity_id = request.args.get("entity_id", "")
    try:
        from ai_court.storage import audit_query
        entries = audit_query(limit, entity_type, entity_id)
    except Exception:
        entries = []
    return jsonify({"entries": entries, "count": len(entries)})

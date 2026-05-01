"""SQLite-backed persistent store for all AI service runtime data.

Replaces JSONL flat files with a single SQLite database that survives
container restarts on free hosting tiers. Zero external dependencies —
sqlite3 is in Python's stdlib.

Tables:
    al_queue        – Active learning uncertainty queue
    labels          – Human-labeled examples for retraining
    audit_log       – Immutable AI action audit trail
    user_feedback   – Thumbs up/down feedback on AI responses
    sessions        – Persisted chat sessions (optional, replaces in-memory dict)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ai_court.db"),
)
_DB_PATH = os.path.abspath(_DB_PATH)

# Thread-local connections (sqlite3 objects can't be shared across threads)
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


@contextmanager
def _cursor():
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _cursor() as cur:
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS al_queue (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            uncertainty REAL DEFAULT 1.0,
            uncertainty_method TEXT DEFAULT 'entropy',
            predicted_label TEXT,
            probabilities TEXT,  -- JSON
            case_type TEXT DEFAULT '',
            added_at REAL,
            label TEXT,
            labeled_at REAL
        );

        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            case_type TEXT DEFAULT '',
            source TEXT DEFAULT 'human',
            timestamp REAL,
            metadata TEXT  -- JSON
        );
        CREATE INDEX IF NOT EXISTS idx_labels_source ON labels(source);

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            epoch REAL,
            actor TEXT DEFAULT 'system',
            action TEXT,
            entity_type TEXT DEFAULT '',
            entity_id TEXT DEFAULT '',
            details TEXT,  -- JSON
            ip_address TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
        CREATE INDEX IF NOT EXISTS idx_audit_epoch ON audit_log(epoch);

        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            response_type TEXT DEFAULT 'unknown',
            helpful INTEGER NOT NULL,  -- 1=true, 0=false
            query_excerpt TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            comment TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT '',
            case_query TEXT DEFAULT '',
            case_type TEXT DEFAULT '',
            tier TEXT DEFAULT 'free',
            created_at REAL,
            last_active REAL,
            analysis_result TEXT,  -- JSON
            conversation TEXT      -- JSON list
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)
    logger.info("SQLite DB initialized at %s", _DB_PATH)


# ── AL Queue ──────────────────────────────────────────────────────────

def al_queue_list() -> List[Dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM al_queue WHERE label IS NULL ORDER BY uncertainty DESC")
        return [dict(r) for r in cur.fetchall()]


def al_queue_add(item: Dict) -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO al_queue (id, text, uncertainty, uncertainty_method, "
            "predicted_label, probabilities, case_type, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item["id"], item["text"], item.get("uncertainty", 1.0),
             item.get("uncertainty_method", "entropy"), item.get("predicted_label"),
             json.dumps(item.get("probabilities", {})), item.get("case_type", ""),
             item.get("added_at", time.time())),
        )


def al_queue_label(item_id: str, label: str) -> bool:
    with _cursor() as cur:
        cur.execute(
            "UPDATE al_queue SET label = ?, labeled_at = ? WHERE id = ?",
            (label, time.time(), item_id),
        )
        return cur.rowcount > 0


def al_queue_count() -> int:
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM al_queue WHERE label IS NULL")
        return cur.fetchone()[0]


# ── Labels ────────────────────────────────────────────────────────────

def label_add(text: str, label: str, case_type: str = "", source: str = "human",
              metadata: Optional[Dict] = None) -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO labels (text, label, case_type, source, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (text, label, case_type, source, time.time(), json.dumps(metadata or {})),
        )


def label_count() -> int:
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM labels")
        return cur.fetchone()[0]


def label_load_all() -> List[Dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM labels ORDER BY timestamp")
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result


# ── Audit Log ─────────────────────────────────────────────────────────

def audit_record(action: str, *, actor: str = "system", entity_type: str = "",
                 entity_id: str = "", details: Optional[Dict] = None,
                 ip_address: str = "") -> Dict:
    from datetime import datetime, timezone
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
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log (timestamp, epoch, actor, action, entity_type, "
            "entity_id, details, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry["timestamp"], entry["epoch"], actor, action, entity_type,
             entity_id, json.dumps(entry["details"]), ip_address),
        )
    return entry


def audit_query(limit: int = 200, entity_type: str = "", entity_id: str = "") -> List[Dict]:
    with _cursor() as cur:
        sql = "SELECT * FROM audit_log WHERE 1=1"
        params: list = []
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if entity_id:
            sql += " AND entity_id = ?"
            params.append(str(entity_id))
        sql += " ORDER BY epoch DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("details"):
                try:
                    d["details"] = json.loads(d["details"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result


# ── User Feedback ─────────────────────────────────────────────────────

def feedback_add(response_type: str, helpful: bool, query_excerpt: str = "",
                 session_id: str = "", comment: str = "") -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO user_feedback (timestamp, response_type, helpful, "
            "query_excerpt, session_id, comment) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), response_type, 1 if helpful else 0,
             query_excerpt[:200], session_id, comment[:500]),
        )


def feedback_stats() -> Dict:
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN helpful=1 THEN 1 ELSE 0 END) as helpful, "
                    "SUM(CASE WHEN helpful=0 THEN 1 ELSE 0 END) as not_helpful "
                    "FROM user_feedback")
        r = cur.fetchone()
        total = r["total"] or 0
        h = r["helpful"] or 0
        nh = r["not_helpful"] or 0
        return {"total": total, "helpful": h, "not_helpful": nh,
                "rate": round(h / total, 3) if total > 0 else 0}


# ── Sessions ──────────────────────────────────────────────────────────

def session_save(session_id: str, data: Dict) -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO sessions (session_id, user_id, case_query, "
            "case_type, tier, created_at, last_active, analysis_result, conversation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, data.get("user_id", ""), data.get("case_query", ""),
             data.get("case_type", ""), data.get("tier", "free"),
             data.get("created_at", time.time()), time.time(),
             json.dumps(data.get("analysis_result") or {}),
             json.dumps(data.get("conversation_history") or [])),
        )


def session_load(session_id: str) -> Optional[Dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        r = cur.fetchone()
        if r is None:
            return None
        d = dict(r)
        for key in ("analysis_result", "conversation"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


def session_delete(session_id: str) -> bool:
    with _cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0


def session_list(user_id: str = "") -> List[Dict]:
    with _cursor() as cur:
        if user_id:
            cur.execute("SELECT session_id, user_id, case_query, tier, created_at, last_active "
                        "FROM sessions WHERE user_id = ? ORDER BY last_active DESC", (user_id,))
        else:
            cur.execute("SELECT session_id, user_id, case_query, tier, created_at, last_active "
                        "FROM sessions ORDER BY last_active DESC LIMIT 100")
        return [dict(r) for r in cur.fetchall()]


def session_cleanup(max_age_seconds: int = 86400) -> int:
    cutoff = time.time() - max_age_seconds
    with _cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff,))
        return cur.rowcount

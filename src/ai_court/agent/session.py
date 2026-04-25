"""Session Manager — per-case contextual chat sessions.

When a user pays for AI on one case, they get a session that:
  - Stores the initial case analysis context
  - Maintains conversation history
  - Answers follow-up questions with full case context
  - Has TTL-based expiry for memory management
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default session TTL: 24 hours
DEFAULT_SESSION_TTL = 86400


@dataclass
class CaseSession:
    """A single user's case analysis session."""

    session_id: str
    user_id: Optional[str] = None
    case_query: str = ""
    case_type: str = ""
    analysis_result: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    ttl: float = DEFAULT_SESSION_TTL
    tier: str = "free"  # free | pro | unlimited | court

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > self.ttl

    def touch(self) -> None:
        self.last_active = time.time()

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": time.time()}
        )
        self.touch()

    def get_history_for_llm(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Return history formatted for LLM messages (role + content only)."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.conversation_history[-max_messages:]
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "case_query": self.case_query,
            "case_type": self.case_type,
            "tier": self.tier,
            "message_count": len(self.conversation_history),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "is_expired": self.is_expired,
        }


class SessionManager:
    """Manages per-case chat sessions with TTL-based cleanup."""

    def __init__(self, max_sessions: int = 1000, cleanup_interval: int = 300):
        self._sessions: Dict[str, CaseSession] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def create_session(
        self,
        *,
        user_id: Optional[str] = None,
        case_query: str = "",
        case_type: str = "",
        analysis_result: Optional[Dict[str, Any]] = None,
        tier: str = "free",
        ttl: float = DEFAULT_SESSION_TTL,
    ) -> CaseSession:
        """Create a new case session after initial analysis."""
        self._maybe_cleanup()

        session_id = str(uuid.uuid4())
        session = CaseSession(
            session_id=session_id,
            user_id=user_id,
            case_query=case_query,
            case_type=case_type,
            analysis_result=analysis_result,
            tier=tier,
            ttl=ttl,
        )

        # Store the initial analysis in conversation history
        if case_query:
            session.add_message("user", case_query)
        if analysis_result and analysis_result.get("analysis"):
            session.add_message("assistant", analysis_result["analysis"])

        with self._lock:
            # Evict oldest if at capacity
            if len(self._sessions) >= self._max_sessions:
                oldest_key = min(
                    self._sessions, key=lambda k: self._sessions[k].last_active
                )
                del self._sessions[oldest_key]
            self._sessions[session_id] = session

        logger.info("Created session %s for user %s", session_id, user_id)
        return session

    def get_session(self, session_id: str) -> Optional[CaseSession]:
        """Get session by ID, returns None if expired or not found."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self._remove_session(session_id)
            return None
        return session

    def _remove_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _maybe_cleanup(self) -> None:
        """Remove expired sessions periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items() if s.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                logger.info("Cleaned up %d expired sessions", len(expired))

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active sessions, optionally filtered by user."""
        with self._lock:
            sessions = list(self._sessions.values())
        active = [s for s in sessions if not s.is_expired]
        if user_id:
            active = [s for s in active if s.user_id == user_id]
        return [s.to_dict() for s in active]

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._sessions.values() if not s.is_expired)

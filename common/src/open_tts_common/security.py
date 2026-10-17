"""Short-lived in-memory owner-code and browser-session management."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import time


@dataclass(slots=True)
class _Code:
    value: str
    expires_at: float


class OwnerSessions:
    def __init__(self, timeout_minutes: int = 60, code_lifetime_seconds: int = 300) -> None:
        self._timeout_seconds = timeout_minutes * 60
        self._code_lifetime_seconds = code_lifetime_seconds
        self._owner_code: _Code | None = None
        self._sessions: dict[str, float] = {}

    def issue_code(self) -> str:
        code = secrets.token_urlsafe(9)
        self._owner_code = _Code(code, time.monotonic() + self._code_lifetime_seconds)
        return code

    def exchange(self, code: str) -> str | None:
        issued = self._owner_code
        self._owner_code = None
        if issued is None or time.monotonic() >= issued.expires_at:
            return None
        if not secrets.compare_digest(code, issued.value):
            return None
        token = secrets.token_urlsafe(32)
        self._sessions[token] = time.monotonic() + self._timeout_seconds
        return token

    def is_valid(self, token: str | None) -> bool:
        if not token:
            return False
        expires_at = self._sessions.get(token)
        if expires_at is None:
            return False
        if time.monotonic() >= expires_at:
            del self._sessions[token]
            return False
        return True

import os
import shutil
import time
import uuid
from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, ".dicom_sessions")
SESSION_TTL = 3600  # 1 hour

_sessions: dict[str, dict] = {}  # session_id -> {"dicom_dir": str, "created_at": float}
_lock = Lock()


def create_session(dicom_dir: str) -> str:
    """Register a DICOM directory and return a new session ID."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    session_id = uuid.uuid4().hex
    dest = os.path.join(SESSIONS_DIR, session_id)
    shutil.copytree(dicom_dir, dest)
    with _lock:
        _sessions[session_id] = {
            "dicom_dir": dest,
            "created_at": time.time(),
        }
    return session_id


def get_session_dicom_dir(session_id: str) -> str | None:
    """Return the DICOM directory path for a session, or None if not found."""
    with _lock:
        entry = _sessions.get(session_id)
    if entry is None:
        return None
    return entry["dicom_dir"]


def cleanup_session(session_id: str) -> None:
    """Remove a session and its files from disk."""
    with _lock:
        entry = _sessions.pop(session_id, None)
    if entry and os.path.isdir(entry["dicom_dir"]):
        shutil.rmtree(entry["dicom_dir"], ignore_errors=True)


def cleanup_expired() -> None:
    """Remove all sessions older than SESSION_TTL."""
    now = time.time()
    with _lock:
        expired = [
            sid for sid, info in _sessions.items()
            if now - info["created_at"] > SESSION_TTL
        ]
    for sid in expired:
        cleanup_session(sid)

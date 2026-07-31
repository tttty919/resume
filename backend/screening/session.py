"""Screening session store — in-memory dict, resets on server restart.

Similar to agent/session.py but for the main screening flow (parse → extract → match → risk → recommend).
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ScreeningSession:
    id: str
    job_id: str = ""
    requirements: list = field(default_factory=list)
    candidates: list = field(default_factory=list)  # list of {index, file_name, status, ...}
    phase: str = "init"          # init | parsing | extracting | matching | done
    active_idx: int = -1
    created_at: str = ""
    events: list = field(default_factory=list)  # [(event_type, payload), ...] for replay
    # LLM 凭据（供 HR 复核后局部重算使用；仅内存，随进程重启清空）
    api_key: str = ""
    base_url: str = ""
    model: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ── global store ──
_sessions: dict[str, ScreeningSession] = {}
_event_queues: dict[str, asyncio.Queue] = {}


def create_session(job_id: str, requirements: list, candidate_names: list[str]) -> ScreeningSession:
    sid = uuid.uuid4().hex[:12]
    candidates = [
        {"index": i, "file_name": n, "name": n, "status": "pending",
         "resume": None, "raw_text": "", "matches": [], "analysis": {}, "scoring": {}}
        for i, n in enumerate(candidate_names)
    ]
    session = ScreeningSession(
        id=sid, job_id=job_id, requirements=requirements,
        candidates=candidates, phase="init",
    )
    _sessions[sid] = session
    _event_queues[sid] = asyncio.Queue()
    for evt in [("session_created", {"session_id": sid, "total": len(candidates)}), ("phase", {"phase": "init"})]:
        session.events.append(evt)
    return session


def get_session(sid: str) -> ScreeningSession | None:
    return _sessions.get(sid)


def get_queue(sid: str) -> asyncio.Queue | None:
    return _event_queues.get(sid)


def emit(sid: str, event_type: str, payload: dict):
    session = _sessions.get(sid)
    if session is None:
        return
    evt = (event_type, payload)
    session.events.append(evt)
    q = _event_queues.get(sid)
    if q:
        q.put_nowait(evt)


def cleanup_session(sid: str):
    _sessions.pop(sid, None)
    _event_queues.pop(sid, None)

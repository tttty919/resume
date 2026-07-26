"""Agent session store — in-memory dict, resets on server restart.

Each session tracks:
- requirements: JD requirements list
- candidates: list of { index, name, file_name, status, resume, matches, ... }
- pause_queue: candidate indices waiting for HR review
- event_log: list of (event_type, data) for replay
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

CANDIDATE_STATUSES = ("pending", "parsing", "matching", "risk_analysis", "recommending", "paused", "done", "error")

@dataclass
class CandidateSlot:
    index: int
    name: str = ""
    file_name: str = ""
    status: str = "pending"        # one of CANDIDATE_STATUSES
    resume: dict = field(default_factory=dict)
    raw_text: str = ""
    matches: list = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    scoring: dict = field(default_factory=dict)
    stop_reason: str = ""
    pending_hr: list = field(default_factory=list)
    retry_count: int = 0

@dataclass
class AgentSession:
    id: str
    requirements: list = field(default_factory=list)
    candidates: list[CandidateSlot] = field(default_factory=list)
    active_idx: int = -1
    phase: str = "init"            # init | processing | waiting_hr | done
    created_at: str = ""
    events: list = field(default_factory=list)  # (event_type, payload) tuples for replay

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ── global store ──
_sessions: dict[str, AgentSession] = {}
_event_queues: dict[str, asyncio.Queue] = {}

def create_session(requirements: list, candidate_names: list[str], pre_parsed: list[dict] | None = None) -> AgentSession:
    sid = uuid.uuid4().hex[:12]
    slots = []
    for i, n in enumerate(candidate_names):
        slot = CandidateSlot(index=i, name=n, file_name=n)
        if pre_parsed and i < len(pre_parsed):
            pp = pre_parsed[i]
            slot.raw_text = pp.get("raw_text", "")
            slot.resume = pp.get("resume", {})
            if slot.resume:
                slot.name = slot.resume.get("basic_info", {}).get("name", n)
        slots.append(slot)
    session = AgentSession(id=sid, requirements=requirements, candidates=slots, phase="init")
    _sessions[sid] = session
    _event_queues[sid] = asyncio.Queue()
    for evt in [("session_created", {"session_id": sid, "total": len(slots)}), ("phase", {"phase": "init"})]:
        session.events.append(evt)
    return session

def get_session(sid: str) -> AgentSession | None:
    return _sessions.get(sid)

def get_queue(sid: str) -> asyncio.Queue | None:
    return _event_queues.get(sid)

def emit(sid: str, event_type: str, payload: dict):
    """Push an event into the session's queue and log."""
    session = _sessions.get(sid)
    if session is None:
        return
    evt = (event_type, payload)
    session.events.append(evt)
    q = _event_queues.get(sid)
    if q:
        q.put_nowait(evt)

def pause_candidate(sid: str, idx: int, stop_reason: str, pending_hr: list):
    """Mark a candidate as paused and emit an agent_pause event."""
    session = _sessions.get(sid)
    if session is None:
        return
    slot = session.candidates[idx]
    slot.status = "paused"
    slot.stop_reason = stop_reason
    slot.pending_hr = pending_hr
    emit(sid, "agent_pause", {
        "index": idx, "name": slot.name,
        "stop_reason": stop_reason,
        "pending_hr": pending_hr,
    })
    # If all candidates are done or paused, transition to waiting_hr
    active = [s for s in session.candidates if s.status not in ("done", "error", "paused")]
    if not active:
        session.phase = "waiting_hr"
        emit(sid, "agent_waiting_hr", {
            "message": "Agent 已暂停，等待 HR 复核",
            "paused_candidates": [
                {"index": s.index, "name": s.name, "pending_hr": s.pending_hr}
                for s in session.candidates if s.status == "paused"
            ],
        })

def resume_candidate(sid: str, idx: int):
    """Resume a paused candidate."""
    session = _sessions.get(sid)
    if session is None:
        return
    slot = session.candidates[idx]
    slot.status = "matching"  # will be re-run from matching step
    slot.stop_reason = ""
    session.phase = "processing"
    emit(sid, "agent_resume", {"index": idx, "name": slot.name, "message": f"Agent 正在恢复处理 {slot.name}"})

def cleanup_session(sid: str):
    """Remove session after TTL or explicit cleanup."""
    _sessions.pop(sid, None)
    _event_queues.pop(sid, None)

"""Space-aware path utilities — used by all routers for multi-tenant isolation."""
import contextvars
import re
from pathlib import Path

_current_space: contextvars.ContextVar[str] = contextvars.ContextVar("space", default="default")


def get_space() -> str:
    """Get current tenant space ID. Safe to call from any context."""
    return _current_space.get()


def set_space(space: str) -> None:
    """Set current tenant space. Called by middleware."""
    _current_space.set(space)


def space_dir(subdir: str = "") -> Path:
    """Get space-aware data directory, creating if needed."""
    space = get_space()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", space) or "default"
    base = Path(f"data/{safe}")
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base


def upload_dir() -> Path:
    """Space-aware upload directory."""
    space = get_space()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", space) or "default"
    p = Path(f"uploads/{safe}")
    p.mkdir(parents=True, exist_ok=True)
    return p

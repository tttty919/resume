"""ResumeStore — SQLite-backed cache for parsed resumes with MD5 dedup."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from backend.core.logger import get_logger

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "resume_cache.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS resume_cache (
    md5         TEXT PRIMARY KEY,
    file_name   TEXT NOT NULL,
    file_size   INTEGER NOT NULL DEFAULT 0,
    raw_text    TEXT NOT NULL DEFAULT '',
    parsed_json TEXT NOT NULL DEFAULT '',
    job_id      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    last_access REAL NOT NULL
)
"""

_DDL_MIGRATIONS = [
    "ALTER TABLE resume_cache ADD COLUMN job_id TEXT NOT NULL DEFAULT ''",
]

log = get_logger()


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    _run_migrations(conn)
    return conn


def _run_migrations(conn: sqlite3.Connection):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(resume_cache)").fetchall()}
    if "job_id" not in cols:
        try:
            conn.execute("ALTER TABLE resume_cache ADD COLUMN job_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass


def _md5_of_file(file_path: str | Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── Public API ──

def check(md5: str) -> dict | None:
    """Check if a resume with given MD5 exists. Returns metadata or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT file_name, file_size, job_id, created_at FROM resume_cache WHERE md5 = ?",
            (md5,),
        ).fetchone()
        if row:
            conn.execute("UPDATE resume_cache SET last_access = ? WHERE md5 = ?", (time.time(), md5))
            conn.commit()
            return {"md5": md5, "file_name": row[0], "file_size": row[1], "job_id": row[2], "created_at": row[3]}
        return None
    finally:
        conn.close()


def get(md5: str) -> dict | None:
    """Get full cached resume data by MD5."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT raw_text, parsed_json, file_name, file_size, job_id, created_at FROM resume_cache WHERE md5 = ?",
            (md5,),
        ).fetchone()
        if row:
            conn.execute("UPDATE resume_cache SET last_access = ? WHERE md5 = ?", (time.time(), md5))
            conn.commit()
            parsed = json.loads(row[1]) if row[1] else {}
            return {"raw_text": row[0], "parsed_resume": parsed, "file_name": row[2], "file_size": row[3], "job_id": row[4], "created_at": row[5]}
        return None
    finally:
        conn.close()


def save(file_path: str | Path, raw_text: str, parsed_resume: dict, job_id: str = ""):
    """Save parsed result to cache, keyed by file MD5."""
    try:
        md5 = _md5_of_file(file_path)
    except OSError:
        return

    fname = Path(file_path).name
    fsize = Path(file_path).stat().st_size
    now = time.time()
    try:
        parsed_json = json.dumps(parsed_resume, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        log.warning("[ResumeStore] JSON serialization failed (%s): %s — skipping cache", fname, e)
        return

    conn = _get_conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO resume_cache
               (md5, file_name, file_size, raw_text, parsed_json, job_id, created_at, last_access)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (md5, fname, fsize, raw_text, parsed_json, job_id, now, now),
        )
        conn.commit()
        log.info("[ResumeStore] Saved — %s (%d bytes, job=%s)", fname, fsize, job_id)
    finally:
        conn.close()


def list_by_job(job_id: str) -> list[dict]:
    """Get all cached resumes for a given job ID."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT md5, file_name, file_size, created_at FROM resume_cache WHERE job_id = ? ORDER BY created_at DESC",
            (job_id,),
        ).fetchall()
        return [{"md5": r[0], "file_name": r[1], "file_size": r[2], "created_at": r[3]} for r in rows]
    finally:
        conn.close()


def delete(md5: str) -> bool:
    """Delete a cached resume by MD5. Returns True if deleted, False if not found."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM resume_cache WHERE md5 = ?", (md5,))
        conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            log.info("[ResumeStore] Deleted — %s", md5)
        return deleted
    finally:
        conn.close()


def get_cached(file_path: str | Path) -> dict | None:
    """Legacy: check cache by file path. Returns {raw_text, parsed_resume} or None."""
    try:
        md5 = _md5_of_file(file_path)
    except OSError:
        return None
    entry = get(md5)
    if entry:
        return {"raw_text": entry["raw_text"], "parsed_resume": entry["parsed_resume"]}
    return None


def clear():
    """Clear all cached entries."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM resume_cache")
        conn.commit()
        log.info("[ResumeStore] Cache cleared")
    finally:
        conn.close()

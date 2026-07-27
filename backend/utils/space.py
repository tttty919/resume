"""多租户工作区隔离 —— 所有用户数据按 space 分目录存放。

机制：每个请求带一个 space 标识（前端注入 X-Space 头，或 ?space= 查询参数），
中间件把它写进 ContextVar；各数据落点（jobs.json / 上传文件 / 简历缓存库）
通过这里的 data_dir()/upload_dir() 拿到本 space 专属目录，互不串数据。

Chroma 同义词库是全局共享数据（非用户数据），不走这里。
"""
import contextvars
import hashlib
import re
from pathlib import Path

# 项目根目录：backend/utils/space.py → parents[2] = 仓库根
_ROOT = Path(__file__).resolve().parents[2]

_current_space: contextvars.ContextVar[str] = contextvars.ContextVar("space", default="default")


def _safe_space(space: str) -> str:
    """把任意 space 标识映射成文件系统安全、且稳定唯一的目录名。

    - 纯 [A-Za-z0-9_-] 的原样返回；
    - 含中文/空格/特殊字符的用「可读前缀_原文哈希」，保证同名→同目录、异名→异目录。
    """
    space = (space or "").strip() or "default"
    slug = re.sub(r"[^a-zA-Z0-9_-]", "", space)
    if slug == space:
        return slug or "default"
    h = hashlib.sha1(space.encode("utf-8")).hexdigest()[:10]
    return ((slug[:16] + "_") if slug else "ws_") + h


def set_space(space: str) -> None:
    """中间件在每个请求开始时调用，写入当前请求的 space。"""
    _current_space.set(_safe_space(space))


def get_space() -> str:
    """取当前请求的 space（已是安全目录名）；无上下文时为 'default'。"""
    return _current_space.get()


def data_dir() -> Path:
    """本 space 的持久化数据目录 data/<space>/（jobs.json、resume_cache.db 等）。"""
    p = _ROOT / "data" / get_space()
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_dir(base: str = "./uploads") -> Path:
    """本 space 的上传目录 <base>/<space>/。"""
    p = Path(base) / get_space()
    p.mkdir(parents=True, exist_ok=True)
    return p

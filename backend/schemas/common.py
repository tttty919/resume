"""通用 API 响应模型"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """统一 API 响应"""
    success: bool = True
    message: str = ""
    data: T | None = None

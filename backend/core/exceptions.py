"""统一异常体系 —— 所有业务异常由此派生"""

from typing import Any


class BaseAppException(Exception):
    """应用异常基类"""
    def __init__(self, message: str, code: str = "UNKNOWN", details: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class LLMException(BaseAppException):
    """LLM 调用失败"""
    def __init__(self, message: str, provider: str = "", model: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="LLM_ERROR", details={
            "provider": provider, "model": model, **(details or {})
        })


class ParseException(BaseAppException):
    """文档解析失败"""
    def __init__(self, message: str, file_type: str = "", file_name: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="PARSE_ERROR", details={
            "file_type": file_type, "file_name": file_name, **(details or {})
        })


class StorageException(BaseAppException):
    """存储操作失败"""
    def __init__(self, message: str, storage_type: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="STORAGE_ERROR", details={
            "storage_type": storage_type, **(details or {})
        })


class ValidationException(BaseAppException):
    """Schema 校验失败"""
    def __init__(self, message: str, field: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="VALIDATION_ERROR", details={
            "field": field, **(details or {})
        })


class WorkflowException(BaseAppException):
    """工作流执行异常"""
    def __init__(self, message: str, node: str = "", state: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="WORKFLOW_ERROR", details={
            "node": node, "state": state, **(details or {})
        })


class ConfigurationException(BaseAppException):
    """配置缺失或错误"""
    def __init__(self, message: str, key: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, code="CONFIG_ERROR", details={
            "key": key, **(details or {})
        })

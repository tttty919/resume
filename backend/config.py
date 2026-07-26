"""向后兼容 —— 从 core.config 重新导出配置"""
from backend.core.config import get_settings, Settings, DeepSeekSettings, AppSettings, ChromaDBSettings  # noqa: F401

_settings = get_settings()

DEEPSEEK_API_KEY = _settings.deepseek.api_key
DEEPSEEK_BASE_URL = _settings.deepseek.base_url
DEEPSEEK_MODEL = _settings.deepseek.model
APP_HOST = _settings.app.host
APP_PORT = _settings.app.port

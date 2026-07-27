"""全局配置 —— 使用 pydantic-settings 从环境变量 / .env 加载"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")


class DeepSeekSettings(BaseSettings):
    """DeepSeek LLM API 配置"""
    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_", env_file=_env_file, env_file_encoding="utf-8")
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"


class AppSettings(BaseSettings):
    """应用基础配置"""
    host: str = "127.0.0.1"
    port: int = 8000
    dev_port: int = 8766
    llm_timeout: int = 60
    llm_max_retries: int = 2
    upload_dir: str = "./uploads"


class ChromaDBSettings(BaseSettings):
    """ChromaDB 向量数据库配置"""
    persist_dir: str = "./chroma_data"
    collection_name: str = "skill_synonyms"


class Settings(BaseSettings):
    """全局配置聚合"""
    model_config = SettingsConfigDict(env_file=_env_file, env_file_encoding="utf-8")

    deepseek: DeepSeekSettings = DeepSeekSettings()
    app: AppSettings = AppSettings()
    chromadb: ChromaDBSettings = ChromaDBSettings()


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()

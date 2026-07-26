"""日志配置 —— 基于 Loguru"""

import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """配置 Loguru 输出到 stderr，JSON 格式保留结构化信息"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def get_logger():
    """获取 logger 实例"""
    return logger

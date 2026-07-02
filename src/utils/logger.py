"""Shared logger — consistent structured logging across all pipeline stages."""

import sys
from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        level=level,
        colorize=True,
    )
    logger.add(
        "logs/pipeline.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


setup_logger()

__all__ = ["logger"]

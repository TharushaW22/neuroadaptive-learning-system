"""Logging setup. File logging can be disabled for privacy."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None, enable_file: bool = False) -> logging.Logger:
    """Configure the root 'vcbu' logger. Console always; file only when enable_file=True."""
    global _CONFIGURED
    logger = logging.getLogger("vcbu")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if enable_file and log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "vcbu.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vcbu.{name}")

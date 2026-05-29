"""
src/logger.py
Centralized logger using Logdash + standard Python logging.
Every script imports this to get consistent, structured logs.
"""

import logging
import os
import sys
from datetime import datetime

# Try to import logdash — gracefully degrade if not installed
try:
    from logdash import LogDash
    LOGDASH_AVAILABLE = True
except ImportError:
    LOGDASH_AVAILABLE = False


class MLOpsLogger:
    """
    Wraps Python logging + Logdash into one clean interface.
    Usage:
        from src.logger import get_logger
        logger = get_logger("train")
        logger.info("Training started")
        logger.warning("High RMSE detected")
        logger.error("Model file not found")
    """

    def __init__(self, name: str):
        self.name = name
        self._setup_python_logger()
        self._setup_logdash()

    def _setup_python_logger(self):
        """Set up Python standard logging with file + console handlers."""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(logging.DEBUG)

        if self._logger.handlers:
            return  # avoid duplicate handlers

        # Format
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        console.setLevel(logging.INFO)
        self._logger.addHandler(console)

        # File handler
        log_dir = os.getenv("LOGS_PATH", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)

    def _setup_logdash(self):
        """Set up Logdash if API key is available."""
        self._logdash = None
        if not LOGDASH_AVAILABLE:
            return

        api_key = os.getenv("LOGDASH_API_KEY", "")
        if not api_key or api_key == "your_logdash_api_key_here":
            return  # silently skip — key not configured

        try:
            ld = LogDash(api_key=api_key)
            self._logdash = ld.logger
        except Exception:
            pass  # don't break the app if logdash fails

    def _send_to_logdash(self, level: str, message: str):
        """Send log to Logdash (if configured)."""
        if not self._logdash:
            return
        try:
            msg = f"[{self.name}] {message}"
            if level == "info":
                self._logdash.info(msg)
            elif level == "warning":
                self._logdash.warning(msg)
            elif level == "error":
                self._logdash.error(msg)
        except Exception:
            pass  # never let logging break the ML pipeline

    def info(self, message: str):
        self._logger.info(message)
        self._send_to_logdash("info", message)

    def warning(self, message: str):
        self._logger.warning(message)
        self._send_to_logdash("warning", message)

    def error(self, message: str):
        self._logger.error(message)
        self._send_to_logdash("error", message)

    def debug(self, message: str):
        self._logger.debug(message)

    def metric(self, name: str, value: float):
        """Log a named metric (useful for monitoring hooks)."""
        msg = f"METRIC | {name} = {value:.4f}"
        self.info(msg)


# Module-level cache so each name gets one logger instance
_loggers = {}

def get_logger(name: str) -> MLOpsLogger:
    if name not in _loggers:
        _loggers[name] = MLOpsLogger(name)
    return _loggers[name]

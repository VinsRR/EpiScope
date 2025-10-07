"""
utils/logger.py

Configure a standard logger for the EpiScope project.  This module
centralises logging configuration so that all parts of the system
produce consistent and informative log messages.
"""
import logging
import os

LOG_LEVEL = os.getenv("EPISCOPE_LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure the root logger with a simple console handler."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


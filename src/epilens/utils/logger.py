"""
utils/logger.py

Configure a standard logger for the EpiLens project.  This module
centralises logging configuration so that all parts of the system
produce consistent and informative log messages.
"""

import logging

from epilens.settings import AppSettings


_NOISY_THIRD_PARTY_LOGGERS = (
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)


def setup_logging() -> None:
    """Show EpiLens logs while keeping dependency request chatter quiet."""
    configured_level = AppSettings.from_env().log_level.upper()
    epilens_level = getattr(logging, configured_level, logging.INFO)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("epilens").setLevel(epilens_level)
    for logger_name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

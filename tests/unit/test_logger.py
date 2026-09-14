import logging

from epilens.utils.logger import _NOISY_THIRD_PARTY_LOGGERS, setup_logging


def test_setup_logging_keeps_dependency_info_quiet(
    monkeypatch, caplog
) -> None:
    logger_names = ("", "epilens", *_NOISY_THIRD_PARTY_LOGGERS)
    previous_levels = {
        name: logging.getLogger(name).level for name in logger_names
    }

    try:
        monkeypatch.setenv("EPILENS_LOG_LEVEL", "DEBUG")
        setup_logging()

        assert logging.getLogger("epilens").level == logging.DEBUG
        assert all(
            logging.getLogger(name).level == logging.WARNING
            for name in _NOISY_THIRD_PARTY_LOGGERS
        )

        with caplog.at_level(logging.DEBUG):
            logging.getLogger("httpx").info("dependency request details")
            logging.getLogger("epilens.example").debug("EpiLens diagnostic")

        assert "dependency request details" not in caplog.messages
        assert "EpiLens diagnostic" in caplog.messages
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)

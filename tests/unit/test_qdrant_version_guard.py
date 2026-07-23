"""Tests for the Qdrant client/server minor-version compatibility guard.

These exercise the pure helpers in ``episcope.vectordb.qdrant`` and do not
require ``qdrant-client`` or a live server.
"""

from __future__ import annotations

import importlib.metadata

import pytest

import episcope.vectordb.qdrant as q
from episcope.vectordb.qdrant import (
    QdrantVersionMismatchError,
    _check_qdrant_version_compatibility,
    _parse_major_minor,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("v1.13.5", (1, 13)),
        ("1.18.0", (1, 18)),
        ("2.0.0rc1", (2, 0)),
        ("weird", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_major_minor(raw, expected) -> None:
    assert _parse_major_minor(raw) == expected


@pytest.fixture()
def _no_skip(monkeypatch):
    monkeypatch.delenv(q._SKIP_VERSION_CHECK_ENV, raising=False)


def _set_client_version(monkeypatch, version: str) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)


def _set_server_version(monkeypatch, version) -> None:
    monkeypatch.setattr(q, "_fetch_server_version", lambda url, api_key, timeout: version)


def test_minor_mismatch_hard_errors(monkeypatch, _no_skip) -> None:
    _set_client_version(monkeypatch, "1.18.0")
    _set_server_version(monkeypatch, "1.13.5")
    with pytest.raises(QdrantVersionMismatchError) as exc:
        _check_qdrant_version_compatibility("http://x:6333", None, 5)
    message = str(exc.value)
    assert "1.18.0" in message and "1.13.5" in message
    # actionable remediation targeting the server minor
    assert "qdrant-client~=1.13.0" in message


def test_matching_minor_passes(monkeypatch, _no_skip) -> None:
    _set_client_version(monkeypatch, "1.13.5")
    _set_server_version(monkeypatch, "1.13.2")  # patch differences are fine
    _check_qdrant_version_compatibility("http://x:6333", None, 5)


def test_major_mismatch_hard_errors(monkeypatch, _no_skip) -> None:
    _set_client_version(monkeypatch, "2.0.0")
    _set_server_version(monkeypatch, "1.13.5")
    with pytest.raises(QdrantVersionMismatchError):
        _check_qdrant_version_compatibility("http://x:6333", None, 5)


def test_unreachable_server_is_skipped(monkeypatch, _no_skip) -> None:
    _set_client_version(monkeypatch, "1.18.0")

    def _boom(url, api_key, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(q, "_fetch_server_version", _boom)
    # Must not raise: connectivity issues surface on the first real query.
    _check_qdrant_version_compatibility("http://x:6333", None, 5)


def test_unknown_server_version_is_skipped(monkeypatch, _no_skip) -> None:
    _set_client_version(monkeypatch, "1.18.0")
    _set_server_version(monkeypatch, None)
    _check_qdrant_version_compatibility("http://x:6333", None, 5)


def test_missing_client_package_is_skipped(monkeypatch, _no_skip) -> None:
    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    _set_server_version(monkeypatch, "1.13.5")
    _check_qdrant_version_compatibility("http://x:6333", None, 5)


def test_escape_hatch_bypasses_check(monkeypatch) -> None:
    monkeypatch.setenv(q._SKIP_VERSION_CHECK_ENV, "1")
    _set_client_version(monkeypatch, "1.18.0")
    _set_server_version(monkeypatch, "1.13.5")
    # Would mismatch, but the env var bypasses the guard entirely.
    _check_qdrant_version_compatibility("http://x:6333", None, 5)

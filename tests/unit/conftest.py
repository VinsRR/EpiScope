from __future__ import annotations

import pytest

from episcope.workflows import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore the global task registry around each test.

    Declarative-task registration mutates module-level dicts; this keeps tests
    isolated from one another.
    """
    classifiers = dict(registry.CLASSIFIERS)
    miners = dict(registry.MINERS)
    try:
        yield
    finally:
        registry.CLASSIFIERS.clear()
        registry.CLASSIFIERS.update(classifiers)
        registry.MINERS.clear()
        registry.MINERS.update(miners)

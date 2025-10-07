"""
episcope.parse.querying.hyde
============================

This module re‑exports the unified :class:`HYDE` implementation from
``episcope.core.hyde``.  It exists for backwards compatibility with
earlier versions of the codebase where HYDE lived under
``parse/querying``.  New code should import directly from
``episcope.core.hyde``.

Example::

    from episcope.core.hyde import HYDE

    hyde = HYDE()
    paragraph = hyde.generate("What is R0 in epidemiology?")

"""
from episcope.core.hyde import HYDE

__all__ = ["HYDE"]
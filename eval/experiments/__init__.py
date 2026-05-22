"""Reporting layer that consumes ``classification.experiments`` manifests.

For every named experiment, ``report.py`` walks the manifest, evaluates
each baseline's predictions against the configured ground truth, computes
the pair-wise significance statistics declared in the config, and emits
paper-ready CSV/JSON summaries plus the figures referenced by the
classification results section.
"""

"""Experiment-level orchestration for the classification baseline grid.

This package is a thin coordination layer on top of the existing family
runners in ``classification/runners/`` and the eval pipeline in ``eval/``.
A single JSON experiment config (see ``eval/configs/classification/``)
names a set of baselines, their per-baseline overrides, and a shared
output / evaluation root. The runner here translates that config into the
existing per-family CLI invocations; the eval-side counterpart in
``eval/experiments/`` reads the same config to discover outputs and
produce paper-ready tables and figures.

The existing scripts under ``scripts/run_*.py`` continue to work
unchanged — this layer is an addition, not a replacement.
"""

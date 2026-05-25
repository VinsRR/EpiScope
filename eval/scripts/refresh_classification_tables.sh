#!/usr/bin/env bash
# Regenerate classification eval outputs and paper-ready pivot tables.
#
# Usage:
#   bash eval/scripts/refresh_classification_tables.sh
#
# Both result trees must be passed together:
#   outputs/baselines  — baseline families (zero-shot, unsupervised, supervised, LLM ablations)
#   outputs/output     — full EpiScope system results (Flash / Pro at various temperatures)
#
# Outputs land in eval_outputs/classification_baselines/.

set -euo pipefail

PYTHON="${PYTHON:-/Users/vins/miniconda3/envs/RAG_demos/bin/python}"
GT="${GT:-sampled_papers_full.csv}"
OUT_DIR="${OUT_DIR:-eval_outputs/classification_baselines}"

echo "=== Step 1: compute per-run metrics ==="
"$PYTHON" eval/scripts/run_classification_eval.py \
  --run-root outputs/baselines \
  --run-root outputs/output \
  --ground-truth "$GT" \
  --out-dir "$OUT_DIR"

echo ""
echo "=== Step 2: build paper pivot tables ==="
"$PYTHON" eval/scripts/build_baseline_paper_tables.py \
  --eval-dir "$OUT_DIR"

echo ""
echo "Done. Tables written to $OUT_DIR/baseline_tables.md"

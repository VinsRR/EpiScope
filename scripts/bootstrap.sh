#!/usr/bin/env bash
# Fastest way to get EpiLens running locally: creates a virtual environment,
# installs the package, and points to the optional provider setup.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/VinsRR/EpiLens/main/scripts/bootstrap.sh | bash
#   # or, from a clone:
#   bash scripts/bootstrap.sh
set -euo pipefail

INSTALL_TARGET="epilens"

VENV_DIR="${EPILENS_VENV_DIR:-.venv}"

echo "==> Creating virtual environment in ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip --quiet

echo "==> Installing EpiLens"
python -m pip install "${INSTALL_TARGET}"

echo
echo "==> Verifying the installation:"
echo "      python -c \"import epilens; print(epilens.__name__)\""
echo "      epilens doctor"

if [ -f .env ]; then
  echo "==> .env already exists, leaving it as-is"
else
  echo "==> No .env created: local parsing and retrieval need no configuration"
  echo "      Run 'epilens quickstart' later if you want generated answers"
fi

cat <<'EOF'

==> Setup complete.

Next steps:
  1. source .venv/bin/activate      # activate this environment in new shells
  2. epilens inspect /path/to/paper.pdf
  3. epilens explore "What data sources were used?" --path /path/to/paper.pdf --quality fast

For generated answers, install one provider and run the guided setup, e.g.:
  python -m pip install "epilens[gemini]"
  epilens quickstart

The first command that needs local embeddings will download a small model
(sentence-transformers/all-MiniLM-L6-v2, ~90MB) — this is a one-time,
one-off download, not a hang.
EOF

#!/usr/bin/env bash
# Fastest way to get EpiLens running locally: creates a virtual environment,
# installs the package, and prepares a .env file to fill in.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/VinsRR/EpiScope/main/scripts/bootstrap.sh | bash
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

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "==> Creating .env from .env.example (optional: add an LLM API key later)"
  cp .env.example .env
elif [ -f .env ]; then
  echo "==> .env already exists, leaving it as-is"
else
  echo "==> No .env.example found in this directory; create .env yourself with at least"
  echo "      EPILENS_LLM_PROVIDER=ollama, or install a hosted-provider extra"
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

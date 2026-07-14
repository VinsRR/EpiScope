#!/usr/bin/env bash
# Fastest way to get EpiScope running locally: creates a virtual environment,
# installs the package, and prepares a .env file to fill in.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/VinsRR/EpiScope/main/scripts/bootstrap.sh | bash
#   # or, from a clone:
#   bash scripts/bootstrap.sh
set -euo pipefail

# EpiScope is not on PyPI yet; this is the only line that changes once it is
# (swap for: pip install epi-scope).
INSTALL_TARGET='epi-scope @ git+https://github.com/VinsRR/EpiScope.git@main'

VENV_DIR="${EPISCOPE_VENV_DIR:-.venv}"

echo "==> Creating virtual environment in ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip --quiet

echo "==> Installing EpiScope (this pulls torch/transformers and can take a few minutes)"
python -m pip install "${INSTALL_TARGET}"

echo
echo "==> Note: the installed package is named 'epi-scope', but you import/run it as 'episcope':"
echo "      python -c \"import episcope; print(episcope.__name__)\""
echo "      episcope doctor"

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "==> Creating .env from .env.example (edit it to add your LLM API key)"
  cp .env.example .env
elif [ -f .env ]; then
  echo "==> .env already exists, leaving it as-is"
else
  echo "==> No .env.example found in this directory; create .env yourself with at least"
  echo "      GEMINI_API_KEY=... (or another provider key, or skip and use --llm-provider ollama)"
fi

cat <<'EOF'

==> Setup complete.

Next steps:
  1. source .venv/bin/activate      # activate this environment in new shells
  2. Edit .env and add an LLM API key (or use --llm-provider ollama for no key)
  3. episcope doctor                # sanity-check your environment
  4. episcope ask "What data sources were used?" --path /path/to/paper.pdf

The first command that needs local embeddings will download a small model
(sentence-transformers/all-MiniLM-L6-v2, ~90MB) — this is a one-time,
one-off download, not a hang.
EOF

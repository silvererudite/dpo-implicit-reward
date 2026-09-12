#!/usr/bin/env bash
# Set up this project locally using uv: create a venv and install dependencies
# from requirements.txt so `python -m src.<module>` works out of the box.
#
# Usage: ./setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [ ! -f pyproject.toml ]; then
    uv init --no-readme --no-workspace --vcs none .
fi

uv venv .venv
uv pip install -r requirements.txt --python .venv

echo
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo
echo "Then run pipeline stages with, e.g.:"
echo "  python -m src.train_sft --config configs/smoke.yaml"

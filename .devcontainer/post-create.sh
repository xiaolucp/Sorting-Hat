#!/usr/bin/env bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/.cache/uv"

cd "$(git rev-parse --show-toplevel)"

uv sync --all-extras

uv run python -c "import sorting_hat; print('sorting_hat', sorting_hat.__version__)"

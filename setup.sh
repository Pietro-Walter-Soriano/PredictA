#!/usr/bin/env bash
# One-time setup: Python venv + dependencies. First ./run.sh then downloads the whole archive (about 1 hour, ~80 MB).
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "python3 (3.10+) is required"; exit 1; }
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
mkdir -p data/manual
[ -f .env ] || cp .env.example .env
chmod 600 .env
echo "ok - now run ./run.sh (first run downloads the 2020-21 .. today archive)"

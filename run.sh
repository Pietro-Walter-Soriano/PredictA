#!/usr/bin/env bash
# Matchday cycle: refresh data, retrain, forecast upcoming Serie A matches, archive pre-kick-off forecasts,
# score every finished round, rebuild the dashboard. Safe to run any time; scheduled by Windows Task Scheduler.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
exec 9>data/.run.lock; flock -n 9 || { echo "already running"; exit 0; }
[ -f .env ] && { set -a; . ./.env; set +a; }          # optional keys, e.g. ODDS_API_KEY
PY=.venv/bin/python
[ -x "$PY" ] || { echo "run ./setup.sh first"; exit 1; }
echo "== $(date -u +%FT%TZ) start"
# whole archive on the first run (cached seasons are skipped), then only the current season
if [ -d data/raw/fotmob ] && [ "$(ls data/raw/fotmob | wc -l)" -gt 1 ]; then
  $PY -I src/fetch_fotmob.py data "$($PY -I -c 'import sys; sys.path.insert(0, "src"); import seasons; print(seasons.fotmob(seasons.current_start()))')"
else
  $PY -I src/fetch_fotmob.py data
fi
$PY -I src/fetch_aux.py data
$PY -I src/fetch_odds.py data
$PY -I src/dataset.py data
$PY -I src/backtest.py data > /dev/null
$PY -I src/predict.py data site/data
$PY -I check/verify.py site/data
$PY -I src/ledger.py site/data data/ledger
$PY -I src/evaluate.py data/ledger data data/rounds.json
$PY -I src/render.py src/template.html site/data site/serie-a.html data/rounds.json
$PY -I src/report.py data/rounds.json site/data/index.json REPORT.md
[ -n "${PUBLISH_DIR:-}" ] && cp site/serie-a.html "$PUBLISH_DIR/" || true   # optional extra copy (e.g. a Windows folder)
echo "== $(date -u +%FT%TZ) done"

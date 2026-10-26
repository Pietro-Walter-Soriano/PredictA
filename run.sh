#!/usr/bin/env bash
# Matchday cycle: refresh data, retrain, forecast upcoming Serie A matches, archive pre-kick-off forecasts,
# score every finished round, rebuild the dashboard. Safe to run any time; scheduled by GitHub Actions (or cron).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
exec 9>data/.run.lock; flock -n 9 || { echo "already running"; exit 0; }
# optional keys from .env (e.g. ODDS_API_KEY); a variable already set in the environment (GitHub secret) wins,
# and empty values in .env are ignored
if [ -f .env ]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -n "$v" ] && [ -z "${!k:-}" ] && export "$k=$v"
  done < <(sed 's/\r$//' .env)
fi
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
# a fast cycle that is not about to redo a forecast does not spend a metered The Odds API request (PREMATCH=true: it does)
if [ "${FAST:-}" = "true" ] && [ "${PREMATCH:-}" != "true" ]; then ODDS_API_KEY= $PY -I src/fetch_odds.py data; else $PY -I src/fetch_odds.py data; fi
$PY -I src/dataset.py data
# FAST=true (set by the planner for time-critical cycles: forecast before kick-off, score a finished match): skip the backtest,
# data/calibration.json and data/backtest.json stay the committed ones; the hourly routine cycle refreshes them
{ [ "${FAST:-}" = "true" ] && [ "${DUMP:-}" != "1" ]; } || $PY -I src/backtest.py data > /dev/null   # DUMP=1 forces it and writes data/backtest_matches.csv
$PY -I src/predict.py data site/data
$PY -I check/verify.py site/data
$PY -I src/ledger.py site/data data/ledger
$PY -I src/evaluate.py data/ledger data data/rounds.json
$PY -I src/render.py src/template.html site/data site/serie-a.html data/rounds.json
$PY -I src/report.py data/rounds.json site/data/index.json REPORT.md
[ -n "${PUBLISH_DIR:-}" ] && cp site/serie-a.html "$PUBLISH_DIR/" || true   # optional extra copy (e.g. a Windows folder)
echo "== $(date -u +%FT%TZ) done"

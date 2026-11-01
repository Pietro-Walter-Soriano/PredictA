"""Bookmaker odds that are better than ESPN/DraftKings, each one optional (skipped when unreachable or not configured):

  football-data.co.uk  Serie A CSV per season: Pinnacle / Bet365 / market-average closing 1X2 and O/U 2.5, from 2020-21.
                       Downloaded automatically, or dropped by hand into data/manual/I1_<yyyy>.csv (e.g. I1_2627.csv).
  The Odds API         pre-match 1X2 and totals of upcoming matches (Pinnacle when listed); needs ODDS_API_KEY in .env.
"""
import glob, json, os, shutil, sys, time
from datetime import datetime, timezone
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seasons

DATA = sys.argv[1]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/I1.csv"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds"
os.makedirs(f"{DATA}/raw/football_data", exist_ok=True)

# --- football-data.co.uk: past seasons once, current season every run ---
for y in seasons.all_starts():
    code = f"{y % 100:02d}{(y + 1) % 100:02d}"
    dest = f"{DATA}/raw/football_data/I1_{code}.csv"
    manual = f"{DATA}/manual/I1_{code}.csv"
    if os.path.exists(manual) and (not os.path.exists(dest) or os.path.getmtime(manual) > os.path.getmtime(dest)):
        shutil.copy(manual, dest); print("football-data", code, "from data/manual", flush=True); continue
    if os.path.exists(dest) and y != seasons.current_start(): continue
    try:
        r = requests.get(FD_URL.format(code=code), headers=UA, timeout=30)
        text = r.content.decode("utf-8-sig", errors="replace")   # newer files start with a UTF-8 BOM
        if r.status_code == 200 and text.lstrip().startswith("Div,"):
            open(dest, "w", encoding="utf-8").write(text); print("football-data", code, "downloaded", flush=True)
        else:
            print("football-data", code, "not available (http", r.status_code, "starts with", repr(text[:20]), ")", flush=True)
    except requests.RequestException as e:
        print("football-data unreachable:", type(e).__name__, "- put the CSVs in data/manual/ instead", flush=True); break
    time.sleep(1)

# --- The Odds API, extra markets (both teams to score, goal lines, corners, cards): the per-event endpoint, one call per match, only for
# matches that start within EXTRA_WINDOW_MIN minutes and not fetched in the last EXTRA_REFRESH_MIN. They are compared with our forecast, never used by it.
# Each call costs one credit per market: EXTRA_MAX caps the calls per run and nothing is fetched when fewer than EXTRA_MIN_CREDITS are left. ---
EXTRA_MARKETS = "btts,alternate_totals,alternate_totals_corners,alternate_totals_cards"
EXTRA_WINDOW_MIN, EXTRA_REFRESH_MIN = 75, 25
EXTRA_MAX = int(os.environ.get("ODDS_EXTRA_MAX", "8"))
EXTRA_MIN_CREDITS = int(os.environ.get("ODDS_EXTRA_MIN_CREDITS", "150"))

def fetch_extra(events, key, remaining):
    try:
        if remaining is not None and float(remaining) < EXTRA_MIN_CREDITS:
            print(f"odds-api extra: skipped, only {remaining} credits left (< {EXTRA_MIN_CREDITS})", flush=True); return
    except ValueError: pass
    now, done = datetime.now(timezone.utc), 0
    os.makedirs(f"{DATA}/raw/odds_extra", exist_ok=True)
    for ev in sorted(events, key=lambda e: e["commence_time"]):
        mins = (datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) - now).total_seconds() / 60
        if not 0 < mins <= EXTRA_WINDOW_MIN: continue
        last = max([datetime.strptime(os.path.basename(f).split("_")[-1][:13], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
                    for f in glob.glob(f"{DATA}/raw/odds_extra/{ev['id']}_*.json")] or [datetime.min.replace(tzinfo=timezone.utc)])
        if (now - last).total_seconds() / 60 < EXTRA_REFRESH_MIN: continue
        if done >= EXTRA_MAX: print("odds-api extra: cap of", EXTRA_MAX, "calls reached", flush=True); break
        try:
            rr = requests.get(f"https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/events/{ev['id']}/odds",
                              params={"apiKey": key, "regions": "eu", "markets": EXTRA_MARKETS, "oddsFormat": "decimal"}, headers=UA, timeout=30)
            if rr.status_code == 200:
                json.dump(rr.json(), open(f"{DATA}/raw/odds_extra/{ev['id']}_{now:%Y%m%dT%H%M}.json", "w")); done += 1
                print("odds-api extra:", ev["home_team"], "-", ev["away_team"], "T-%d" % mins, "credits left", rr.headers.get("x-requests-remaining"), flush=True)
            else:
                print("odds-api extra: http", rr.status_code, rr.text[:100].replace(key, "***"), flush=True)
                if rr.status_code in (401, 402, 429): break
        except requests.RequestException as e:
            print("odds-api extra unreachable:", type(e).__name__, flush=True); break
        time.sleep(0.5)

# --- The Odds API: one snapshot per run of all upcoming Serie A matches (1 request; free tier is metered per request) ---
key = os.environ.get("ODDS_API_KEY")
if not key:
    print("odds-api: ODDS_API_KEY not set, skipped", flush=True)
else:
    try:
        r = requests.get(ODDS_URL, params={"apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}, headers=UA, timeout=30)
        if r.status_code == 200:
            snap = f"{DATA}/raw/odds_api/{datetime.now(timezone.utc):%Y%m%dT%H%M}.json"
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            json.dump(r.json(), open(snap, "w"))
            print("odds-api:", len(r.json()), "matches, requests left", r.headers.get("x-requests-remaining"), flush=True)
            fetch_extra(r.json(), key, r.headers.get("x-requests-remaining"))
        else:
            print("odds-api: http", r.status_code, r.text[:120].replace(key, "***"), flush=True)
    except requests.RequestException as e:
        print("odds-api unreachable:", type(e).__name__, flush=True)

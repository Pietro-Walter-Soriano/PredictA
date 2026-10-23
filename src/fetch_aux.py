"""Understat league seasons (team pressing/territory) and ESPN Serie A scoreboards with odds."""
import gzip, json, os, sys, time
from datetime import date, timedelta
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seasons

DATA = sys.argv[1]
US_SEASONS = seasons.all_starts()
ESPN_FROM = date(seasons.current_start(), 8, 1)   # ESPN only exposes odds for the current season
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36", "X-Requested-With": "XMLHttpRequest"}
s = requests.Session(); s.headers.update(UA)

def get(url, as_json=True):
    for attempt in range(4):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200: return r.json() if as_json else r.text
            print("http", r.status_code, url, flush=True)
            if r.status_code in (400, 404): return None
        except Exception as e:
            print("err", e, url, flush=True)
        time.sleep(5 * (attempt + 1))
    return None

def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wt", encoding="utf-8") as f: json.dump(obj, f) if not isinstance(obj, str) else f.write(obj)
    os.replace(path + ".tmp", path)

# --- Understat: league season files (team pressing/territory per match); current season re-fetched ---
for y in US_SEASONS:
    p = f"{DATA}/raw/understat/{y}/league.json.gz"
    if y == US_SEASONS[-1] or not os.path.exists(p):
        d = get(f"https://understat.com/getLeagueData/Serie_A/{y}")
        if d: save(p, d)
        time.sleep(0.5)
print("understat ok", flush=True)

# --- ESPN Serie A scoreboards (odds incl. closing lines of past matches of this season, upcoming fixtures) ---
d0 = ESPN_FROM
while d0 <= date.today() + timedelta(days=10):
    p = f"{DATA}/raw/espn/sb_{d0:%Y%m%d}.json.gz"
    if d0 >= date.today() - timedelta(days=2) or not os.path.exists(p):
        sb = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates={d0:%Y%m%d}")
        if sb is not None: save(p, sb)
        time.sleep(0.2)
    d0 += timedelta(days=1)
print("espn ok", flush=True)

# --- ESPN closing odds of finished matches (scoreboards drop them after kick-off; the summary keeps them) ---
for f in sorted(os.listdir(f"{DATA}/raw/espn")):
    if not f.startswith("sb_"): continue
    for e in json.load(gzip.open(f"{DATA}/raw/espn/{f}", "rt")).get("events", []):
        p = f"{DATA}/raw/espn/odds_{e['id']}.json.gz"
        if e["competitions"][0]["status"]["type"]["completed"] and not os.path.exists(p):
            d = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/all/summary?event={e['id']}")
            if d: save(p, {"pickcenter": d.get("pickcenter", []), "odds": d.get("odds", []), "header": d["header"]})
            time.sleep(0.2)
print("espn odds ok", flush=True)

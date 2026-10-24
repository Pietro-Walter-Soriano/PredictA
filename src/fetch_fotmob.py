"""Cache FotMob Serie A fixtures and match details (finished matches once, upcoming ones refreshed each run).
Cup matches (Champions/Europa/Conference League, Coppa Italia, Supercoppa) are cached only when a Serie A club plays:
they feed player form and load, never the team ratings."""
import gzip, json, os, sys, time
from datetime import datetime, timedelta, timezone
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seasons

DATA = sys.argv[1]
LEAGUE = 55
CUPS = {42: "ucl", 73: "uel", 10216: "uecl", 141: "coppa", 222: "supercoppa"}
SEASONS = sys.argv[2:] or [seasons.fotmob(y) for y in seasons.all_starts()]   # e.g. "2026/2027"; with no args: whole archive
UPCOMING_DAYS = 10
SETTLE_HOURS = 6   # a match that ended less than this ago is fetched again: FotMob completes xG, shots and player stats after the whistle
PAUSE = 0.5
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
BASE = "https://www.fotmob.com/api/data"
s = requests.Session(); s.headers.update(UA)

def get(url):
    for attempt in range(4):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200: return r.json()
            print("http", r.status_code, url, flush=True)
        except Exception as e:
            print("err", e, url, flush=True)
        time.sleep(5 * (attempt + 1))
    return None

def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wt", encoding="utf-8") as f: json.dump(obj, f)
    os.replace(path + ".tmp", path)

def cached_finished(path):
    """False for a file saved before the match ended (it would otherwise stay without result and statistics for good)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f: return bool(json.load(f)["general"].get("finished"))
    except Exception: return False

now = datetime.now(timezone.utc)
for season in SEASONS:
    tag = season.replace("/", "-")
    lg = get(f"{BASE}/leagues?id={LEAGUE}&season={season.replace('/', '%2F')}")
    if not lg: continue
    save(f"{DATA}/raw/fotmob/{tag}/league.json.gz", {k: lg[k] for k in ("details", "fixtures", "table")})
    todo = []
    for m in lg["fixtures"]["allMatches"]:
        st = m["status"]
        if st.get("cancelled"): continue
        path = f"{DATA}/raw/fotmob/{tag}/{m['id']}.json.gz"
        ko = datetime.fromisoformat(st["utcTime"].replace("Z", "+00:00"))
        if st.get("finished"):
            if not os.path.exists(path) or now - ko < timedelta(hours=SETTLE_HOURS) or not cached_finished(path): todo.append((m["id"], path))
        elif ko < now + timedelta(days=UPCOMING_DAYS):
            todo.append((m["id"], path))
    print(season, "to fetch", len(todo), flush=True)
    for i, (mid, path) in enumerate(todo):
        d = get(f"{BASE}/matchDetails?matchId={mid}")
        if d and "content" in d:
            save(path, {"general": d["general"], "header": d["header"], "content": {k: d["content"].get(k) for k in
                       ("matchFacts", "stats", "playerStats", "shotmap", "lineup", "weather", "attackingZones", "h2h")}})
        if i % 50 == 0: print(season, i, "/", len(todo), flush=True)
        time.sleep(PAUSE)
# --- cups: finished matches involving a Serie A club of the same season (current season always re-listed) ---
for season in SEASONS:
    tag = season.replace("/", "-")
    lgp = f"{DATA}/raw/fotmob/{tag}/league.json.gz"
    if not os.path.exists(lgp): continue
    clubs = {int(x) for m in json.load(gzip.open(lgp, "rt"))["fixtures"]["allMatches"] for x in (m["home"]["id"], m["away"]["id"])}
    for cid, code in CUPS.items():
        idx = f"{DATA}/raw/fotmob_cups/{code}/{tag}/league.json.gz"
        if season != SEASONS[-1] and os.path.exists(idx):
            lg = json.load(gzip.open(idx, "rt"))
        else:
            lg = get(f"{BASE}/leagues?id={cid}&season={season.replace('/', '%2F')}")
            if not lg or "fixtures" not in lg: continue
            lg = {"details": lg["details"], "fixtures": {"allMatches": lg["fixtures"]["allMatches"]}}
            save(idx, lg)
        todo = [m["id"] for m in lg["fixtures"]["allMatches"] if m["status"].get("finished")
                and (int(m["home"]["id"]) in clubs or int(m["away"]["id"]) in clubs)
                and not os.path.exists(f"{DATA}/raw/fotmob_cups/{code}/{tag}/{m['id']}.json.gz")]
        for mid in todo:
            d = get(f"{BASE}/matchDetails?matchId={mid}")
            if d and "content" in d:
                save(f"{DATA}/raw/fotmob_cups/{code}/{tag}/{mid}.json.gz", {"general": d["general"], "header": d["header"],
                     "content": {k: d["content"].get(k) for k in ("matchFacts", "playerStats", "shotmap", "lineup")}})
            time.sleep(PAUSE)
        if todo: print(code, season, "fetched", len(todo), flush=True)
print("done", flush=True)

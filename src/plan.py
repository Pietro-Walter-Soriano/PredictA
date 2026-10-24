"""Decide what the scheduled GitHub run has to do right now, so the heavy cycle (run.sh) only starts when it is useful.

  python src/plan.py data          look at the fixtures and write data/.plan.json; prints GitHub outputs (run, fast, prematch, reasons)
  python src/plan.py data mark     after a successful cycle: record in data/schedule.json what has been done

Triggers, per match (all times UTC):
  lineup   50 min before kick-off, if the official XI is already out -> forecast again with it
  fallback 30 min before kick-off, if it was not (and no forecast was redone yet) -> forecast again with the probable XI
  result   the match is finished and not yet scored -> fetch the final data, score it, publish the KPIs
  settle   ~20 min after that -> once more, FotMob completes xG, shots and player stats a little after the final whistle
  routine  nothing else to do but the last full cycle is older than ROUTINE_MIN -> refresh everything (as the old hourly run)
Time-critical triggers run FAST (no backtest: data/calibration.json and data/backtest.json are the committed ones)."""
import glob, json, os, sys
from datetime import datetime, timedelta, timezone
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seasons

DATA = sys.argv[1]
CMD = sys.argv[2] if len(sys.argv) > 2 else "plan"
STATE, PENDING, ROUNDS = f"{DATA}/schedule.json", f"{DATA}/.plan.json", f"{DATA}/rounds.json"
LINEUP_MIN, FALLBACK_MIN = 50, 30   # minutes before kick-off
MIN_LEAD = 6                        # a cycle started closer than this to kick-off would end after it: the forecast could not be archived
SETTLE_MIN = 20                     # minutes between the first scoring and the settling pass
ROUTINE_MIN = 55
MAX_TRIES = 6                       # result attempts per match
WINDOW_DAYS = 2                     # finished matches older than this are left alone
BASE = "https://www.fotmob.com/api/data"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
now = datetime.now(timezone.utc)
iso = lambda t: t.isoformat(timespec="minutes")

def load(path, default):
    try: return json.load(open(path))
    except (OSError, ValueError): return default

def get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200: return r.json()
            print("http", r.status_code, url, file=sys.stderr)
        except requests.RequestException as e:
            print("err", type(e).__name__, url, file=sys.stderr)
    return None

def scored_ids():
    return {str(m["match_id"]) for r in load(ROUNDS, {"rounds": []})["rounds"] for m in r["matches"]}

def archived_ids():
    return {os.path.basename(f).split("_")[0] for f in glob.glob(f"{DATA}/ledger/*/R*/*_final.json")}

def official_lineup(mid):
    d = get(f"{BASE}/matchDetails?matchId={mid}")
    lu = ((d or {}).get("content") or {}).get("lineup") or {}
    return lu.get("lineupType") == "standard" and all(len((lu.get(s) or {}).get("starters") or []) == 11 for s in ("homeTeam", "awayTeam"))

def out(run, fast, prematch, reasons):
    line = f"run={'true' if run else 'false'}\nfast={'true' if fast else 'false'}\nreasons={' | '.join(reasons) or 'nothing to do'}\n"
    print(line, end="")
    if os.environ.get("GITHUB_OUTPUT"): open(os.environ["GITHUB_OUTPUT"], "a").write(line)

def plan():
    state = load(STATE, {"matches": {}, "last_full": None})
    last_full = datetime.fromisoformat(state["last_full"]) if state.get("last_full") else None
    routine = last_full is None or now - last_full.replace(tzinfo=timezone.utc) >= timedelta(minutes=ROUTINE_MIN)
    reasons, marks = [], {}
    lg = get(f"{BASE}/leagues?id=55&season={seasons.fotmob(seasons.current_start()).replace('/', '%2F')}")
    scored, archived = scored_ids(), archived_ids()
    for m in ((lg or {}).get("fixtures") or {}).get("allMatches", []):
        st = m["status"]
        if st.get("cancelled"): continue
        mid, ko = str(m["id"]), datetime.fromisoformat(st["utcTime"].replace("Z", "+00:00"))
        s, mins, name = state["matches"].get(mid, {}), (ko - now).total_seconds() / 60, f"{m['home']['name']}-{m['away']['name']}"
        base = {"ko": iso(ko)}
        if not st.get("finished") and MIN_LEAD <= mins <= LINEUP_MIN and not s.get("t50") and not s.get("t30"):
            if mins <= FALLBACK_MIN:
                reasons.append(f"{name} T-{mins:.0f}: forecast with the XI available"); marks[mid] = base | {"t30": 1}
            elif official_lineup(mid):
                reasons.append(f"{name} T-{mins:.0f}: official XI out, forecast again"); marks[mid] = base | {"t50": 1}
        if st.get("finished") and now - ko < timedelta(days=WINDOW_DAYS) and not s.get("settled"):
            if mid not in scored:
                if mid in archived and s.get("tries", 0) < MAX_TRIES:   # without a pre-kick-off forecast there is nothing to score
                    reasons.append(f"{name}: finished, score it"); marks[mid] = base | {"tries": 1}
            elif not s.get("scored_at") or now >= datetime.fromisoformat(s["scored_at"]).replace(tzinfo=timezone.utc) + timedelta(minutes=SETTLE_MIN):
                reasons.append(f"{name}: settle final stats"); marks[mid] = base | {"settled": 1}
    fast = bool(reasons)
    prematch = any(("t50" in v or "t30" in v) for v in marks.values())   # odds are metered: fetch them only when a forecast is redone
    if not reasons and routine: reasons.append("routine cycle")
    json.dump({"marks": marks, "full": not fast}, open(PENDING, "w"))
    out(bool(reasons) or os.environ.get("FORCE") == "true", fast, prematch, reasons)

def mark():
    pend, state = load(PENDING, None), load(STATE, {"matches": {}, "last_full": None})
    if pend is None: return
    scored, stamp = scored_ids(), iso(now)
    for mid, upd in pend["marks"].items():
        s = state["matches"].setdefault(mid, {})
        s["ko"] = upd["ko"]
        for k in ("t50", "t30", "settled"):
            if k in upd: s[k] = stamp
        if "tries" in upd:
            s["tries"] = s.get("tries", 0) + 1
            if mid in scored: s["scored_at"] = s.get("scored_at") or stamp
        if "settled" in upd and not s.get("scored_at"): s.pop("settled", None)   # not scored yet: settle later
    if pend.get("full"): state["last_full"] = stamp
    keep = iso(now - timedelta(days=14))
    state["matches"] = {k: v for k, v in sorted(state["matches"].items()) if v.get("ko", keep) >= keep}
    json.dump(state, open(STATE, "w"), indent=1)
    os.remove(PENDING)
    print("schedule: marked", len(pend["marks"]), "matches")

mark() if CMD == "mark" else plan()

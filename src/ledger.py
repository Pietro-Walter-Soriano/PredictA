"""Archive forecasts per round. A forecast is stored only if made before kick-off, then never touched again:
  <id>_first.json  first forecast of the week (kept as is)
  <id>_final.json  latest forecast before kick-off (usually with the official XI)"""
import glob, json, os, sys
from datetime import datetime

SRC, LEDGER = sys.argv[1], sys.argv[2]
PLAYER_KEYS = ("id", "name", "grp", "starter", "exp_min", "props")   # what scoring needs; player history stays in the live board only

def slim(d):
    d = dict(d)
    d["teams"] = {s: {k: v for k, v in t.items() if k in ("name", "formation", "lineup_feats")} |
                     {"players": [{k: p.get(k) for k in PLAYER_KEYS} for p in t["players"]]} for s, t in d["teams"].items()}
    return d

saved = 0
for f in glob.glob(f"{SRC}/2*.json"):
    d = json.load(open(f))
    made, ko = datetime.fromisoformat(d["generated"].rstrip("Z")), datetime.fromisoformat(d["date"].rstrip("Z"))
    if made >= ko: continue
    base = f"{LEDGER}/{d['season']}/R{d['round']:02d}/{d['match_id']}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    out = json.dumps(slim(d), ensure_ascii=False, separators=(",", ":"))
    if not os.path.exists(base + "_first.json"): open(base + "_first.json", "w").write(out)
    open(base + "_final.json", "w").write(out); saved += 1
print("ledger: saved", saved, "pre-kickoff forecasts")

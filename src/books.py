"""Extra bookmaker markets from The Odds API (per-event endpoint), read only to be compared with our forecast, never used by it.

fetch_odds.py saves one file per event and fetch (data/raw/odds_extra/<eventId>_<yyyymmddThhmm>.json, the event as the API returns it).
parse_event() turns it into de-vigged probabilities, the median across bookmakers for each line:
  goals    Over probability of the total goals, by line (alternate_totals / totals)
  btts     both teams to score, Yes
  corners  Over probability of the total corners, by line
  cards    Over probability of the total cards, by line
load_extra() / find() match those files with our fixtures by team names and kick-off."""
import glob, json, os, statistics
from datetime import datetime, timedelta

MARKETS = {"totals": "goals", "alternate_totals": "goals", "alternate_totals_corners": "corners", "alternate_totals_cards": "cards"}
OA_NAME = {"AC Milan": "Milan", "Inter Milan": "Inter", "AS Roma": "Roma", "Hellas Verona FC": "Hellas Verona", "Verona": "Hellas Verona",
           "SS Lazio": "Lazio", "SSC Napoli": "Napoli", "Atalanta BC": "Atalanta", "Parma Calcio 1913": "Parma"}


def canon(name, known):
    n = OA_NAME.get(name, name)
    if n in known: return n
    low = n.lower()
    return next((k for k in known if k.lower().startswith(low) or low.startswith(k.lower())), n)


def _devig(over, under):
    a, b = 1 / float(over), 1 / float(under)
    return a / (a + b)


def parse_event(ev):
    """{'goals': {'2.5': p}, 'corners': {...}, 'cards': {...}, 'btts': p or None, 'books': n}; empty dicts where the market is not quoted."""
    lines, btts, names = {}, {}, set()
    for b in ev.get("bookmakers") or []:
        for m in b.get("markets") or []:
            key, outs = m.get("key"), m.get("outcomes") or []
            if key == "btts":
                pr = {o.get("name"): o.get("price") for o in outs if o.get("price")}
                if pr.get("Yes") and pr.get("No"): btts[b["key"]] = _devig(pr["Yes"], pr["No"]); names.add(b["key"])
            elif key in MARKETS:
                by = {}
                for o in outs:
                    if o.get("point") is None or not o.get("price"): continue
                    by.setdefault(float(o["point"]), {})[o.get("name")] = o["price"]
                for pt, pr in by.items():
                    if pr.get("Over") and pr.get("Under"):
                        lines.setdefault((MARKETS[key], pt), {})[b["key"]] = _devig(pr["Over"], pr["Under"]); names.add(b["key"])
    out = {"goals": {}, "corners": {}, "cards": {}, "btts": None, "books": len(names)}
    for (mk, pt), per_book in lines.items(): out[mk][str(pt)] = round(float(statistics.median(per_book.values())), 4)
    if btts: out["btts"] = round(float(statistics.median(btts.values())), 4)
    return out


def load_extra(data, known, days=4):
    """Latest parsed snapshot per fixture: {(home, away): [(kickoff, parsed), ...]} from the files of the last `days` days."""
    best = {}
    for f in glob.glob(f"{data}/raw/odds_extra/*.json"):
        try:
            taken = datetime.strptime(os.path.basename(f).split("_")[-1][:13], "%Y%m%dT%H%M")
            ev = json.load(open(f, encoding="utf-8"))
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, KeyError, OSError, IndexError): continue
        if taken >= ko or datetime.utcnow() - ko > timedelta(days=days): continue          # only prices taken before kick-off
        k = (canon(ev.get("home_team", ""), known), canon(ev.get("away_team", ""), known), ko)
        if k not in best or taken > best[k][0]: best[k] = (taken, ev)
    out = {}
    for (h, a, ko), (_, ev) in best.items():
        p = parse_event(ev)
        if p["books"]: out.setdefault((h, a), []).append((ko, p))
    return out


def find(extra, home, away, ko):
    """The parsed snapshot of this fixture (kick-off within a day of ours), or None."""
    ko = ko.replace(tzinfo=None) if getattr(ko, "tzinfo", None) else ko
    for k, p in extra.get((home, away), []):
        if abs(k - ko) <= timedelta(days=1): return p
    return None

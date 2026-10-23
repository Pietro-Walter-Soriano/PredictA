"""Normalize cached FotMob / Understat / ESPN files into three tables: matches, team_match, player_match (pickle)."""
import glob, gzip, json, os, re, sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DATA = sys.argv[1]
NAME = {"AC Milan": "Milan", "Parma Calcio 1913": "Parma", "Verona": "Hellas Verona", "AS Roma": "Roma", "Internazionale": "Inter"}
canon = lambda n: NAME.get(n, n)
load = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))
POS = {0: "GK", 1: "D", 2: "M", 3: "F"}

TEAM_STATS = {  # FotMob stat key -> column
    "BallPossesion": "poss", "expected_goals": "xg", "expected_goals_non_penalty": "npxg", "expected_goals_open_play": "xg_open",
    "expected_goals_set_play": "xg_set", "expected_goals_on_target": "xgot", "total_shots": "shots", "ShotsOnTarget": "sot",
    "shots_inside_box": "shots_box", "big_chance": "big_ch", "touches_opp_box": "box_touches", "passes": "passes",
    "accurate_passes": "acc_passes", "own_half_passes": "own_half_passes", "opposition_half_passes": "opp_half_passes",
    "long_balls_accurate": "long_balls", "accurate_crosses": "crosses", "corners": "corners", "Offsides": "offsides",
    "matchstats.headers.tackles": "tackles", "interceptions": "interceptions", "clearances": "clearances", "keeper_saves": "saves",
    "duel_won": "duels_won", "aerials_won": "aerials", "dribbles_succeeded": "dribbles", "yellow_cards": "yc", "red_cards": "rc", "fouls": "fouls",
}
PLAYER_STATS = {
    "FotMob rating": "rating", "Minutes played": "min", "Goals": "g", "Assists": "a", "Expected goals (xG)": "xg", "xG Non-penalty": "npxg",
    "Expected assists (xA)": "xa", "Total shots": "sh", "Shots on target": "sot", "Touches": "touches", "Touches in opposition box": "box_touches",
    "Chances created": "kp", "Big chances created": "bcc", "Passes into final third": "final_third", "Accurate passes": "acc_passes",
    "Accurate crosses": "crosses", "Successful dribbles": "dribbles", "Tackles": "tackles", "Interceptions": "interceptions",
    "Recoveries": "recoveries", "Clearances": "clearances", "Blocks": "blocks", "Defensive actions": "def_actions", "Duels won": "duels_won",
    "Duels lost": "duels_lost", "Aerial duels won": "aerials", "Fouls committed": "fouls", "Was fouled": "fouled", "Dispossessed": "dispossessed",
    "Saves": "saves", "Goals conceded": "conceded", "xGOT faced": "xgot_faced", "Penalties won": "pen_won", "Conceded penalty": "pen_conceded",
}

def num(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    m = re.match(r"\s*(-?[\d.]+)", str(v))
    return float(m.group(1)) if m else None

def players_of(d, mid, ko, season, comp):
    """Per-player rows of one finished match (FotMob stats; cards from the match events; starter flag from the lineup)."""
    g, c = d["general"], d["content"]
    side_of = {g["homeTeam"]["id"]: "home", g["awayTeam"]["id"]: "away"}
    team_of = {g["homeTeam"]["id"]: canon(g["homeTeam"]["name"]), g["awayTeam"]["id"]: canon(g["awayTeam"]["name"])}
    evs = ((c.get("matchFacts") or {}).get("events") or {}).get("events") or []
    pens = [s for s in (c.get("shotmap") or {}).get("shots", []) if s.get("situation") == "Penalty"]
    cards = {}
    for e in evs:
        if e["type"] == "Card" and (e.get("player") or {}).get("id"):
            k = "rc" if e.get("card") in ("Red", "YellowRed") else "yc"
            cards.setdefault(e["player"]["id"], {"yc": 0, "rc": 0})[k] += 1
    lu = c.get("lineup") or {}
    xi = {s["id"]: s for side in ("homeTeam", "awayTeam") for s in (lu.get(side) or {}).get("starters") or [] if "id" in s}
    has_xg = bool((c.get("shotmap") or {}).get("shots"))  # older Coppa Italia matches carry ratings but no xG/shot data
    rows = []
    for p in (c.get("playerStats") or {}).values():
        if not p.get("stats"): continue
        pr = {"match_id": mid, "date": ko, "season": season, "comp": comp, "player_id": p["id"], "name": p["name"],
              "team": team_of.get(p["teamId"]), "venue": side_of.get(p["teamId"]), "grp": POS.get(p.get("usualPosition"), "M"),
              "starter": p["id"] in xi, "slot": (xi.get(p["id"]) or {}).get("positionId"), "mv": (xi.get(p["id"]) or {}).get("marketValue")}
        for grp in p["stats"]:
            for k, v in grp["stats"].items():
                if k in PLAYER_STATS: pr[PLAYER_STATS[k]] = num(v["stat"].get("value"))
        pr.update(cards.get(p["id"], {"yc": 0, "rc": 0}))
        pr["pen_att"] = sum(1 for sh in pens if sh.get("playerId") == p["id"])
        pr["has_xg"] = has_xg
        rows.append(pr)
    return rows

matches, team_rows, player_rows = [], [], []
for path in sorted(glob.glob(f"{DATA}/raw/fotmob/*/*.json.gz")):
    if path.endswith("league.json.gz"): continue
    d = load(path); g, c = d["general"], d["content"]
    season = path.split("/")[-2]
    mid = int(g["matchId"]); home, away = canon(g["homeTeam"]["name"]), canon(g["awayTeam"]["name"])
    ids = {g["homeTeam"]["id"]: "home", g["awayTeam"]["id"]: "away"}
    ko = datetime.fromisoformat(g["matchTimeUTCDate"].replace("Z", "+00:00")).replace(tzinfo=None)
    mf = c.get("matchFacts") or {}; info = mf.get("infoBox") or {}
    ref = (info.get("Referee") or {}).get("text")
    lu = c.get("lineup") or {}
    row = {"match_id": mid, "season": season, "date": ko, "round": int(num(g.get("matchRound")) or 0), "home": home, "away": away,
           "home_fm_id": g["homeTeam"]["id"], "away_fm_id": g["awayTeam"]["id"], "finished": bool(g.get("finished")),
           "lineup_type": lu.get("lineupType"), "referee": ref,
           "attendance": num((info.get("Attendance") or None)), "temp": (c.get("weather") or {}).get("temperature"),
           "wind": (c.get("weather") or {}).get("windSpeed"), "rain": (c.get("weather") or {}).get("precipitation")}
    if g.get("finished"):
        sc = {t["id"]: t["score"] for t in d["header"]["teams"]}
        row["hg"], row["ag"] = sc[g["homeTeam"]["id"]], sc[g["awayTeam"]["id"]]
        evs = (mf.get("events") or {}).get("events") or []
        half = next((i for i, e in enumerate(evs) if e["type"] == "Half"), len(evs))
        row["ht_hg"] = sum(1 for e in evs[:half] if e["type"] == "Goal" and e.get("isHome"))
        row["ht_ag"] = sum(1 for e in evs[:half] if e["type"] == "Goal" and not e.get("isHome"))
        pens = [s for s in (c.get("shotmap") or {}).get("shots", []) if s.get("situation") == "Penalty"]
        row["home_pens"] = sum(1 for s in pens if ids.get(s["teamId"]) == "home")
        row["away_pens"] = sum(1 for s in pens if ids.get(s["teamId"]) == "away")
        # team stats
        st = {}
        for grp in ((c.get("stats") or {}).get("Periods", {}).get("All", {}).get("stats") or []):
            for s in grp["stats"]:
                if s.get("key") in TEAM_STATS and s.get("stats") and s["stats"][0] is not None:
                    st[TEAM_STATS[s["key"]]] = (num(s["stats"][0]), num(s["stats"][1]))
        shots = (c.get("shotmap") or {}).get("shots", [])
        for side, gf, ga in (("home", row["hg"], row["ag"]), ("away", row["ag"], row["hg"])):
            i = 0 if side == "home" else 1
            fm_id = g[f"{side}Team"]["id"]
            tr = {"match_id": mid, "season": season, "date": ko, "team": home if side == "home" else away, "opp": away if side == "home" else home,
                  "venue": side, "gf": gf, "ga": ga, "referee": ref,
                  "formation": (lu.get(f"{side}Team") or {}).get("formation"), "fm_rating": (lu.get(f"{side}Team") or {}).get("rating")}
            for k, v in st.items(): tr[k], tr["opp_" + k] = v[i], v[1 - i]
            own = [s for s in shots if s["teamId"] == fm_id]
            tr["shot_xg_fastbreak"] = sum(s["expectedGoals"] or 0 for s in own if s.get("situation") == "FastBreak")
            tr["shot_xg_corner"] = sum(s["expectedGoals"] or 0 for s in own if s.get("situation") == "FromCorner")
            tr["shot_xg_head"] = sum(s["expectedGoals"] or 0 for s in own if s.get("shotType") == "Header")
            tr["shots_1h"] = sum(1 for s in own if s.get("period") == "FirstHalf")
            team_rows.append(tr)
        player_rows.extend(players_of(d, mid, ko, season, "serie_a"))
    # lineup info (starter flag, position slot, market value, age), also for upcoming matches
    row["lineup"] = {}
    for side in ("home", "away"):
        t = lu.get(f"{side}Team") or {}
        row[f"{side}_formation"] = t.get("formation")
        row["lineup"][side] = {"starters": [{"id": s["id"], "name": s["name"], "pos": s.get("positionId"), "grp": POS.get(s.get("usualPlayingPositionId"), "GK" if s.get("positionId") == 11 else "M"),
                                             "no": s.get("shirtNumber"), "age": s.get("age"), "mv": s.get("marketValue"), "cap": s.get("isCaptain", False),
                                             "x": (s.get("verticalLayout") or {}).get("x"), "y": (s.get("verticalLayout") or {}).get("y")} for s in t.get("starters") or []],
                               "subs": [{"id": s["id"], "name": s["name"], "grp": POS.get(s.get("usualPlayingPositionId"), "M"), "no": s.get("shirtNumber"),
                                         "age": s.get("age"), "mv": s.get("marketValue")} for s in t.get("subs") or []],
                               "unavailable": [{"id": u["id"], "name": u["name"], "type": (u.get("unavailability") or {}).get("type"),
                                                "ret": (u.get("unavailability") or {}).get("expectedReturn")} for u in t.get("unavailable") or [] if u.get("positionId") != 1100],
                               "coach": (t.get("coach") or {}).get("name")}
    matches.append(row)

M = pd.DataFrame(matches).sort_values("date").reset_index(drop=True)
T = pd.DataFrame(team_rows).sort_values("date").reset_index(drop=True)
# cup matches of Serie A clubs: player rows only (player form/load is the same whatever the competition)
for path in sorted(glob.glob(f"{DATA}/raw/fotmob_cups/*/*/*.json.gz")):
    if path.endswith("league.json.gz"): continue
    d = load(path); g = d["general"]
    if not g.get("finished"): continue
    ko = datetime.fromisoformat(g["matchTimeUTCDate"].replace("Z", "+00:00")).replace(tzinfo=None)
    player_rows.extend(players_of(d, int(g["matchId"]), ko, path.split("/")[-2], path.split("/")[-3]))
P = pd.DataFrame(player_rows).drop_duplicates(["match_id", "player_id"]).sort_values("date").reset_index(drop=True)
COUNTS = [c for c in PLAYER_STATS.values() if c not in ("rating", "min", "xgot_faced", "saves", "conceded")]
P[COUNTS] = P[COUNTS].fillna(0)  # FotMob omits a counting stat when it is zero
SHOT_STATS = ["xg", "npxg", "xa", "sh", "sot", "bcc"]
P.loc[~P.has_xg.astype(bool), SHOT_STATS] = np.nan  # ...but a match without shot data must not count as zero xG

# Understat: pressing (PPDA) and deep completions per team-match
us = []
for f in glob.glob(f"{DATA}/raw/understat/*/league.json.gz"):
    for t in load(f)["teams"].values():
        for h in t["history"]:
            us.append({"team": canon(t["title"]), "day": h["date"][:10], "ppda": h["ppda"]["att"] / max(h["ppda"]["def"], 1),
                       "ppda_allowed": h["ppda_allowed"]["att"] / max(h["ppda_allowed"]["def"], 1), "deep": h["deep"], "deep_allowed": h["deep_allowed"]})
U = pd.DataFrame(us)
T["day"] = T.date.dt.strftime("%Y-%m-%d")
T = T.merge(U, on=["team", "day"], how="left")
for dd in (-1, 1):  # late kick-offs can sit on the next UTC day
    miss = T.ppda.isna()
    alt = T.loc[miss, ["team", "date"]].assign(day=(T.loc[miss, "date"] + timedelta(days=dd)).dt.strftime("%Y-%m-%d")).merge(U, on=["team", "day"], how="left")
    for col in ("ppda", "ppda_allowed", "deep", "deep_allowed"): T.loc[miss, col] = alt[col].values

# ESPN odds (DraftKings): open/close 1X2, O/U line and prices
def amer(o):
    try: o = float(str(o).replace("+", ""))
    except (TypeError, ValueError): return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)
odds = []
for f in glob.glob(f"{DATA}/raw/espn/sb_*.json.gz"):
    for e in load(f).get("events", []):
        comp = e["competitions"][0]; tm = {c["homeAway"]: canon(c["team"]["displayName"]) for c in comp["competitors"]}
        o = (comp.get("odds") or [None])[0]
        of = f"{DATA}/raw/espn/odds_{e['id']}.json.gz"
        if glob.glob(of):
            od = load(of); o = (od.get("odds") or [None])[0] or o
            pc = (od.get("pickcenter") or [None])[0]
        else: pc = None
        if not o: continue
        ml = o.get("moneyline") or {}
        pick = lambda side, kind: amer(((ml.get(side) or {}).get(kind) or {}).get("odds"))
        tot = o.get("total") or {}
        rec = {"day": e["date"][:10], "home": tm["home"], "away": tm["away"], "espn_id": e["id"], "ou_line": o.get("overUnder"),
               "p1_close": pick("home", "close"), "px_close": pick("draw", "close"), "p2_close": pick("away", "close"),
               "p1_open": pick("home", "open"), "px_open": pick("draw", "open"), "p2_open": pick("away", "open"),
               "pover": amer(((tot.get("over") or {}).get("close") or {}).get("odds")) or (amer(pc.get("overOdds")) if pc else None),
               "punder": amer(((tot.get("under") or {}).get("close") or {}).get("odds")) or (amer(pc.get("underOdds")) if pc else None)}
        odds.append(rec)
O = pd.DataFrame(odds).drop_duplicates(["home", "away", "day"], keep="last")
M["day"] = M.date.dt.strftime("%Y-%m-%d")
M = M.merge(O, on=["home", "away", "day"], how="left")
miss = M.espn_id.isna()
if miss.any():
    alt = M.loc[miss, ["home", "away", "date"]].assign(day=(M.loc[miss, "date"] + timedelta(days=1)).dt.strftime("%Y-%m-%d")).merge(O, on=["home", "away", "day"], how="left")
    for col in O.columns.drop(["home", "away", "day"]): M.loc[miss, col] = alt[col].values

# Better bookmakers override DraftKings where available. Priority: Pinnacle closing > market average > Bet365 (football-data.co.uk),
# then The Odds API pre-match snapshot (latest before kick-off, Pinnacle if listed, else the median bookmaker).
FD_NAME = {"Verona": "Hellas Verona", "AC Milan": "Milan", "Milan": "Milan", "Inter": "Inter", "Roma": "Roma", "Parma": "Parma", "Spal": "SPAL"}
OA_NAME = {"AC Milan": "Milan", "Inter Milan": "Inter", "AS Roma": "Roma", "Hellas Verona FC": "Hellas Verona", "Verona": "Hellas Verona",
           "SS Lazio": "Lazio", "SSC Napoli": "Napoli", "Atalanta BC": "Atalanta", "Parma Calcio 1913": "Parma"}
known = set(M.home) | set(M.away)
def fd_team(n):
    n = FD_NAME.get(n, n)
    return n if n in known else next((k for k in known if k.lower().startswith(n.lower()) or n.lower().startswith(k.lower())), n)
def inv(*odds):
    try:
        v = [1 / float(o) for o in odds]
        return v if all(x > 0 for x in v) else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None
ext = []
for f in sorted(glob.glob(f"{DATA}/raw/football_data/I1_*.csv")):
    for r in pd.read_csv(f, encoding="latin-1").to_dict("records"):
        if pd.isna(r.get("HomeTeam")) or pd.isna(r.get("Date")): continue
        day = pd.to_datetime(r["Date"], dayfirst=True).strftime("%Y-%m-%d")
        for src, cols, ou in (("pinnacle", ("PSCH", "PSCD", "PSCA"), ("PC>2.5", "PC<2.5")), ("pinnacle", ("PSH", "PSD", "PSA"), ("P>2.5", "P<2.5")),
                              ("market_avg", ("AvgCH", "AvgCD", "AvgCA"), ("AvgC>2.5", "AvgC<2.5")), ("market_avg", ("AvgH", "AvgD", "AvgA"), ("Avg>2.5", "Avg<2.5")),
                              ("bet365", ("B365CH", "B365CD", "B365CA"), ("B365C>2.5", "B365C<2.5")), ("bet365", ("B365H", "B365D", "B365A"), ("B365>2.5", "B365<2.5"))):
            p = inv(*(r.get(c) for c in cols))
            if p:
                o = inv(*(r.get(c) for c in ou)) or [None, None]
                ext.append({"home": fd_team(r["HomeTeam"]), "away": fd_team(r["AwayTeam"]), "day": day, "src": src,
                            "x1": p[0], "xx": p[1], "x2": p[2], "xline": 2.5 if o[0] else None, "xover": o[0], "xunder": o[1]})
                break
snaps = sorted(glob.glob(f"{DATA}/raw/odds_api/*.json"))
for f in snaps:
    taken = datetime.strptime(os.path.basename(f)[:13], "%Y%m%dT%H%M")
    for ev in json.load(open(f)):
        ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        if taken >= ko: continue
        books = {b["key"]: b for b in ev.get("bookmakers", [])}
        def mk(b, key):
            m = next((m for m in b["markets"] if m["key"] == key), None)
            return {o["name"] + str(o.get("point", "")): o["price"] for o in m["outcomes"]} if m else None
        h, a = OA_NAME.get(ev["home_team"], ev["home_team"]), OA_NAME.get(ev["away_team"], ev["away_team"])
        rows = [(k, mk(b, "h2h"), mk(b, "totals")) for k, b in books.items()]
        rows = [x for x in rows if x[1] and ev["home_team"] in x[1] and ev["away_team"] in x[1] and "Draw" in x[1]]
        if not rows: continue
        pick = next((x for x in rows if x[0] == "pinnacle"), None)
        if pick: k, h2h, tot = pick
        else:  # median bookmaker by home price
            k, h2h, tot = sorted(rows, key=lambda x: x[1][ev["home_team"]])[len(rows) // 2]
        p = inv(h2h[ev["home_team"]], h2h["Draw"], h2h[ev["away_team"]])
        o = inv((tot or {}).get("Over2.5"), (tot or {}).get("Under2.5")) or [None, None]
        ext.append({"home": h, "away": a, "day": ko.strftime("%Y-%m-%d"), "src": f"odds_api:{k}", "taken": taken,
                    "x1": p[0], "xx": p[1], "x2": p[2], "xline": 2.5 if o[0] else None, "xover": o[0], "xunder": o[1]})
if ext:
    E = pd.DataFrame(ext)
    E["rank"] = E.src.map(lambda s: 0 if s == "pinnacle" else 1 if s == "market_avg" else 2 if s == "bet365" else 3 if "pinnacle" in s else 4)
    if "taken" not in E: E["taken"] = pd.NaT
    E = E.sort_values(["rank", "taken"], ascending=[True, False]).drop_duplicates(["home", "away", "day"])
    keys = list(zip(M.home, M.away, M.date.dt.strftime("%Y-%m-%d"))) ; idx = {k: i for i, k in enumerate(zip(E.home, E.away, E.day))}
    hit = 0
    for i, (h, a, d) in enumerate(keys):
        j = idx.get((h, a, d)) if (h, a, d) in idx else idx.get((h, a, (pd.Timestamp(d) - timedelta(days=1)).strftime("%Y-%m-%d")))
        if j is None: continue
        e = E.iloc[j]; hit += 1
        M.loc[i, ["p1_close", "px_close", "p2_close"]] = [e.x1, e.xx, e.x2]
        M.loc[i, "odds_src"] = e.src
        if e.xover: M.loc[i, ["ou_line", "pover", "punder"]] = [e.xline, e.xover, e.xunder]
    unmatched = sorted({(h, a) for h, a in zip(E.home, E.away) if h not in known or a not in known})
    print("external odds rows", len(E), "matched", hit, "unmatched team names", unmatched[:8])
M["odds_src"] = M.get("odds_src", pd.Series(index=M.index, dtype=object)).where(M.p1_close.notna() & M.get("odds_src", pd.Series(index=M.index)).notna(),
                                                                              np.where(M.p1_close.notna(), "draftkings", None))

pd.to_pickle({"matches": M, "team_match": T, "player_match": P}, f"{DATA}/dataset.pkl")
print("matches", len(M), "finished", int(M.finished.sum()), "seasons", sorted(M.season.unique()))
print("team rows", len(T), "with xg", int(T.xg.notna().sum()), "with ppda", int(T.ppda.notna().sum()))
print("players rows", len(P), "with rating", int(P.rating.notna().sum()), "by comp", P.comp.value_counts().to_dict())
print("odds matched", int(M.p1_close.notna().sum()), M.odds_src.value_counts().to_dict(), "pens found", int((M.home_pens.fillna(0) + M.away_pens.fillna(0)).sum()))
print(M[~M.finished][["date", "home", "away", "lineup_type", "referee", "p1_close", "ou_line"]].head(12).to_string())

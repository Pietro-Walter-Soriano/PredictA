"""Score every completed round of the ledger against actual results and write data/rounds.json.

KPIs per round (and cumulative):
  - 1X2: hit rate of the favourite, log-loss, Brier, RPS; same for the bookmaker and for the "always home" baseline
  - exact score: hit of the top-1 / top-3 scores
  - binary markets (O1.5/2.5/3.5, BTTS, clean sheets): hit rate at 50% threshold and Brier
  - counts (goals, corners, cards, shots, shots on target, fouls): MAE vs actual and vs league-average baseline,
    and hit rate of the O/U line closest to 50%
  - players: anytime scorer (Brier, how many actual scorers were in our top-3 per team, hit rate of picks >= 35%)
  - calibration buckets (predicted vs observed) for every binary event of the round
  - first vs final forecast (did the official XI improve us?)
  - value: flat 1-unit stake on every 1X2 outcome where our probability beats the bookmaker's (fair odds), ROI
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

LEDGER, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EDGE = 0.03          # minimum probability edge over the market to count as a "value" pick
D = pd.read_pickle(f"{DATA}/dataset.pkl")
M = D["matches"].set_index("match_id"); T = D["team_match"].set_index(["match_id", "venue"]); P = D["player_match"]
# naive references come from the seasons BEFORE the one being scored, so no scored match leaks into its own baseline
SEASON_NOW = M.season.max()
prior = M[M.finished & (M.season < SEASON_NOW)]
BASE = {"1": float((prior.hg > prior.ag).mean()), "X": float((prior.hg == prior.ag).mean()), "2": float((prior.hg < prior.ag).mean())}  # naive "climatology"
GOAL_EVENTS = {"Over 1.5": lambda h, a: h + a > 1.5, "Over 2.5": lambda h, a: h + a > 2.5, "Over 3.5": lambda h, a: h + a > 3.5,
               "Gol entrambe": lambda h, a: h > 0 and a > 0, "Porta inviolata casa": lambda h, a: a == 0, "Porta inviolata ospite": lambda h, a: h == 0}
GOAL_BASE = {k: float(np.mean([f(h, a) for h, a in zip(prior.hg, prior.ag)])) for k, f in GOAL_EVENTS.items()}
FIXED_LINES = {"corners": 9.5, "cards": 4.5, "sot": 8.5, "shots": 24.5, "fouls": 24.5}   # standard book lines, scored at a fixed threshold
STATS = {"corners": "corners", "cards": "yc", "sot": "sot", "shots": "shots", "fouls": "fouls", "offsides": "offsides"}
_tp = D["team_match"][D["team_match"].season < SEASON_NOW].groupby("match_id")[["corners", "yc", "sot", "shots", "fouls"]].sum()
STAT_BASE = {k: float((_tp[STATS[k]] > l).mean()) for k, l in FIXED_LINES.items()}

def ll(p): return -math.log(max(p, 1e-9))
def mean(x): x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]; return sum(x) / len(x) if x else None

def score_match(d, kind):
    mid = d["match_id"]
    if mid not in M.index or not M.at[mid, "finished"]: return None
    hg, ag = int(M.at[mid, "hg"]), int(M.at[mid, "ag"])
    res = "1" if hg > ag else "X" if hg == ag else "2"
    p = d["p"]; o = {k: float(k == res) for k in "1X2"}
    fav = max("1X2", key=lambda k: p[k])
    cum = lambda q: (q["1"], q["1"] + q["X"])
    rps = lambda q: ((cum(q)[0] - cum(o)[0]) ** 2 + (cum(q)[1] - cum(o)[1]) ** 2) / 2
    r = {"match_id": mid, "kind": kind, "home": d["home"], "away": d["away"], "score": f"{hg}-{ag}", "res": res,
         "p": p, "fav": fav, "fav_hit": fav == res, "p_res": p[res], "logloss": ll(p[res]),
         "brier": sum((p[k] - o[k]) ** 2 for k in "1X2"), "rps": rps(p), "lineup_type": d["lineup_type"],
         "lam": d["lambda"], "exp_goals": d["lambda"]["home"] + d["lambda"]["away"], "goals": hg + ag}
    if d.get("p_market"):
        q = {k: d["p_market"][k] for k in "1X2"}
        r.update({"mk_p": q, "mk_fav_hit": max("1X2", key=lambda k: q[k]) == res, "mk_logloss": ll(q[res]),
                  "mk_brier": sum((q[k] - o[k]) ** 2 for k in "1X2"), "mk_rps": rps(q)})
        # value bets at fair odds 1/q: stake 1 where our p exceeds the market by EDGE
        bets = [(k, p[k], q[k]) for k in "1X2" if p[k] - q[k] >= EDGE]
        r["bets"] = [{"pick": k, "p": pk, "q": qk, "odds": round(1 / qk, 2), "won": k == res, "pnl": (1 / qk - 1) if k == res else -1.0} for k, pk, qk in bets]
    top = [s["s"] for s in d["top_scores"]]
    r["top1_hit"], r["top3_hit"] = top[0] == f"{hg}-{ag}", f"{hg}-{ag}" in top[:3]
    r["top_pred"] = top[0]
    m = d["markets"]
    pg = {"Over 1.5": m["ou"]["1.5"], "Over 2.5": m["ou"]["2.5"], "Over 3.5": m["ou"]["3.5"], "Gol entrambe": m["btts"],
          "Porta inviolata casa": m["cs"]["home"], "Porta inviolata ospite": m["cs"]["away"]}
    binary = {k: (pg[k], f(hg, ag)) for k, f in GOAL_EVENTS.items()}
    r["skill"] = {"goal": [((pg[k] - float(f(hg, ag))) ** 2, (GOAL_BASE[k] - float(f(hg, ag))) ** 2) for k, f in GOAL_EVENTS.items()],
                  "stat": [], "scorer": []}
    if (mid, "home") in T.index:
        th, ta = T.loc[(mid, "home")], T.loc[(mid, "away")]
        r["counts"] = {}
        for name, col in STATS.items():
            act = th.get(col) + ta.get(col) if not pd.isna(th.get(col)) else None
            if act is None or name not in d["lines"]: continue
            ln = d["lines"][name]
            line = min(ln["over"], key=lambda l: abs(ln["over"][l] - 0.5))   # the line where we are closest to a coin flip
            r["counts"][name] = {"pred": ln["mean"], "act": act, "abs_err": abs(ln["mean"] - act), "line": float(line), "p_over": ln["over"][line]}
            binary[f"{ {'corners': 'Angoli', 'cards': 'Gialli', 'sot': 'Tiri in porta', 'shots': 'Tiri', 'fouls': 'Falli', 'offsides': 'Fuorigioco'}[name]} over {line}"] = (ln["over"][line], act > float(line))
            if name in FIXED_LINES and str(FIXED_LINES[name]) in ln["over"]:
                y = float(act > FIXED_LINES[name])
                r["skill"]["stat"].append(((ln["over"][str(FIXED_LINES[name])] - y) ** 2, (STAT_BASE[name] - y) ** 2))
        r["counts"]["goals"] = {"pred": r["exp_goals"], "act": hg + ag, "abs_err": abs(r["exp_goals"] - hg - ag)}
        r["xg_act"] = {"home": th.get("npxg"), "away": ta.get("npxg")}
    r["binary"] = {k: {"p": float(v[0]), "y": bool(v[1]), "hit": (v[0] >= .5) == bool(v[1]), "brier": (v[0] - float(v[1])) ** 2} for k, v in binary.items()}
    # players: anytime scorer
    act_pl = P[P.match_id == mid].set_index("player_id")
    scorers = set(act_pl.index[act_pl.g > 0])
    pl, top3_hits, picks = [], 0, []
    for side in ("home", "away"):
        cand = [x for x in d["teams"][side]["players"] if x.get("props")]
        for x in cand:
            played = x["id"] in act_pl.index and act_pl.at[x["id"], "min"] > 0 if x["id"] in act_pl.index else False
            g = int(act_pl.at[x["id"], "g"]) if x["id"] in act_pl.index else 0
            pl.append({"name": x["name"], "side": side, "p": x["props"]["p_score"], "scored": g > 0, "played": bool(played), "starter_pred": x["starter"],
                       "started": bool(act_pl.at[x["id"], "starter"]) if x["id"] in act_pl.index else False})
        team_min = sum(x["exp_min"] for x in cand) or 1
        for x in cand:  # naive reference: the team's goals spread over players by expected minutes only
            y = float(x["id"] in scorers)
            r["skill"]["scorer"].append(((x["props"]["p_score"] - y) ** 2, (1 - math.exp(-d["lambda"][side] * x["exp_min"] / team_min * 0.9) - y) ** 2))
        top3 = sorted(cand, key=lambda x: -x["props"]["p_score"])[:3]
        top3_hits += sum(1 for x in top3 if x["id"] in scorers)
        picks += [{"name": x["name"], "p": x["props"]["p_score"], "scored": x["id"] in scorers} for x in cand if x["props"]["p_score"] >= .35]
    xi_pred = {x["id"] for s in ("home", "away") for x in d["teams"][s]["players"] if x["starter"]}
    xi_act = set(act_pl.index[act_pl.starter])
    named = {x["id"] for s in ("home", "away") for x in d["teams"][s]["players"]}
    r["players"] = {"brier_scorer": mean([(x["p"] - x["scored"]) ** 2 for x in pl]), "scorers": len(scorers),
                    "scorers_covered": sum(1 for sid in scorers if sid in named), "top3_hits": top3_hits, "picks": picks,
                    "xi_overlap": len(xi_pred & xi_act) / max(len(xi_act), 1) if xi_act else None,
                    "scorer_names": [act_pl.at[s, "name"] for s in scorers]}
    return r

rounds = defaultdict(lambda: {"first": [], "final": []})
for f in sorted(glob.glob(f"{LEDGER}/*/R*/*_*.json")):
    season, rnd = f.split("/")[-3], f.split("/")[-2]
    kind = "first" if f.endswith("_first.json") else "final"
    s = score_match(json.load(open(f)), kind)
    if s: rounds[(season, rnd)][kind].append(s)

def summarize(ms):
    if not ms: return None
    n = len(ms); mk = [m for m in ms if "mk_logloss" in m]
    bets = [b for m in ms for b in m.get("bets", [])]
    allbin = defaultdict(list)
    for m in ms:
        for k, v in m["binary"].items(): allbin[k.split(" over ")[0] if " over " in k else k].append(v)
    counts = defaultdict(list)
    for m in ms:
        for k, v in m.get("counts", {}).items(): counts[k].append(v)
    calib = defaultdict(lambda: [0, 0.0, 0.0])
    for m in ms:
        for v in list(m["binary"].values()) + [{"p": m["p"][k], "y": m["res"] == k} for k in "1X2"]:
            b = min(int(v["p"] * 5), 4); c = calib[b]; c[0] += 1; c[1] += v["p"]; c[2] += float(v["y"])
    picks = [p for m in ms for p in m["players"]["picks"]]
    sk = {k: [x for m in ms for x in m["skill"][k]] for k in ("goal", "stat", "scorer")}
    skill = {k: (1 - sum(a for a, _ in v) / sum(b for _, b in v)) if v else None for k, v in sk.items()}
    return {
        "n": n,
        "fav_hit": mean([m["fav_hit"] for m in ms]), "logloss": mean([m["logloss"] for m in ms]), "brier": mean([m["brier"] for m in ms]), "rps": mean([m["rps"] for m in ms]),
        "baseline_logloss": mean([ll(BASE[m["res"]]) for m in ms]),
        "market": {"n": len(mk), "fav_hit": mean([m["mk_fav_hit"] for m in mk]), "logloss": mean([m["mk_logloss"] for m in mk]),
                   "brier": mean([m["mk_brier"] for m in mk]), "rps": mean([m["mk_rps"] for m in mk]),
                   "model_logloss_same": mean([m["logloss"] for m in mk])} if mk else None,
        "top1_hit": mean([m["top1_hit"] for m in ms]), "top3_hit": mean([m["top3_hit"] for m in ms]),
        "binary": {k: {"n": len(v), "hit": mean([x["hit"] for x in v]), "brier": mean([x["brier"] for x in v])} for k, v in allbin.items()},
        "counts": {k: {"n": len(v), "mae": mean([x["abs_err"] for x in v]), "line_hit": mean([(x["p_over"] >= .5) == (x["act"] > x["line"]) for x in v if "line" in x])} for k, v in counts.items()},
        "calibration": [{"bin": f"{b * 20}-{b * 20 + 20}%", "n": c[0], "pred": c[1] / c[0], "obs": c[2] / c[0]} for b, c in sorted(calib.items())],
        "players": {"brier_scorer": mean([m["players"]["brier_scorer"] for m in ms]), "scorers": sum(m["players"]["scorers"] for m in ms),
                    "scorers_covered": sum(m["players"]["scorers_covered"] for m in ms), "top3_hits": sum(m["players"]["top3_hits"] for m in ms),
                    "picks": len(picks), "picks_hit": mean([p["scored"] for p in picks]), "picks_exp": mean([p["p"] for p in picks]),
                    "xi_overlap": mean([m["players"]["xi_overlap"] for m in ms])},
        "value": {"bets": len(bets), "won": sum(b["won"] for b in bets), "pnl": sum(b["pnl"] for b in bets), "roi": sum(b["pnl"] for b in bets) / len(bets) if bets else None},
        "goals": {"pred": sum(m["exp_goals"] for m in ms), "act": sum(m["goals"] for m in ms)},
        "skill": skill,
    }

def components(s):
    """Each component is a skill score vs its proper naive reference, mapped to 0-100 with 50 = no skill.
    Brier skill 0.10 (10% lower error than the reference) = 75; for 1X2 vs bookmaker 0.05 log-loss better = 75."""
    if not s: return None
    c = {"1X2 vs storico": 50 + 250 * (1 - s["logloss"] / s["baseline_logloss"])}
    if s["market"]: c["1X2 vs bookmaker"] = 50 + 500 * (s["market"]["logloss"] - s["market"]["model_logloss_same"])
    names = {"goal": "Mercati gol", "stat": "Linee statistiche", "scorer": "Marcatori"}
    for k, v in (s.get("skill") or {}).items():
        if v is not None: c[names[k]] = 50 + 250 * v
    return {k: round(max(0, min(100, v))) for k, v in c.items()}

def grade(s):
    c = components(s)
    return round(sum(c.values()) / len(c)) if c else None

out = []
for (season, rnd), v in sorted(rounds.items()):
    fin = summarize(v["final"]); first = summarize(v["first"])
    matches = sorted(v["final"], key=lambda m: M.at[m["match_id"], "date"])
    out.append({"season": season, "round": int(rnd[1:]), "complete": len(v["final"]) == int(((M.season == season) & (M["round"] == int(rnd[1:]))).sum()),
                "replay": any(json.load(open(f)).get("replay") for f in glob.glob(f"{LEDGER}/{season}/{rnd}/*_final.json")),
                "grade": grade(fin), "components": components(fin), "final": fin, "first": first, "matches": matches})
allm = [m for r in out for m in r["matches"]]
total = summarize(allm)
json.dump({"rounds": out, "total": total, "total_grade": grade(total), "total_components": components(total), "edge": EDGE, "base": BASE}, open(OUT, "w"), ensure_ascii=False, default=float, indent=0)
for r in out:
    f = r["final"]
    print(r["components"])
    print(f"{r['season']} R{r['round']:02d} {'replay' if r['replay'] else 'live'} n={f['n']} voto={r['grade']} favorito {f['fav_hit']:.0%} logloss {f['logloss']:.3f} (base {f['baseline_logloss']:.3f}"
          f"{', mercato %.3f' % f['market']['logloss'] if f['market'] else ''}) top3 {f['top3_hit']:.0%} value {f['value']['bets']} bet pnl {f['value']['pnl']:+.2f}")
print("TOTALE voto", grade(total), components(total))

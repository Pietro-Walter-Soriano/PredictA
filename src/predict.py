"""Forecast every upcoming Serie A match (next N days) and write one JSON per match + an index for the dashboard."""
import json, math, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA, OUT = sys.argv[1], sys.argv[2]
DAYS_AHEAD = 10
SUB_BOOST = 1.7          # same player scores ~1.7x more per minute off the bench (Serie A 2023-26, 528 players: open late-game)
AS_OF = os.environ.get("AS_OF")          # replay mode: forecast as if run at this UTC time (data after it is hidden)
ONLY_ROUND = os.environ.get("ONLY_ROUND")
GOAL_SHARE_1H = 0.44     # share of goals in the first half (Serie A long-run, re-estimated below)
os.makedirs(OUT, exist_ok=True)
for old in __import__("glob").glob(f"{OUT}/2*.json"): os.remove(old)  # matches already played drop out of the board
cal = json.load(open(f"{DATA}/calibration.json"))
W_MODEL = cal["model_weight"]
bt = json.load(open(f"{DATA}/backtest.json"))

D = pd.read_pickle(f"{DATA}/dataset.pkl")
now = datetime.fromisoformat(AS_OF) if AS_OF else datetime.now(timezone.utc).replace(tzinfo=None)
if AS_OF:
    Mx = D["matches"]; hide = Mx.date >= now
    Mx.loc[hide, "finished"] = False
    Mx.loc[hide, ["hg", "ag", "ht_hg", "ht_ag"]] = np.nan
    gone = set(Mx.loc[hide, "match_id"])
    D["team_match"] = D["team_match"][~D["team_match"].match_id.isin(gone)]
    D["player_match"] = D["player_match"][~D["player_match"].match_id.isin(gone) & (D["player_match"].date < now)]  # cups too
M, T, P, tr = features.build(D)
L = model.lineup_strength(P, M)
R = model.referee_table(M, T)
V = model.team_view(M, T, L, R)
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
fin = V[V.finished]
tm = T.set_index(["match_id", "venue"])
gm = model.GoalModel().fit(fin)
sm = {t: model.StatModel(t).fit(fin, pd.Series([tm[t].get((m, s), np.nan) for m, s in zip(fin.match_id, fin.side)], index=fin.index)) for t in model.STAT_TARGETS}
Mf = M[M.finished]
share_1h = float((Mf.ht_hg + Mf.ht_ag).sum() / (Mf.hg + Mf.ag).sum())
last = P.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
up = M[(~M.finished) & (M.date >= now - timedelta(hours=3)) & (M.date <= now + timedelta(days=DAYS_AHEAD))]
if ONLY_ROUND: up = up[up["round"] == int(ONLY_ROUND)]
season_now = M.season.max()
league_means = T[T.season == season_now][list(model.STAT_TARGETS) + ["poss"]].mean()

def r3(x): return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)

def team_block(team, side, Vrow, lu, ko):
    Tt = T[T.team == team].sort_values("date")
    cur = Tt[Tt.season == season_now]
    st = Tt.iloc[-1] if len(Tt) else None
    style = {k: r3(st[f"cur_{k}"]) if st is not None and f"cur_{k}" in st else None for k in features.STYLE + ["npxg", "opp_npxg"]}
    lg = T[T.season == season_now]
    pct = {k: r3((lg.groupby("team")[k].mean() <= cur[k].mean()).mean()) if len(cur) and k in cur else None for k in features.STYLE + ["npxg", "opp_npxg"]}
    form = [{"date": r.date.strftime("%Y-%m-%d"), "opp": r.opp, "venue": r.venue, "gf": int(r.gf), "ga": int(r.ga),
             "xg": r3(r.npxg), "xga": r3(r.opp_npxg), "poss": r3(r.poss), "ppda": r3(r.ppda)} for r in Tt.tail(10).itertuples()]
    ratings = {"att": r3(tr.att.get(team)), "def": r3(tr.dfn.get(team))}
    table = {"pts": int(((cur.gf > cur.ga) * 3 + (cur.gf == cur.ga)).sum()), "gp": len(cur), "gf": int(cur.gf.sum()), "ga": int(cur.ga.sum()),
             "xg": r3(cur.npxg.sum()), "xga": r3(cur.opp_npxg.sum())}
    subs = lu["subs"]
    if not subs:  # predicted lineups come without a bench: use the squad's recent players who are not starting / unavailable
        out_ids = {s["id"] for s in lu["starters"]} | {u["id"] for u in lu["unavailable"]}
        recent = P[(P.team == team) & (P.date >= now - timedelta(days=60))].sort_values("date").groupby("player_id").tail(1)
        recent = recent[~recent.player_id.isin(out_ids)].sort_values("min_before" if "min_before" in recent else "cur_minutes", ascending=False)
        subs = [{"id": int(x.player_id), "name": x.name, "grp": x.grp, "no": None, "age": None, "mv": x.mv, "estimated": True} for x in recent.head(12).itertuples()]
    players = []
    for s in lu["starters"] + subs:
        starter = any(s is x for x in lu["starters"])
        p = last.loc[s["id"]] if s["id"] in last.index else None
        hist = P[P.player_id == s["id"]].sort_values("date")
        hs = hist[hist.season == season_now]
        g = lambda c: r3(p[c]) if p is not None and c in p and not pd.isna(p[c]) else None
        exp_min = (g("cur_min_start") or 80.0) if starter else 0.0
        players.append({
            "id": int(s["id"]), "name": s["name"], "bench_estimated": s.get("estimated", False), "no": s.get("no"), "grp": s.get("grp"), "pos": s.get("pos"), "x": s.get("x"), "y": s.get("y"),
            "age": s.get("age"), "mv": r3((s.get("mv") or 0) / 1e6), "cap": s.get("cap", False), "starter": starter, "exp_min": round(exp_min),
            "rating": g("cur_rating"), "form5": g("cur_form5"), "npxg90": g("cur_npxg"), "xa90": g("cur_xa"), "sh90": g("cur_sh"), "sot90": g("cur_sot"),
            "kp90": g("cur_kp"), "yc90": g("cur_yc"), "fouls90": g("cur_fouls"), "fouled90": g("cur_fouled"), "def90": g("cur_def_actions"),
            "tackles90": g("cur_tackles"), "int90": g("cur_interceptions"), "rec90": g("cur_recoveries"), "aer90": g("cur_aerials"),
            "drb90": g("cur_dribbles"), "saves90": g("cur_saves"), "xgot_faced90": g("cur_xgot_faced"), "conceded90": g("cur_conceded"),
            "start_rate5": g("cur_start_rate5"), "apps": int(len(hist)), "minutes": int(hist["min"].sum()),
            "cup_apps": int((hs.comp != "serie_a").sum()),
            "season": {"apps": int(len(hs)), "starts": int(hs.starter.sum()), "min": int(hs["min"].sum()), "g": int(hs.g.sum()), "a": int(hs.a.sum()),
                       "npxg": r3(hs.npxg.sum()), "xa": r3(hs.xa.sum()), "yc": int(hs.yc.sum()), "rating": r3(hs.rating.mean())},
            "pens": int(hist.pen_att.sum()), "pens_season": int(hs.pen_att.sum()),
            "rest_days": (ko - hist.date.iloc[-1]).days if len(hist) else None,  # days between his last game (any competition) and this kick-off
            "min_14d": int(hist[hist.date >= now - timedelta(days=14)]["min"].sum()),
            "log": [{"date": r.date.strftime("%Y-%m-%d"), "comp": r.comp, "team": r.team if isinstance(r.team, str) else None, "min": int(r.min), "start": bool(r.starter), "rating": r3(r.rating), "g": int(r.g),
                     "a": int(r.a), "npxg": r3(r.npxg), "xa": r3(r.xa), "sh": None if pd.isna(r.sh) else int(r.sh), "yc": int(r.yc)} for r in hist.tail(10).itertuples()],
            "by_season": [{"season": k[0], "comp": k[1], "apps": int(len(v)), "min": int(v["min"].sum()), "g": int(v.g.sum()), "a": int(v.a.sum()),
                           "npxg": r3(v.npxg.sum()) if v.npxg.notna().any() else None, "xa": r3(v.xa.sum()) if v.xa.notna().any() else None,
                           "rating": r3(v.rating.mean()), "yc": int(v.yc.sum())} for k, v in hist.groupby(["season", "comp"])]})
    # bench minutes fill the gap to 990 using each sub's usual sub-minutes and recent involvement
    gap = max(990 - sum(p["exp_min"] for p in players if p["starter"]), 0)
    subs = [p for p in players if not p["starter"] and p["grp"] != "GK"]
    wts = [(last.at[p["id"], "cur_min_sub"] if p["id"] in last.index and not pd.isna(last.at[p["id"], "cur_min_sub"]) else 10) * (1 - (p["start_rate5"] or 0) * 0.3) for p in subs]
    for p, w in zip(subs, wts): p["exp_min"] = round(gap * w / sum(wts)) if sum(wts) else 0
    return {"name": team, "side": side, "formation": lu.get("formation"), "coach": lu["coach"], "lineup_type": None,
            "ratings": ratings, "style": style, "style_pct": pct, "form": form, "table": table,
            "unavailable": lu["unavailable"], "players": players,
            "lineup_feats": {k: r3(Vrow.get(k)) for k in ("xi_att", "xi_att_rel", "xi_rating", "xi_rating_rel", "xi_form", "xi_mv", "xi_rest", "xi_min14", "xi_games10", "xi_short_rest")}}

def player_props(tb, lam, side_stats):
    pl = [p for p in tb["players"] if p["exp_min"] > 0]
    nz = lambda v, d: d if v is None else v  # a recorded 0 (e.g. a goalkeeper's shots) must stay 0, only missing data falls back
    eff = lambda p: p["exp_min"] / 90 * (1 if p["starter"] else SUB_BOOST)  # effective attacking minutes
    pen_taker = max(pl, key=lambda p: (p["pens_season"], p["pens"]), default=None)
    # a player's share of the team's goals blends his npxG share and his shot share (both per 90 x expected minutes):
    # out of sample on 2025-26 this beats npxG alone for anytime-scorer probabilities
    xg_tot = sum(nz(p["npxg90"], 0.05) * eff(p) for p in pl) or 1
    sh_tot = sum(nz(p["sh90"], 0.8) * eff(p) for p in pl) or 1
    share = lambda p: (nz(p["npxg90"], 0.05) * eff(p) / xg_tot + nz(p["sh90"], 0.8) * eff(p) / sh_tot) / 2
    xa_tot = sum(nz(p["xa90"], 0.03) * p["exp_min"] / 90 for p in pl) or 1
    pen_goals = 0.11 * lam / 1.3 * 0.78  # ~0.11 penalties per team-match scaled with attacking strength, 78% scored
    for p in pl:
        npg = (lam - pen_goals) * share(p)
        eg = npg + (pen_goals if pen_taker and p["id"] == pen_taker["id"] else 0)
        ea = lam * 0.75 * (nz(p["xa90"], 0.03) * p["exp_min"] / 90) / xa_tot
        sh = nz(p["sh90"], 0.8) * p["exp_min"] / 90 * side_stats["shots"] / max(sum(nz(q["sh90"], 0.8) * q["exp_min"] / 90 for q in pl), 1)
        yc = nz(p["yc90"], 0.18) * p["exp_min"] / 90
        p["props"] = {"exp_goals": r3(eg), "p_score": r3(1 - math.exp(-eg)), "p_assist": r3(1 - math.exp(-ea)), "p_ga": r3(1 - math.exp(-eg - ea)),
                      "exp_shots": r3(sh), "p_sot": r3(1 - math.exp(-sh * nz(p["sot90"], 0.3) / max(nz(p["sh90"], 0.8), 0.1))),
                      "p_card": r3(1 - math.exp(-yc)), "exp_fouls": r3(nz(p["fouls90"], 1) * p["exp_min"] / 90),
                      "exp_tackles": r3(nz(p["tackles90"], 0) * p["exp_min"] / 90), "pen_taker": bool(pen_taker and p["id"] == pen_taker["id"])}
        if p["grp"] == "GK":
            p["props"]["exp_saves"] = r3(side_stats["opp_sot"] - side_stats["opp_lam"] * 0.95)
    for p in tb["players"]:
        p.setdefault("props", None)

index = []
for r in up.itertuples():
    vh = V[(V.match_id == r.match_id) & (V.side == "home")]
    va = V[(V.match_id == r.match_id) & (V.side == "away")]
    if vh.empty or va.empty or any(len(r.lineup[s]["starters"]) != 11 for s in ("home", "away")):
        print("skip", r.home, "-", r.away, ": lineup not complete yet", flush=True); continue
    both = pd.concat([vh, va])
    lam_model = gm.predict(both)
    has_mk = not pd.isna(r.p1_close)
    if has_mk:
        pov = r.pover / (r.pover + r.punder) if r.pover and r.punder and not pd.isna(r.pover) else None
        mlh, mla = model.market_lambdas(r.p1_close, r.px_close, r.p2_close, r.ou_line, pov)
        lh, la = lam_model[0] ** W_MODEL * mlh ** (1 - W_MODEL), lam_model[1] ** W_MODEL * mla ** (1 - W_MODEL)
    else:
        mlh = mla = None; lh, la = lam_model
    S = model.dixon_coles(lh, la)
    Smodel = model.dixon_coles(*lam_model)
    i = np.arange(model.MAXG + 1); tot = i[:, None] + i[None, :]; diff = i[:, None] - i[None, :]
    H1 = model.dixon_coles(lh * share_1h, la * share_1h, 0)
    stats = {}
    for t, m_ in sm.items():
        pr = m_.predict(both)
        stats[t] = {"home": r3(pr[0]), "away": r3(pr[1]), "disp": r3(m_.dispersion), "league_avg": r3(league_means[t])}
    stats["poss"] = {"home": r3(50 + (vh.s_poss.iloc[0] - va.s_poss.iloc[0]) / 2), "away": r3(50 - (vh.s_poss.iloc[0] - va.s_poss.iloc[0]) / 2)}
    stats["saves"] = {"home": r3(max(stats["sot"]["away"] - la * 0.95, 0)), "away": r3(max(stats["sot"]["home"] - lh * 0.95, 0))}
    lines = {}
    for name, mean, disp, ls in (("corners", stats["corners"]["home"] + stats["corners"]["away"], stats["corners"]["disp"], (7.5, 8.5, 9.5, 10.5, 11.5)),
                                 ("corners_home", stats["corners"]["home"], stats["corners"]["disp"], (3.5, 4.5, 5.5, 6.5)),
                                 ("corners_away", stats["corners"]["away"], stats["corners"]["disp"], (2.5, 3.5, 4.5, 5.5)),
                                 ("cards", stats["yc"]["home"] + stats["yc"]["away"], stats["yc"]["disp"], (2.5, 3.5, 4.5, 5.5, 6.5)),
                                 ("sot", stats["sot"]["home"] + stats["sot"]["away"], stats["sot"]["disp"], (6.5, 7.5, 8.5, 9.5, 10.5)),
                                 ("shots", stats["shots"]["home"] + stats["shots"]["away"], stats["shots"]["disp"], (20.5, 22.5, 24.5, 26.5, 28.5)),
                                 ("fouls", stats["fouls"]["home"] + stats["fouls"]["away"], stats["fouls"]["disp"], (20.5, 22.5, 24.5, 26.5, 28.5)),
                                 ("offsides", stats["offsides"]["home"] + stats["offsides"]["away"], stats["offsides"]["disp"], (2.5, 3.5, 4.5))):
        lines[name] = {"mean": r3(mean), "disp": r3(disp), "over": {str(l): r3(model.count_over(mean, l, disp)) for l in ls}}
    H = team_block(r.home, "home", vh.iloc[0], {**r.lineup["home"], "formation": r.home_formation}, r.date)
    A = team_block(r.away, "away", va.iloc[0], {**r.lineup["away"], "formation": r.away_formation}, r.date)
    player_props(H, lh, {"shots": stats["shots"]["home"], "opp_sot": stats["sot"]["away"], "opp_lam": la})
    player_props(A, la, {"shots": stats["shots"]["away"], "opp_sot": stats["sot"]["home"], "opp_lam": lh})
    ref = R[R.match_id == r.match_id].iloc[0] if (R.match_id == r.match_id).any() else None
    P_ = lambda mask: r3(S[mask].sum())
    out = {
        "match_id": int(r.match_id), "season": r.season, "date": r.date.isoformat() + "Z", "round": int(r.round), "home": r.home, "away": r.away,
        "generated": now.isoformat(timespec="minutes") + "Z", "replay": bool(AS_OF),
        "lineup_type": r.lineup_type, "referee": {"name": r.referee, "n": int(ref.ref_n) if ref is not None else 0,
                                                  "yc_idx": r3(ref.ref_yc) if ref is not None else None, "fouls_idx": r3(ref.ref_fouls) if ref is not None else None},
        "weather": {"temp": r.temp, "wind": r.wind, "rain": r.rain},
        "lambda": {"home": r3(lh), "away": r3(la)}, "lambda_model": {"home": r3(lam_model[0]), "away": r3(lam_model[1])},
        "lambda_base": {"home": r3(vh.lam_base.iloc[0]), "away": r3(va.lam_base.iloc[0])},
        "lambda_market": {"home": r3(mlh), "away": r3(mla)} if has_mk else None, "model_weight": W_MODEL if has_mk else 1.0,
        "p": {**{k: r3(v) for k, v in model.outcome_probs(S).items()}},
        "p_model": {k: r3(v) for k, v in model.outcome_probs(Smodel).items()}, "p_model_over25": r3(Smodel[tot > 2.5].sum()),
        "p_market": {"1": r3(r.p1_close / (r.p1_close + r.px_close + r.p2_close)), "X": r3(r.px_close / (r.p1_close + r.px_close + r.p2_close)),
                     "2": r3(r.p2_close / (r.p1_close + r.px_close + r.p2_close)), "ou_line": r3(r.ou_line),
                     "over": r3(r.pover / (r.pover + r.punder)) if r.pover and not pd.isna(r.pover) else None,
                     "open": {"1": r3(r.p1_open), "X": r3(r.px_open), "2": r3(r.p2_open)}} if has_mk else None,
        "markets": {
            "ou": {str(l): r3(S[tot > l].sum()) for l in (0.5, 1.5, 2.5, 3.5, 4.5)},
            "btts": P_((i[:, None] > 0) & (i[None, :] > 0)),
            "dc": {"1X": P_(diff >= 0), "12": P_(diff != 0), "X2": P_(diff <= 0)},
            "cs": {"home": r3(S[:, 0].sum()), "away": r3(S[0, :].sum())},
            "wtn": {"home": r3(S[1:, 0].sum()), "away": r3(S[0, 1:].sum())},
            "team_ou": {"home": {str(l): r3(S[i > l, :].sum()) for l in (0.5, 1.5, 2.5)}, "away": {str(l): r3(S[:, i > l].sum()) for l in (0.5, 1.5, 2.5)}},
            "hcp": {"home_-1.5": P_(diff >= 2), "home_-2.5": P_(diff >= 3), "away_-1.5": P_(diff <= -2), "home_+1.5": P_(diff >= -1), "away_+1.5": P_(diff <= 1)},
            "ht": {"1": r3(np.tril(H1, -1).sum()), "X": r3(np.trace(H1)), "2": r3(np.triu(H1, 1).sum()), "o0.5": r3(1 - H1[0, 0]),
                   "o1.5": r3(H1[(i[:, None] + i[None, :]) > 1.5].sum())},
            "goals_dist": [r3(S[tot == n].sum()) for n in range(6)] + [r3(S[tot >= 6].sum())],
            "margin_dist": {str(d): r3(S[diff == d].sum()) for d in range(-3, 4)},
        },
        "top_scores": [{"s": f"{a}-{b}", "p": r3(S[a, b])} for a, b in sorted(((a, b) for a in range(7) for b in range(7)), key=lambda x: -S[x])[:9]],
        "matrix": [[r3(S[a, b]) for b in range(6)] for a in range(6)],
        "stats": stats, "lines": lines, "teams": {"home": H, "away": A},
        "share_1h": r3(share_1h),
    }
    fn = f"{r.date:%Y%m%d}_{r.home}_{r.away}".replace(" ", "-")
    json.dump(out, open(f"{OUT}/{fn}.json", "w"), ensure_ascii=False)
    index.append({"file": fn, "date": out["date"], "round": out["round"], "home": r.home, "away": r.away, "p": out["p"], "lambda": out["lambda"],
                  "lineup_type": r.lineup_type, "o25": out["markets"]["ou"]["2.5"], "btts": out["markets"]["btts"], "top": out["top_scores"][0],
                  "p_market": out["p_market"]})
json.dump({"generated": now.isoformat(timespec="minutes") + "Z", "matches": index, "backtest": bt, "model_weight": W_MODEL},
          open(f"{OUT}/index.json", "w"), ensure_ascii=False, default=float)
print(len(index), "matches forecast")
for x in index: print(x["date"][:16], f'{x["home"]:>12} - {x["away"]:<12}', x["lineup_type"], x["p"], x["lambda"], "mkt", x["p_market"] and {k: x["p_market"][k] for k in "1X2"})

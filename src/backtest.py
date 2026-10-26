"""Walk-forward backtest: refit the learned layers every N rounds on strictly earlier matches, score the next ones.

Reports log-loss / Brier / RPS for 1X2, log-loss for O2.5 and BTTS, MAE for stat counts, calibration buckets,
and compares against the bookmaker closing line where available. Writes calibration params to data/calibration.json.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM = "2021-2022"
REFIT_DAYS = 21

D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
L = model.lineup_strength(P, M)
R = model.referee_table(M, T)
V = model.team_view(M, T, L, R)
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
# pre-match features saved next to the predictions (data/backtest_matches.csv, only with DUMP=1) to study what the bookmaker prices
DUMP = os.environ.get("DUMP") == "1"
DUMP_FEATS = [c for c in V.columns if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_"))] + ["lam_base", "o_lam_base"] if DUMP else []

def stat_y(target):
    return pd.Series([tm[target].get((m, s), np.nan) if target in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index)
Y = {t: stat_y(t) for t in model.STAT_TARGETS}

fin = M[M.finished].copy()
test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
pred = []
for a, b in zip(cuts[:-1], cuts[1:]):
    past = V[(V.date < a) & V.finished]
    nxt = V[(V.date >= a) & (V.date < b) & V.finished]
    if nxt.empty or len(past) < 300: continue
    gm = model.GoalModel().fit(past)
    lam = gm.predict(nxt)
    sm = {t: model.StatModel(t).fit(past, Y[t][past.index]) for t in model.STAT_TARGETS}
    out = nxt[["match_id", "side", "lam_base", "goals"]].copy()
    out["lam"] = lam
    for c in DUMP_FEATS: out["f_" + c] = nxt[c].astype(float).values if c in nxt else np.nan
    for t, m_ in sm.items():
        out["pred_" + t] = m_.predict(nxt); out["y_" + t] = Y[t][nxt.index].values; out["disp_" + t] = m_.dispersion
        out["mean_" + t] = Y[t][past.index].mean()  # naive benchmark known before kick-off
    pred.append(out)
Pr = pd.concat(pred)
W = Pr.pivot_table(index="match_id", columns="side", values=["lam", "lam_base"]).dropna()
W.columns = [f"{a}_{b}" for a, b in W.columns]
W = W.join(M.set_index("match_id")[["season", "date", "home", "away", "hg", "ag", "p1_close", "px_close", "p2_close", "pover", "punder", "ou_line"]])

def probs(lh, la):
    S = model.dixon_coles(lh, la)
    i = np.arange(model.MAXG + 1); tot = i[:, None] + i[None, :]
    return {**model.outcome_probs(S), "o25": S[tot > 2.5].sum(), "btts": S[1:, 1:].sum(),
            "o05": S[tot > 0.5].sum(), "o15": S[tot > 1.5].sum(), "o35": S[tot > 3.5].sum(), "o45": S[tot > 4.5].sum(),
            "cs_h": S[:, 0].sum(), "cs_a": S[0, :].sum()}   # clean sheet of the home / away side

rows = []
for mid, r in W.iterrows():
    pm, pb = probs(r.lam_home, r.lam_away), probs(r.lam_base_home, r.lam_base_away)
    res = "1" if r.hg > r.ag else "X" if r.hg == r.ag else "2"
    rows.append({"match_id": mid, "season": r.season, "res": res, "o25": int(r.hg + r.ag > 2.5), "btts": int(r.hg > 0 and r.ag > 0),
                 **{f"m_{k}": v for k, v in pm.items()}, **{f"b_{k}": v for k, v in pb.items()},
                 "k1": r.p1_close, "kx": r.px_close, "k2": r.p2_close})
S = pd.DataFrame(rows)

def ll(p): return -np.mean(np.log(np.clip(p, 1e-9, 1)))
def score(prefix, df):
    p = np.stack([df[f"{prefix}_1"], df[f"{prefix}_X"], df[f"{prefix}_2"]], 1)
    p = p / p.sum(1, keepdims=True)
    o = np.stack([(df.res == k).astype(float) for k in "1X2"], 1)
    cum_p, cum_o = np.cumsum(p, 1)[:, :2], np.cumsum(o, 1)[:, :2]
    return {"logloss_1x2": ll((p * o).sum(1)), "brier_1x2": np.mean(((p - o) ** 2).sum(1)), "rps": np.mean(((cum_p - cum_o) ** 2).sum(1) / 2),
            "logloss_o25": ll(np.where(df.o25 == 1, df[f"{prefix}_o25"], 1 - df[f"{prefix}_o25"])) if f"{prefix}_o25" in df else None,
            "logloss_btts": ll(np.where(df.btts == 1, df[f"{prefix}_btts"], 1 - df[f"{prefix}_btts"])) if f"{prefix}_btts" in df else None}

if DUMP:   # one row per match: result, model goal rates and probabilities, de-vigged market, home/away pre-match features
    fcols = ["f_" + c for c in DUMP_FEATS]
    ft = Pr.pivot_table(index="match_id", columns="side", values=["lam"] + fcols)
    ft.columns = [f"{a}_{b}" for a, b in ft.columns]
    dump = S.merge(ft, left_on="match_id", right_index=True).merge(
        M[["match_id", "date", "home", "away", "hg", "ag", "p1_close", "px_close", "p2_close", "ou_line", "pover", "punder", "odds_src"]], on="match_id", how="left")
    dump.to_csv(f"{DATA}/backtest_matches.csv", index=False)
    print("dumped", dump.shape, "->", f"{DATA}/backtest_matches.csv", file=sys.stderr)

freq = S.res.value_counts(normalize=True)
S["c_1"], S["c_X"], S["c_2"] = freq["1"], freq["X"], freq["2"]
report = {"n_matches": len(S), "seasons": sorted(S.season.unique()),
          "climatology": score("c", S), "base_elo_xg": score("b", S), "full_model": score("m", S)}
mk = S.dropna(subset=["k1"]).copy()
if len(mk):
    tot = mk[["k1", "kx", "k2"]].sum(1)
    for k, c in (("1", "k1"), ("X", "kx"), ("2", "k2")): mk[f"k_{k}"] = mk[c] / tot
    report["vs_market"] = {"n": len(mk), "market": score("k", mk), "full_model": score("m", mk)}
    # blend in goal-rate space (geometric mean of model and market-implied rates), so every market stays consistent
    W2 = W.loc[mk.match_id]
    ml = [model.market_lambdas(r.p1_close, r.px_close, r.p2_close, r.ou_line, (r.pover / (r.pover + r.punder)) if r.pover and r.punder else None) for r in W2.itertuples()]
    mk["mlh"], mk["mla"] = [x[0] for x in ml], [x[1] for x in ml]
    grid = {}
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        bl = [probs(r.lam_home ** w * a ** (1 - w), r.lam_away ** w * b ** (1 - w)) for r, a, b in zip(W2.itertuples(), mk.mlh, mk.mla)]
        for k in ("1", "X", "2", "o25", "btts"): mk[f"bl_{k}"] = [x[k] for x in bl]
        grid[float(w)] = score("bl", mk)
    best = min(grid, key=lambda w: grid[w]["logloss_1x2"] + grid[w]["logloss_o25"])
    report["vs_market"]["blend_grid"] = grid
    report["vs_market"]["best_model_weight"] = best
    json.dump({"model_weight": best, "n_market_matches": len(mk)}, open(f"{DATA}/calibration.json", "w"))

# calibration of the full model (home-win and over 2.5)
def calib(p, y, bins=(0, .2, .35, .5, .65, .8, 1)):
    c = pd.cut(p, bins)
    return [{"bin": str(k), "n": int(g.size), "pred": round(float(p[g.index].mean()), 3), "obs": round(float(y[g.index].mean()), 3)} for k, g in p.groupby(c, observed=True)]
report["calibration_home_win"] = calib(S.m_1, (S.res == "1").astype(float))
report["calibration_over25"] = calib(S.m_o25, S.o25.astype(float))
for t in model.STAT_TARGETS:
    ok = Pr["y_" + t].notna()
    e = Pr.loc[ok, "pred_" + t] - Pr.loc[ok, "y_" + t]
    base = Pr.loc[ok, "mean_" + t] - Pr.loc[ok, "y_" + t]
    report[f"stat_{t}"] = {"mae": round(float(e.abs().mean()), 3), "mae_league_mean": round(float(base.abs().mean()), 3),
                           "mse": round(float((e ** 2).mean()), 3), "mse_league_mean": round(float((base ** 2).mean()), 3),
                           "dispersion": round(float(Pr["disp_" + t].iloc[-1]), 2)}
# ---- every bettable market, walk-forward: the model against (a) the frequency of the event over all earlier matches and
# (b) the bookmaker, where we have its price (only Over/Under 2.5). Written to data/backtest_markets.json.
from scipy.stats import nbinom, poisson
LINES = {"corners": ("Angoli", (7.5, 8.5, 9.5, 10.5, 11.5)), "yc": ("Gialli", (2.5, 3.5, 4.5, 5.5, 6.5)), "sot": ("Tiri in porta", (6.5, 7.5, 8.5, 9.5, 10.5)),
         "shots": ("Tiri", (20.5, 22.5, 24.5, 26.5, 28.5)), "fouls": ("Falli", (20.5, 22.5, 24.5, 26.5, 28.5)), "offsides": ("Fuorigioco", (2.5, 3.5, 4.5))}
Fin = M[M.finished].sort_values(["date", "match_id"]).set_index("match_id")
hist = pd.DataFrame(index=Fin.index)
tg = Fin.hg + Fin.ag
for nm, ln in (("o05", .5), ("o15", 1.5), ("o25", 2.5), ("o35", 3.5), ("o45", 4.5)): hist[nm] = (tg > ln).astype(float)
hist["btts"] = ((Fin.hg > 0) & (Fin.ag > 0)).astype(float); hist["cs_h"] = (Fin.ag == 0).astype(float); hist["cs_a"] = (Fin.hg == 0).astype(float)
totals = T.groupby("match_id")[[t for t in LINES if t in T]].sum(min_count=1)
for t, (_, lns) in LINES.items():
    if t not in totals: continue
    for ln in lns: hist[f"{t}_{ln}"] = (totals[t].reindex(hist.index) > ln).astype(float).where(totals[t].reindex(hist.index).notna())
prior = hist.expanding().mean().shift(1)            # frequency over matches played earlier (strictly before this one in the date order)
prior_n = hist.expanding().count().shift(1)

def over_vec(mean, line, disp):
    k = np.floor(line); out = poisson.sf(k, mean)
    nb = disp > 1.0001
    if nb.any():
        n = mean[nb] / (disp[nb] - 1); out[nb] = nbinom.sf(k, n, n / (n + mean[nb]))
    return out

def market_row(name, p, y, base, season, extra=None):
    ok = ~(np.isnan(p) | np.isnan(y) | np.isnan(base))
    p, y, base, season = np.clip(p[ok], 1e-4, 1 - 1e-4), y[ok], np.clip(base[ok], 1e-4, 1 - 1e-4), season[ok]
    if len(p) < 60: return None
    lm = -(y * np.log(p) + (1 - y) * np.log(1 - p)); lb = -(y * np.log(base) + (1 - y) * np.log(1 - base))
    d = lm - lb
    r = {"market": name, "n": int(len(p)), "rate": round(float(y.mean()), 3), "model_p": round(float(p.mean()), 3),
         "logloss_model": round(float(lm.mean()), 4), "logloss_freq": round(float(lb.mean()), 4),
         "brier_skill": round(1 - float(((p - y) ** 2).mean() / ((base - y) ** 2).mean()), 4),
         "diff": round(float(d.mean()), 4), "z": round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2),   # negative = the model beats the frequency
         "hit": round(float(((p >= .5) == (y == 1)).mean()), 3), "hit_freq": round(float(((base >= .5) == (y == 1)).mean()), 3), "by_season": {}}
    for s in sorted(set(season)):
        m_ = season == s
        if m_.sum() >= 30: r["by_season"][s] = {"n": int(m_.sum()), "diff": round(float(d[m_].mean()), 4), "brier_skill": round(1 - float(((p[m_] - y[m_]) ** 2).mean() / ((base[m_] - y[m_]) ** 2).mean()), 4)}
    if extra: r.update(extra)
    return r

ids = W.index; seas = W.season.values; mk_rows = []
# 1) goals markets: model probabilities from the Poisson/Dixon-Coles rates, outcome from the final score
Sg = S.set_index("match_id").reindex(ids)
tot_g = (W.hg + W.ag).values
EV = {"o05": ("Over 0.5", tot_g > .5), "o15": ("Over 1.5", tot_g > 1.5), "o25": ("Over 2.5", tot_g > 2.5), "o35": ("Over 3.5", tot_g > 3.5), "o45": ("Over 4.5", tot_g > 4.5),
      "btts": ("Gol entrambe", (W.hg.values > 0) & (W.ag.values > 0)), "cs_h": ("Porta inviolata casa", W.ag.values == 0), "cs_a": ("Porta inviolata ospite", W.hg.values == 0)}
for key, (name, yv) in EV.items():
    p = Sg[f"m_{key}"].values.astype(float); base = prior[key].reindex(ids).values; npr = prior_n[key].reindex(ids).values
    base = np.where(npr >= 100, base, np.nan)
    extra = None
    if key == "o25":   # the only bettable market with a bookmaker price in our data
        qo = (W.pover / (W.pover + W.punder)).where(W.ou_line.eq(2.5)).values.astype(float)
        b = ~np.isnan(qo) & ~np.isnan(p)
        if b.sum() >= 60:
            yb = yv[b].astype(float); pb = np.clip(p[b], 1e-4, 1 - 1e-4); qb = np.clip(qo[b], 1e-4, 1 - 1e-4)
            lmb = -(yb * np.log(pb) + (1 - yb) * np.log(1 - pb)); lkb = -(yb * np.log(qb) + (1 - yb) * np.log(1 - qb))
            extra = {"bookmaker": {"n": int(b.sum()), "logloss_model": round(float(lmb.mean()), 4), "logloss_book": round(float(lkb.mean()), 4),
                                   "diff_vs_book": round(float((lmb - lkb).mean()), 4), "z_vs_book": round(float((lmb - lkb).mean() / ((lmb - lkb).std(ddof=1) / np.sqrt(b.sum()))), 2)}}
    r = market_row(name, p, yv.astype(float), base, seas, extra)
    if r: mk_rows.append(r)
# 2) statistic totals (both teams): count model with its dispersion, lines as published in the dashboard
for t, (label, lns) in LINES.items():
    if f"pred_{t}" not in Pr: continue
    g_ = Pr.pivot_table(index="match_id", columns="side", values=[f"pred_{t}", f"y_{t}", f"disp_{t}"]).reindex(ids)
    mean = (g_[f"pred_{t}"]["home"] + g_[f"pred_{t}"]["away"]).values.astype(float); act = (g_[f"y_{t}"]["home"] + g_[f"y_{t}"]["away"]).values.astype(float)
    disp = g_[f"disp_{t}"]["home"].values.astype(float)
    for ln in lns:
        key = f"{t}_{ln}"
        if key not in prior: continue
        p = over_vec(mean, ln, np.where(np.isnan(disp), 1.0, disp)); y = (act > ln).astype(float); y[np.isnan(act)] = np.nan
        base = np.where(prior_n[key].reindex(ids).values >= 100, prior[key].reindex(ids).values, np.nan)
        r = market_row(f"{label} over {ln}", p, y, base, seas)
        if r: mk_rows.append(r)
json.dump({"seasons": sorted(set(seas)), "n_matches": int(len(ids)), "markets": mk_rows}, open(f"{DATA}/backtest_markets.json", "w"), indent=1, default=float)
print("backtest_markets:", len(mk_rows), "markets", file=sys.stderr)

json.dump(report, open(f"{DATA}/backtest.json", "w"), indent=1, default=float)
print(json.dumps(report, indent=1, default=lambda x: round(float(x), 4)))

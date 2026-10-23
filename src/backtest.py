"""Walk-forward backtest: refit the learned layers every N rounds on strictly earlier matches, score the next ones.

Reports log-loss / Brier / RPS for 1X2, log-loss for O2.5 and BTTS, MAE for stat counts, calibration buckets,
and compares against the bookmaker closing line where available. Writes calibration params to data/calibration.json.
"""
import json, sys
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
    return {**model.outcome_probs(S), "o25": S[tot > 2.5].sum(), "btts": S[1:, 1:].sum()}

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
json.dump(report, open(f"{DATA}/backtest.json", "w"), indent=1, default=float)
print(json.dumps(report, indent=1, default=lambda x: round(float(x), 4)))

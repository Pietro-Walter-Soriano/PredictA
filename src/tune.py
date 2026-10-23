"""Tune rating / goal-model hyper-parameters on past seasons only (2021-22 .. 2025-26); the current season stays out-of-sample.

Objective: mean of 1X2 log-loss and O2.5 log-loss of the walk-forward goal model. Writes data/params.json.
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model, seasons

DATA = sys.argv[1]
TUNE_SEASONS = tuple(f"{y}-{y + 1}" for y in seasons.all_starts()[1:-1])   # every past season except the first (warm-up); current stays out
REFIT_DAYS = 42
D = pd.read_pickle(f"{DATA}/dataset.pkl")
i = np.arange(model.MAXG + 1); TOT = i[:, None] + i[None, :]

def evaluate(p):
    features.SEASON_REGRESS = p["season_regress"]
    M, T, P, _ = features.build(D, half_life=p["half_life"], w_goals=p["w_goals"], w_xg=p["w_xg"], w_xgot=1 - p["w_goals"] - p["w_xg"],
                                home_adv=p["home_adv"])
    L = model.lineup_strength(P, M); R = model.referee_table(M, T); V = model.team_view(M, T, L, R)
    test = M[M.finished & M.season.isin(TUNE_SEASONS)]
    cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
    ll1, ll2 = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past, nxt = V[(V.date < a) & V.finished], V[(V.date >= a) & (V.date < b) & V.finished & V.season.isin(TUNE_SEASONS)]
        if nxt.empty or len(past) < 300: continue
        nxt = nxt.assign(lam=model.GoalModel(alpha=p["alpha"]).fit(past).predict(nxt))
        w = nxt.pivot_table(index="match_id", columns="side", values=["lam", "goals"]).dropna()
        for lh, la, hg, ag in zip(w[("lam", "home")], w[("lam", "away")], w[("goals", "home")], w[("goals", "away")]):
            S = model.dixon_coles(lh * p["scale"], la * p["scale"], p["rho"])
            o = model.outcome_probs(S); res = "1" if hg > ag else "X" if hg == ag else "2"
            po = S[TOT > 2.5].sum()
            ll1.append(-np.log(o[res])); ll2.append(-np.log(po if hg + ag > 2.5 else 1 - po))
    return float(np.mean(ll1)), float(np.mean(ll2)), len(ll1)

base = {"half_life": 14, "w_goals": 0.3, "w_xg": 0.5, "home_adv": 0.10, "season_regress": 0.70, "alpha": 2.0, "rho": -0.06, "scale": 1.0}
grid = {"half_life": [8, 14, 22, 32], "w_goals": [0.15, 0.3, 0.45], "w_xg": [0.4, 0.5, 0.65], "home_adv": [0.06, 0.10, 0.14],
        "season_regress": [0.55, 0.7, 0.85], "alpha": [0.5, 2.0, 8.0], "rho": [-0.12, -0.06, 0.0], "scale": [0.97, 1.0, 1.03]}
cur = dict(base); best = evaluate(cur); log = [{"params": dict(cur), "ll_1x2": best[0], "ll_o25": best[1], "n": best[2]}]
print("start", cur, best, flush=True)
for sweep in range(2):  # coordinate descent: one parameter at a time, two passes
    for k, vals in grid.items():
        for v in vals:
            if v == cur[k]: continue
            cand = {**cur, k: v}
            if cand["w_goals"] + cand["w_xg"] > 0.95: continue
            r = evaluate(cand); log.append({"params": cand, "ll_1x2": r[0], "ll_o25": r[1], "n": r[2]})
            if (r[0] + r[1]) / 2 < (best[0] + best[1]) / 2 - 1e-5:
                cur, best = cand, r
                print("better", k, v, [round(x, 4) for x in r[:2]], flush=True)
json.dump({"params": cur, "ll_1x2": best[0], "ll_o25": best[1], "n": best[2], "start": log[0], "seasons": TUNE_SEASONS},
          open(f"{DATA}/params.json", "w"), indent=1)
print("best", cur, best)

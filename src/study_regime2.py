"""Regime factor with a gate: follow the league level only when it has clearly moved.

  python src/study_regime2.py data        (needs data/dataset.pkl; REGIME_CACHE=<file> keeps the walk-forward predictions)

src/study_regime.py showed that a factor "actual / predicted over the recent matches of the whole league" gains 0.7% (goals) to 5.6%
(shots on target) in 2026-27, when the league level jumped, but costs 0.1-1.3% in the five normal seasons (it chases noise).
Here the factor is applied only when the recent scarto is statistically clear:  z = (sum actual - sum predicted) / sqrt(variance)  with
the exponentially weighted sums and the Poisson variance (times the dispersion); otherwise it is 1. Grid: half-life HL, shrinkage K, gate Z0."""
import os, pickle, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM, REFIT_DAYS = "2021-2022", 42
TARGETS = ["goals", "sot", "shots", "offsides", "fouls", "yc"]
CACHE = os.environ.get("REGIME_CACHE")

D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
V = model.team_view(M, T, model.lineup_strength(P, M), model.referee_table(M, T))
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
Y = {"goals": V.goals}
for t in model.STAT_TARGETS: Y[t] = pd.Series([tm[t].get((m, s), np.nan) if t in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index)

if CACHE and os.path.exists(CACHE):
    frames = pickle.load(open(CACHE, "rb"))
else:
    fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
    cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
    parts = {t: [] for t in TARGETS}
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300: continue
        parts["goals"].append(pd.DataFrame({"p": model.GoalModel().fit(past).predict(nxt), "disp": 1.0}, index=nxt.index))
        for t in TARGETS[1:]:
            if Y[t][nxt.index].notna().sum() == 0: continue
            m = model.StatModel(t).fit(past, Y[t][past.index])
            parts[t].append(pd.DataFrame({"p": m.predict(nxt), "disp": m.dispersion}, index=nxt.index))
        print("fit", a.date(), flush=True)
    frames = {}
    for t in TARGETS:
        R = pd.concat(parts[t]).join(V[["match_id", "date", "season", "side"]]).join(Y[t].rename("y")).dropna(subset=["y"])
        frames[t] = R.sort_values(["date", "match_id", "side"]).reset_index(drop=True)
    if CACHE: pickle.dump(frames, open(CACHE, "wb"))

def dev(y, p): return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0) - (y - p))

def adjust(R, hl, k, z0):
    dec, pbar, disp = 0.5 ** (1 / hl), R.p.mean(), float(R.disp.mean())
    out, y, p = R.p.values.copy(), R.y.values, R.p.values
    sy = sp = vp = 0.0; nz = 0
    for idx in R.groupby("match_id", sort=False).indices.values():
        z = (sy - sp) / np.sqrt(max(vp * disp, 1e-9)) if vp > 0 else 0.0
        if z0 is None or abs(z) > z0: out[idx] = p[idx] * (sy + k * pbar) / (sp + k * pbar); nz += len(idx)
        for i in idx: sy = sy * dec + y[i]; sp = sp * dec + p[i]; vp = vp * dec * dec + p[i]
    return out, nz / len(R)

for t in TARGETS:
    R = frames[t]; base = dev(R.y.values, R.p.values); cur = (R.season == "2026-2027").values
    print(f"\n=== {t}: 2026-27 previsto {R.p[cur].mean():.3f} contro reale {R.y[cur].mean():.3f} ({(R.y[cur].mean() / R.p[cur].mean() - 1) * 100:+.1f}%)", flush=True)
    print(f"  {'HL':>5s} {'K':>4s} {'Z0':>4s} {'attivo':>7s} {'tutte':>7s} {'2026-27':>8s} {'prime 5 stagioni':>17s}   per stagione")
    for hl, k, z0 in [(40, 100, None), (100, 100, None)] + [(hl, 30, z) for hl in (100, 250) for z in (2.0, 3.0, 4.0)]:
        pa, share = adjust(R, hl, k, z0); d1 = dev(R.y.values, pa)
        per = pd.DataFrame({"s": R.season, "b": base, "a": d1}).groupby("s").apply(lambda x: (1 - x.a.mean() / x.b.mean()) * 100)
        old = (1 - d1[~cur].mean() / base[~cur].mean()) * 100; now = (1 - d1[cur].mean() / base[cur].mean()) * 100
        print(f"  {hl:5d} {k:4d} {('-' if z0 is None else z0):>4} {share:6.0%} {(1 - d1.mean() / base.mean()) * 100:+6.2f}% {now:+7.2f}% {old:+16.2f}%   " + " ".join(f"{v:+.2f}" for v in per.values), flush=True)
print("FINE")

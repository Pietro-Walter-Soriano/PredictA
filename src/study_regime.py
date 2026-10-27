"""The league level moves from one season to the next (goals per match 3.06 -> 2.43 over five years, 3.01 so far in 2026-27 after the
new time-wasting rules). The ratings adapt slowly and the count models not at all. Does a correction that follows the league level help?

  python src/study_regime.py data     (needs data/dataset.pkl)

1. Walk-forward predictions of the production models (goals and the six counts), refit every REFIT_DAYS on strictly earlier matches.
2. Post-hoc regime factor: for every match, ratio = sum(actual) / sum(predicted) over the earlier matches of the whole league, exponentially
   weighted (half-life HL team-rows) and shrunk towards 1 with K pseudo-rows. The prediction is multiplied by the ratio.
3. Scored on the same matches for every (HL, K): Poisson deviance of the counts, and for goals / shots on target the log-loss of the
   markets built on them. Season by season, and 2026-27 alone."""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM, REFIT_DAYS = "2021-2022", 42
HLS, KS = (40, 100, 250, 600), (0, 30, 100)
LINES = {"sot": (6.5, 7.5, 8.5, 9.5, 10.5), "shots": (20.5, 22.5, 24.5, 26.5, 28.5), "corners": (7.5, 8.5, 9.5, 10.5, 11.5),
         "yc": (2.5, 3.5, 4.5, 5.5), "fouls": (20.5, 22.5, 24.5, 26.5, 28.5), "offsides": (2.5, 3.5, 4.5)}

D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
V = model.team_view(M, T, model.lineup_strength(P, M), model.referee_table(M, T))
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
Y = {"goals": V.goals}
for t in model.STAT_TARGETS: Y[t] = pd.Series([tm[t].get((m, s), np.nan) if t in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index)

fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
parts = {t: [] for t in Y}
for a, b in zip(cuts[:-1], cuts[1:]):
    past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
    if nxt.empty or len(past) < 300: continue
    parts["goals"].append(pd.DataFrame({"p": model.GoalModel().fit(past).predict(nxt), "disp": 1.0}, index=nxt.index))
    for t in model.STAT_TARGETS:
        if Y[t][nxt.index].notna().sum() == 0: continue
        m = model.StatModel(t).fit(past, Y[t][past.index])
        parts[t].append(pd.DataFrame({"p": m.predict(nxt), "disp": m.dispersion}, index=nxt.index))
    print("fit", a.date(), flush=True)

def frame(t):
    R = pd.concat(parts[t]).join(V[["match_id", "date", "season", "side"]]).join(Y[t].rename("y")).dropna(subset=["y"])
    return R.sort_values(["date", "match_id", "side"]).reset_index(drop=True)

def adjust(R, hl, k):
    if hl is None: return R.p.values.copy()
    dec, pbar = 0.5 ** (1 / hl), R.p.mean()
    out = R.p.values.copy(); y, p = R.y.values, R.p.values; sy = sp = 0.0
    for idx in R.groupby("match_id", sort=False).indices.values():
        out[idx] = p[idx] * (sy + k * pbar) / (sp + k * pbar)
        for i in idx: sy = sy * dec + y[i]; sp = sp * dec + p[i]
    return out

def dev(y, p): return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0) - (y - p))

def over(mean, line, disp):
    k = np.floor(line)
    if disp <= 1.0001: return poisson.sf(k, mean)
    n = mean / (disp - 1); return nbinom.sf(k, n, n / (n + mean))

def market_ll(R, padj, t):
    """Log-loss of the markets built on the adjusted means: goals -> 1X2 / Over 1.5 / 2.5 / 3.5 / both score; counts -> the lines on the total."""
    R = R.assign(q=padj); g = R.pivot_table(index="match_id", columns="side", values=["q", "y", "disp"])
    g = g.dropna()
    if t == "goals":
        i = np.arange(model.MAXG + 1); tot = i[:, None] + i[None, :]; ll = {"1X2": [], "O1.5": [], "O2.5": [], "O3.5": [], "BTTS": []}
        for lh, la, hg, ag in zip(g[("q", "home")], g[("q", "away")], g[("y", "home")], g[("y", "away")]):
            S = model.dixon_coles(lh, la); pr = model.outcome_probs(S); res = "1" if hg > ag else "X" if hg == ag else "2"
            ll["1X2"].append(-np.log(max(pr[res], 1e-9)))
            for key, pp, yy in (("O1.5", S[tot > 1.5].sum(), hg + ag > 1.5), ("O2.5", S[tot > 2.5].sum(), hg + ag > 2.5), ("O3.5", S[tot > 3.5].sum(), hg + ag > 3.5), ("BTTS", S[1:, 1:].sum(), hg > 0 and ag > 0)):
                pp = min(max(pp, 1e-4), 1 - 1e-4); ll[key].append(-np.log(pp if yy else 1 - pp))
        return {k: float(np.mean(v)) for k, v in ll.items()}
    mean = (g[("q", "home")] + g[("q", "away")]).values; act = (g[("y", "home")] + g[("y", "away")]).values; d = g[("disp", "home")].values
    out = []
    for ln in LINES[t]:
        pr = np.clip(np.array([over(m_, ln, d_) for m_, d_ in zip(mean, d)]), 1e-4, 1 - 1e-4); yy = (act > ln).astype(float)
        out.append(float(-(yy * np.log(pr) + (1 - yy) * np.log(1 - pr)).mean()))
    return {"linee": float(np.mean(out))}

for t in ["goals", "sot", "shots", "yc", "fouls", "corners", "offsides"]:
    R = frame(t); base = dev(R.y.values, R.p.values)
    cur = R.season == "2026-2027"
    print(f"\n=== {t}: {len(R)} righe | 2026-27: previsto {R.p[cur].mean():.3f} contro reale {R.y[cur].mean():.3f} ({(R.y[cur].mean() / R.p[cur].mean() - 1) * 100:+.1f}%)", flush=True)
    mb = market_ll(R, R.p.values, t) if t in ("goals", "sot") else None
    print(f"  {'HL':>5s} {'K':>4s} {'devianza':>9s} {'vs base':>8s} {'2026-27':>8s} {'stag. migliori':>15s}" + ("   mercati (log-loss, base -> corretto)" if mb else ""))
    best = None
    for hl in HLS:
        for k in KS:
            pa = adjust(R, hl, k); d1 = dev(R.y.values, pa)
            gain = (1 - d1.mean() / base.mean()) * 100; gcur = (1 - d1[cur].mean() / base[cur].mean()) * 100
            per = pd.DataFrame({"s": R.season, "b": base, "a": d1}).groupby("s").apply(lambda x: (1 - x.a.mean() / x.b.mean()) * 100)
            line = f"  {hl:5d} {k:4d} {d1.mean():9.4f} {gain:+7.2f}% {gcur:+7.2f}% {int((per > 0).sum()):>6d}/{len(per)}"
            if mb:
                ma = market_ll(R, pa, t); line += "   " + " ".join(f"{k_} {mb[k_]:.4f}->{ma[k_]:.4f}" for k_ in mb)
            print(line, flush=True)
            if best is None or gain > best[0]: best = (gain, hl, k, per, pa)
    gain, hl, k, per, pa = best
    print(f"  migliore: HL {hl}, K {k} | per stagione: " + " ".join(f"{s} {v:+.2f}" for s, v in per.items()) + f" | 2026-27 previsto dopo la correzione {pa[cur.values].mean():.3f}")
print("FINE")

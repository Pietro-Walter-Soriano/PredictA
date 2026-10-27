"""Can the per-team count models (corners, shots, shots on target, cards, fouls, offsides) be improved?

  python src/study_stats.py data     (needs data/dataset.pkl)

Same walk-forward as backtest.py (refit every REFIT_DAYS on strictly earlier matches), several variants of the model per target,
all scored on the same matches: mean squared error of the per-team count and log-loss of the over/under lines on the match total."""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM, REFIT_DAYS = "2021-2022", 42
LINES = {"corners": (7.5, 8.5, 9.5, 10.5, 11.5), "yc": (2.5, 3.5, 4.5, 5.5), "sot": (6.5, 7.5, 8.5, 9.5, 10.5),
         "shots": (20.5, 22.5, 24.5, 26.5, 28.5), "fouls": (20.5, 22.5, 24.5, 26.5, 28.5), "offsides": (2.5, 3.5, 4.5)}

D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
V = model.team_view(M, T, model.lineup_strength(P, M), model.referee_table(M, T))
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
Y = {t: pd.Series([tm[t].get((m, s), np.nan) if t in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index) for t in model.STAT_TARGETS}
ALL = [c for c in V.columns if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_"))] + ["is_home", "lam_base", "o_lam_base"]
ALL = [c for c in ALL if V[c].dtype.kind in "fiu" and V[c].notna().mean() > .5]

class Lin:       # Poisson GLM on a feature list (the production model is Lin with the STAT_TARGETS list and alpha 1)
    def __init__(self, feats, alpha=1.0): self.f, self.a = feats, alpha
    def fit(self, V, y):
        ok = y.notna(); V, y = V[ok], y[ok]
        self.mu, self.sd = V[self.f].mean(), V[self.f].std().replace(0, 1)
        X = ((V[self.f].astype(float) - self.mu) / self.sd).fillna(0).clip(-4, 4)
        self.m = PoissonRegressor(alpha=self.a / len(V) * 100, max_iter=800).fit(X, y)
        self.disp = max(np.mean((y - self.m.predict(X)) ** 2) / np.mean(self.m.predict(X)), 1.0); return self
    def predict(self, V): return self.m.predict(((V[self.f].astype(float) - self.mu) / self.sd).fillna(0).clip(-4, 4))

class GBM:       # gradient boosting with a Poisson loss, small and heavily regularised
    def __init__(self, feats): self.f = feats
    def fit(self, V, y):
        ok = y.notna(); V, y = V[ok], y[ok]
        self.m = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.04, max_iter=180, min_samples_leaf=40, l2_regularization=5.0, random_state=0).fit(V[self.f].astype(float), y)
        p = self.m.predict(V[self.f].astype(float)); self.disp = max(np.mean((y - p) ** 2) / np.mean(p), 1.0); return self
    def predict(self, V): return self.m.predict(V[self.f].astype(float))

class Blend:     # average of two models' means
    def __init__(self, a, b): self.a, self.b = a, b
    def fit(self, V, y): self.a.fit(V, y); self.b.fit(V, y); self.disp = (self.a.disp + self.b.disp) / 2; return self
    def predict(self, V): return (self.a.predict(V) + self.b.predict(V)) / 2

def variants(t):
    base = model.STAT_TARGETS[t] + ["is_home", "lam_base", "o_lam_base"]
    own = [f"s_{c}" for c in ("corners", "shots", "sot", "yc", "fouls", "offsides")]
    return {"attuale": lambda: Lin(base, 1.0),
            "tutte le feature (alpha 5)": lambda: Lin(ALL, 5.0),
            "tutte le feature (alpha 20)": lambda: Lin(ALL, 20.0),
            "boosting": lambda: GBM(ALL),
            "media attuale+boosting": lambda: Blend(Lin(base, 1.0), GBM(ALL))}

def over(mean, line, disp):
    k = np.floor(line)
    if disp <= 1.0001: return poisson.sf(k, mean)
    n = mean / (disp - 1); return nbinom.sf(k, n, n / (n + mean))

fin = V[V.finished]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
res = {}
for t in model.STAT_TARGETS:
    preds = {n: [] for n in variants(t)}; rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished]
        if nxt.empty or len(past) < 300: continue
        yt = Y[t][nxt.index]
        if yt.notna().sum() == 0: continue
        rows.append(pd.DataFrame({"match_id": nxt.match_id.values, "side": nxt.side.values, "season": nxt.season.values, "y": yt.values, "ref": Y[t][past.index].mean()}))
        for n, mk in variants(t).items():
            m = mk().fit(past, Y[t][past.index]); preds[n].append(pd.DataFrame({"p": m.predict(nxt), "d": m.disp}))
    R = pd.concat(rows, ignore_index=True)
    for n in preds: R["p_" + n] = pd.concat(preds[n], ignore_index=True).p.values; R["d_" + n] = pd.concat(preds[n], ignore_index=True).d.values
    R = R[R.y.notna()]
    out = {}
    for n in preds:
        mse = float(((R["p_" + n] - R.y) ** 2).mean())
        # lines on the match total
        g = R.groupby("match_id").agg(p=("p_" + n, "sum"), d=("d_" + n, "mean"), y=("y", "sum"), c=("y", "size")); g = g[g.c == 2]
        lls = []
        for ln in LINES[t]:
            pr = np.clip(np.array([over(m_, ln, d_) for m_, d_ in zip(g.p, g.d)]), 1e-4, 1 - 1e-4); yy = (g.y > ln).astype(float).values
            lls.append(float(-(yy * np.log(pr) + (1 - yy) * np.log(1 - pr)).mean()))
        out[n] = (mse, float(np.mean(lls)))
    res[t] = (out, len(R), R)
    base_mse = float(((R.ref - R.y) ** 2).mean())
    print(f"\n{t}  ({len(R)} righe, MSE della media di lega {base_mse:.3f})")
    for n, (mse, ll) in out.items():
        print(f"  {n:30s} MSE {mse:.3f} ({(1 - mse / base_mse) * 100:+5.1f}% vs media)   log-loss linee {ll:.4f}")

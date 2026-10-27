"""Where does the GOAL model (expected goals per team -> 1X2, Over/Under, both teams score, clean sheets) still go wrong?

  python src/study_goals.py data     (needs data/dataset.pkl)

1. Walk-forward predictions of the production GoalModel (refit every REFIT_DAYS on strictly earlier matches).
2. Pearson residual r = (goals - lam) / sqrt(lam), correlated with the same candidate features as src/study_resid.py plus the
   persistence of the error (what the model got wrong so far for this team's attack / this opponent's defence).
3. The candidates that pass a Bonferroni threshold with the same sign in most seasons are ADDED to the goal model and the
   variant is scored on the same matches: Poisson deviance of the goals, log-loss of 1X2, Over 2.5 and both-teams-score."""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(HERE, "study_resid.py"), encoding="utf-8").read()
exec(src[:src.index("fin = V[V.finished]")])        # data, V (team view), X (candidate features), z(), model, features, DATA, TEST_FROM, REFIT_DAYS

fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")

def walk(make):
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300: continue
        parts.append(pd.DataFrame({"lam": make().fit(past).predict(nxt)}, index=nxt.index))
    return pd.concat(parts)

# ---------- 1-2. residuals of the production model and scan ----------
R = walk(model.GoalModel).join(V[["goals", "season", "date", "match_id", "side", "team", "opp"]])
R["r"] = (R.goals - R.lam) / np.sqrt(R.lam)
print(f"goal model: {len(R)} team-match, mean residual {R.r.mean():+.3f}, sd {R.r.std():.3f}", flush=True)
R = R.sort_values(["date", "match_id", "side"])
for who, key in (("att", "team"), ("def", "opp")):
    for hl in (6, 15):
        e = R.groupby(key)["r"].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3).mean())
        n = R.groupby(key)["r"].transform(lambda s: s.shift(1).expanding().count())
        R[f"persist:{who}_hl{hl}"] = e * n / (n + 4)
cand = pd.concat([R[[c for c in R.columns if c.startswith("persist:")]], X.loc[R.index]], axis=1)
rows = []
for c in cand.columns:
    s = pd.concat([cand[c], R.r, R.season], axis=1, keys=["x", "r", "s"]).dropna()
    if len(s) < 300 or s.x.std() == 0: continue
    zz = z(s.x); rr = np.corrcoef(zz, s.r)[0, 1]
    by = [np.sign(np.corrcoef(zz[s.s == k], s.r[s.s == k])[0, 1]) for k in sorted(s.s.unique()) if (s.s == k).sum() > 100]
    rows.append((c, len(s), rr, abs(rr) * np.sqrt(len(s)), max(sum(1 for v in by if v > 0), sum(1 for v in by if v < 0)), len(by)))
S = pd.DataFrame(rows, columns=["feature", "n", "corr", "z", "seasons_same_sign", "seasons"]).sort_values("z", ascending=False)
thr = float(norm.isf(0.05 / 2 / len(S)))
S.to_csv(f"{DATA}/study_goals.csv", index=False)
print(f"\ntest {len(S)}, soglia Bonferroni |z|>{thr:.1f}; attesi per caso con |z|>3: ~{len(S) * 0.0027:.1f}")
print(S.head(15).assign(corr=S["corr"].round(3), z=S.z.round(1)).to_string(index=False), flush=True)
pick = S[(S.z > thr) & (S.seasons_same_sign >= S.seasons - 1) & ~S.feature.str.startswith("persist:")].feature.tolist()[:8]
print("\ncandidati aggiunti al modello:", pick, flush=True)
if not pick: print("FINE"); sys.exit()

# ---------- 3. the same walk-forward with the candidates added ----------
for c in pick: V["cand:" + c] = X[c]
add = ["cand:" + c for c in pick]

class GM(model.GoalModel):
    def __init__(self, extra, alpha=2.0):
        super().__init__(alpha); self.f = model.GOAL_FEATS + extra
    def _X(self, V): return ((V[self.f].astype(float) - self.mu) / self.sd).fillna(0.0).clip(-4, 4)
    def fit(self, V):
        V = V[V.finished & V.goals.notna() & V.lam_base.notna()]
        self.mu, self.sd = V[self.f].mean(), V[self.f].std().replace(0, 1)
        self.m = model.PoissonRegressor(alpha=self.alpha / len(V) * 100, max_iter=500).fit(self._X(V), V.goals.values / V.lam_base.values, sample_weight=V.lam_base.values)
        return self

def evaluate(P):
    P = P.join(V[["goals", "season", "match_id", "side"]])
    dev = 2 * (P.goals * np.log(np.where(P.goals > 0, P.goals, 1) / P.lam) - (P.goals - P.lam))
    g = P.pivot_table(index="match_id", columns="side", values=["lam", "goals"]); i = np.arange(model.MAXG + 1); tot = i[:, None] + i[None, :]
    out = {"1": [], "o25": [], "btts": [], "o15": []}
    for lh, la, hg, ag in zip(g[("lam", "home")], g[("lam", "away")], g[("goals", "home")], g[("goals", "away")]):
        Sx = model.dixon_coles(lh, la); pr = model.outcome_probs(Sx)
        res = "1" if hg > ag else "X" if hg == ag else "2"
        out["1"].append(-np.log(max(pr[res], 1e-9)))
        for k, p, y in (("o25", Sx[tot > 2.5].sum(), hg + ag > 2.5), ("btts", Sx[1:, 1:].sum(), hg > 0 and ag > 0), ("o15", Sx[tot > 1.5].sum(), hg + ag > 1.5)):
            p = min(max(p, 1e-4), 1 - 1e-4); out[k].append(-np.log(p if y else 1 - p))
    return dev, pd.DataFrame(out, index=g.index), P

base = walk(model.GoalModel); var = walk(lambda: GM(add))
dev0, m0, P0 = evaluate(base); dev1, m1, P1 = evaluate(var)
print("\n--- variante con i candidati contro il modello attuale (stesse partite) ---")
print(f"devianza dei gol per squadra: {dev0.mean():.4f} -> {dev1.mean():.4f}  ({(1 - dev1.mean() / dev0.mean()) * 100:+.2f}%)")
for k, name in (("1", "log-loss 1X2"), ("o25", "log-loss Over 2.5"), ("o15", "log-loss Over 1.5"), ("btts", "log-loss gol entrambe")):
    d = m0[k] - m1[k]; print(f"{name:24s} {m0[k].mean():.4f} -> {m1[k].mean():.4f}  ({d.mean():+.4f}, z {d.mean() / (d.std() / np.sqrt(len(d))):+.1f})")
per = pd.DataFrame({"s": P0.season, "d0": dev0, "d1": dev1}).groupby("s").apply(lambda x: (1 - x.d1.mean() / x.d0.mean()) * 100)
print("miglioramento della devianza per stagione:", " ".join(f"{k} {v:+.2f}" for k, v in per.items()), f"| stagioni migliori {int((per > 0).sum())}/{len(per)}")
print("FINE")

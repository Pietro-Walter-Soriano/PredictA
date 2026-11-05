"""Honest version of study_players.py: the XI profiles are chosen using ONLY the seasons before the one that is tested.

  python src/study_players2.py data

study_players.py picked the best profiles on the residuals of all seasons and then scored them on the same seasons (selection bias).
Here, for every test season s: select the profiles whose correlation with the residual of the production model, over the seasons < s,
passes a Bonferroni threshold; add them to the model; refit inside s with the usual walk-forward; compare the squared error with and
without them. A gain that survives this is not an artefact of the selection."""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(HERE, "study_players.py"), encoding="utf-8").read()
exec(src[:src.index("fin = V[V.finished & V.goals.notna()]")])      # data, V with the XI profiles (PF), Y, zs, TEST_FROM, REFIT_DAYS

fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
THR = float(norm.isf(0.05 / 2 / len(PF)))
TOPK = 6

def fit_predict(t, past, nxt, add):
    ok = Y[t][past.index].notna(); pv, yv = past[ok], Y[t][past.index][ok]
    m = model.StatModel(t)
    feats = m.feats
    if t in model.WIDE_STATS:
        feats = [c for c in pv.columns if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_")) and c not in PF and pv[c].dtype.kind in "fiu" and pv[c].notna().mean() > .5] \
                + ["is_home", "lam_base", "o_lam_base"] + model.CONTEXT_STATS.get(t, [])
    m.feats = list(dict.fromkeys(feats + add))
    m.mu, m.sd = pv[m.feats].mean(), pv[m.feats].std().replace(0, 1)
    m.m = model.PoissonRegressor(alpha=m.alpha / len(pv) * 100, max_iter=500).fit(m._X(pv), yv)
    return m.predict(nxt), float(max(np.mean((yv - m.m.predict(m._X(pv))) ** 2) / np.mean(m.m.predict(m._X(pv))), 1.0))

seasons = sorted(test.season.unique())
for t in ["fouls", "yc", "sot", "shots"]:
    base_rows, new_rows, chosen = [], [], {}
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        s = nxt.season.mode().iloc[0]
        if s not in chosen:                   # profiles chosen once per season, from the residuals of the seasons before it
            earlier = pd.concat(base_rows) if base_rows else pd.DataFrame()
            earlier = earlier[earlier.season < s] if len(earlier) else earlier
            sel = []
            if len(earlier) > 1500:
                zz = {}
                for c in PF:
                    x = pd.concat([V[c].reindex(earlier.index), earlier.r], axis=1, keys=["x", "r"]).dropna()
                    if len(x) > 800 and x.x.std() > 0: zz[c] = abs(np.corrcoef(zs(x.x), x.r)[0, 1]) * np.sqrt(len(x))
                sel = [c for c, v in sorted(zz.items(), key=lambda kv: -kv[1]) if v > THR][:TOPK]
            chosen[s] = sel
        p0, d0 = fit_predict(t, past, nxt, [])
        r0 = pd.DataFrame({"p": p0, "disp": d0}, index=nxt.index).join(V[["season"]]).join(Y[t].rename("y"))
        r0["r"] = (r0.y - r0.p) / np.sqrt(r0.p * r0.disp); base_rows.append(r0.dropna(subset=["y"]))
        if chosen[s]:
            p1, _ = fit_predict(t, past, nxt, chosen[s])
            new_rows.append(pd.DataFrame({"p1": p1}, index=nxt.index))
    B = pd.concat(base_rows)
    N = pd.concat(new_rows) if new_rows else pd.DataFrame({"p1": []})
    B = B.join(N)
    B["p1"] = B.p1.fillna(B.p)                      # seasons where nothing passed the threshold keep the base model
    e0, e1 = (B.p - B.y) ** 2, (B.p1 - B.y) ** 2; d = e0 - e1
    per = pd.DataFrame({"s": B.season, "a": e0, "b": e1}).groupby("s").apply(lambda x: (1 - x.b.mean() / x.a.mean()) * 100)
    print(f"\n{t}: errore {e0.mean():.4f} -> {e1.mean():.4f} ({d.mean() / e0.mean() * 100:+.2f}%, z {d.mean() / (d.std() / np.sqrt(len(d))):+.1f}) | per stagione: " +
          " ".join(f"{k} {v:+.2f}" for k, v in per.items()), flush=True)
    for s, sel in chosen.items(): print(f"   {s}: profili scelti sulle stagioni precedenti -> {sel if sel else 'nessuno'}")
print("FINE")

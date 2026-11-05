"""Player against player, second look: ALL the 1v1 variables at once with a heavy penalty and no selection (so no selection bias).

  python src/study_duels2.py data

Stage 1 is the production model. Stage 2 is a ridge-penalised Poisson on the RATIO actual / stage-1 prediction (weights = prediction), whose
features are the 576 standardised 1v1 variables (mean / sd of the training part only). The final prediction is stage-1 x stage-2.
Walk-forward as always; several penalties; squared error with and without, season by season."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(HERE, "study_duels.py"), encoding="utf-8").read()
exec(src[:src.index("# ---------- walk-forward, residuals and honest gain ----------")])      # V with the 1v1 columns (DUEL), Y, model, TEST_FROM, REFIT_DAYS

fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
ALPHAS = (300, 3000, 30000)
ZONE = {"tutte": [c for c in DUEL if c.startswith("duel:all:")], "tutte+fasce+centro": DUEL}

for t in ["goals"] + list(model.STAT_TARGETS):
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished & V.goals.notna()]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        ok = Y[t][past.index].notna(); pv, yv = past[ok], Y[t][past.index][ok]
        if t == "goals": m0 = model.GoalModel().fit(past); p_in, p_out = m0.predict(pv), m0.predict(nxt)
        else: m0 = model.StatModel(t).fit(past, Y[t][past.index]); p_in, p_out = m0.predict(pv), m0.predict(nxt)
        r = pd.DataFrame({"y": Y[t][nxt.index].values, "p0": p_out, "season": nxt.season.values}, index=nxt.index)
        for zname, cols in ZONE.items():
            mu, sd = pv[cols].mean(), pv[cols].std().replace(0, 1)
            Xp = ((pv[cols] - mu) / sd).fillna(0.0).clip(-4, 4).values; Xn = ((nxt[cols] - mu) / sd).fillna(0.0).clip(-4, 4).values
            for al in ALPHAS:
                m = model.PoissonRegressor(alpha=al / len(pv) * 100, max_iter=300).fit(Xp, yv.values / p_in, sample_weight=p_in)
                r[f"{zname}|{al}"] = p_out * m.predict(Xn)
        rows.append(r)
    R = pd.concat(rows).dropna(subset=["y"])
    e0 = (R.p0 - R.y) ** 2
    print(f"\n{t}: errore del modello attuale {e0.mean():.4f}", flush=True)
    for k in [c for c in R.columns if "|" in c]:
        e1 = (R[k] - R.y) ** 2; d = e0 - e1
        per = pd.DataFrame({"s": R.season, "a": e0, "b": e1}).groupby("s").apply(lambda x: (1 - x.b.mean() / x.a.mean()) * 100)
        print(f"   {k:28s} {d.mean() / e0.mean() * 100:+.3f}% (z {d.mean() / (d.std() / np.sqrt(len(d))):+.1f}) stagioni migliori {int((per > 0).sum())}/{len(per)}: " + " ".join(f"{v:+.2f}" for v in per.values), flush=True)
print("FINE")

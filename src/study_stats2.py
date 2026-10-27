"""Do the systematic errors found by study_resid.py (mismatch, kick-off hour, opponent concession persistence, own x opponent
interaction) improve the count models when added as features?   python src/study_stats2.py data

Variants per target, same walk-forward as study_stats.py:
  base       the production model
  +ctx       base + |lam_gap| (mismatch), kick-off hour, own x opponent-concedes interaction
  +ctx+cl    + the opponent's concession rate at a short (3) and a long (25) half-life: the production column has a single one (8)
Reports squared error of the per-team count, log-loss of the over/under lines on the match total, and the squared-error gain season by season."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], sys.argv[1]]
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "study_stats.py"), encoding="utf-8").read()
exec(src[:src.index("fin = V[V.finished]")])        # data, V, Y, ALL, Lin, over, LINES, TEST_FROM, REFIT_DAYS

STATS = ["corners", "shots", "sot", "yc", "fouls", "offsides"]
V["lam_gap_abs"] = (V.lam_base - V.o_lam_base).abs()
V["hour"] = V.date.dt.hour + V.date.dt.minute / 60
for c in STATS:
    V[f"x_{c}"] = V[f"s_{c}"] * V[f"o_s_opp_{c}"] / (V[f"s_{c}"].mean() + 1e-9)
Ts = T.sort_values("date")
cl = Ts[["match_id", "team"]].copy()
for c in STATS:
    for hl in (3, 25):
        cl[f"o_cl_{c}_{hl}"] = Ts.groupby("team")[f"opp_{c}"].transform(lambda s: s.ewm(halflife=hl, ignore_na=True).mean().shift(1))
V = V.merge(cl.rename(columns={"team": "opp"}), on=["match_id", "opp"], how="left")
assert len(V) == len(Y["corners"]) and (V.index == Y["corners"].index).all()

def feats(t, kind):
    base = model.STAT_TARGETS[t] + ["is_home", "lam_base", "o_lam_base"]
    if t in model.WIDE_STATS: base = ALL
    f = list(base)
    if kind in ("ctx", "ctx+cl"): f += ["lam_gap_abs", "hour", f"x_{t}"] + (["x_fouls"] if t == "yc" else [])
    if kind == "ctx+cl": f += [f"o_cl_{t}_3", f"o_cl_{t}_25"]
    return list(dict.fromkeys(f))

fin = V[V.finished]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
for t in STATS:
    alpha = model.WIDE_STATS.get(t, 1.0)
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        r = pd.DataFrame({"match_id": nxt.match_id.values, "season": nxt.season.values, "y": Y[t][nxt.index].values})
        for kind in ("base", "ctx", "ctx+cl"):
            m = Lin(feats(t, kind), alpha).fit(past, Y[t][past.index]); r["p_" + kind] = m.predict(nxt); r["d_" + kind] = m.disp
        rows.append(r)
    R = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    ref = float(((Y[t][V.finished & (V.season < TEST_FROM)].mean() - R.y) ** 2).mean())
    print(f"\n{t}  ({len(R)} righe)", flush=True)
    for kind in ("base", "ctx", "ctx+cl"):
        mse = float(((R["p_" + kind] - R.y) ** 2).mean())
        g = R.groupby("match_id").agg(p=("p_" + kind, "sum"), d=("d_" + kind, "mean"), y=("y", "sum"), c=("y", "size")); g = g[g.c == 2]
        lls = []
        for ln in LINES[t]:
            pr = np.clip(np.array([over(m_, ln, d_) for m_, d_ in zip(g.p, g.d)]), 1e-4, 1 - 1e-4); yy = (g.y > ln).astype(float).values
            lls.append(float(-(yy * np.log(pr) + (1 - yy) * np.log(1 - pr)).mean()))
        extra = ""
        if kind != "base":
            e0, e1 = (R.p_base - R.y) ** 2, (R["p_" + kind] - R.y) ** 2; d = e0 - e1
            per = R.assign(e0=e0, e1=e1).groupby("season").apply(lambda x: (1 - x.e1.mean() / x.e0.mean()) * 100)
            extra = f" | vs base: {d.mean() / e0.mean() * 100:+.2f}% (z {d.mean() / (d.std() / np.sqrt(len(d))):+.1f}) stagioni migliori {int((per > 0).sum())}/{len(per)}: " + " ".join(f"{v:+.1f}" for v in per)
        print(f"  {kind:8s} MSE {mse:.3f} log-loss linee {np.mean(lls):.4f}{extra}", flush=True)
print("FINE")

"""Player against player: does the direct opponent of each player explain what the team models miss?

  python src/study_duels.py data       (needs data/dataset.pkl; writes data/study_duels.json for the dashboard)

1. DIRECT OPPONENTS. Each match: both starting elevens on the same pitch (the away team mirrored, FotMob verticalLayout), outfield players
   matched one to one by minimum total distance (Hungarian assignment). A pair is a 'flank' duel when the player stands wide, else 'centre'.
2. 1v1 FEATURES. For every pair, an own statistic (recent per-90 rate before the match: dribbles, key passes, shots, crosses, fouls suffered,
   aerial duels, ...) times an opponent statistic (tackles, duels lost, interceptions, defensive actions, ...); summed over the pairs of the
   team, for all pairs / flank pairs / centre pairs.
3. TEST. (a) correlation with the Pearson residual of the production models (goals, shots, shots on target, corners, fouls, yellow cards,
   offsides) with a Bonferroni threshold and the sign across seasons; (b) honest walk-forward: for each season the best features are chosen on the
   seasons BEFORE it only, added to the model, and the squared error with and without them is compared."""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(HERE, "study_players.py"), encoding="utf-8").read()
exec(src[:src.index("fin = V[V.finished & V.goals.notna()]")])      # data, M, P (with pre_ rates), V, Y, zs, TEST_FROM, REFIT_DAYS, model, features

S1 = ["dribbles", "kp", "sh", "sot", "npxg", "xa", "crosses", "fouled", "box_touches", "final_third", "aerials", "dispossessed", "fouls", "tackles", "duels_won", "duels_lost"]
S2 = ["tackles", "interceptions", "duels_lost", "duels_won", "def_actions", "fouls", "aerials", "clearances", "blocks", "recoveries", "dribbles", "yc"]
ALLS = sorted(set(S1) | set(S2))
FLANK = 0.2          # |x - 0.5| above this = wide player

# ---------- direct opponents ----------
RATE = P[["pre_" + c for c in ALLS]].to_numpy(float)
ROW = {(m, p): i for i, (m, p) in enumerate(zip(P.match_id.values, P.player_id.values))}
col = {c: i for i, c in enumerate(ALLS)}
i1, i2 = [col[c] for c in S1], [col[c] for c in S2]

def pairs_of(r):
    L = {s: [p for p in r.lineup[s]["starters"] if p.get("pos") != 11 and p.get("x") is not None] for s in ("home", "away")}
    if min(len(L["home"]), len(L["away"])) < 8: return None
    A = np.array([[p["x"], p["y"]] for p in L["home"]]); B = np.array([[1 - p["x"], 1 - p["y"]] for p in L["away"]])
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    ia, ib = linear_sum_assignment(cost)
    return [(L["home"][i], L["away"][j], abs(A[i, 0] - 0.5) > FLANK, float(cost[i, j])) for i, j in zip(ia, ib)]

def vec(mid, pid, idx):
    k = ROW.get((mid, pid))
    return RATE[k, idx] if k is not None else np.full(len(idx), np.nan)

feat_rows = []
for r in M.itertuples():
    pr = pairs_of(r)
    if pr is None: continue
    for side in ("home", "away"):
        acc = {z: np.zeros(len(S1) * len(S2)) for z in ("all", "flank", "centre")}; cnt = {z: np.zeros(len(S1) * len(S2)) for z in acc}
        for ph, pa, wide, _ in pr:
            own, opp = (ph, pa) if side == "home" else (pa, ph)
            prod = np.outer(vec(r.match_id, own["id"], i1), vec(r.match_id, opp["id"], i2)).ravel()
            ok = ~np.isnan(prod)
            for z in ("all", "flank" if wide else "centre"):
                acc[z][ok] += prod[ok]; cnt[z][ok] += 1
        row = {"match_id": r.match_id, "side": side}
        for z, need in (("all", 6), ("flank", 2), ("centre", 2)):
            v = np.where(cnt[z] >= need, acc[z], np.nan)
            for k, (a, b) in enumerate((a, b) for a in S1 for b in S2): row[f"duel:{z}:{a}*{b}"] = v[k]
        feat_rows.append(row)
DF = pd.DataFrame(feat_rows)
DUEL = [c for c in DF.columns if c.startswith("duel:")]
V = V.merge(DF, on=["match_id", "side"], how="left")
print(f"{len(DUEL)} variabili 1v1 ({len(S1)} statistiche proprie x {len(S2)} dell'avversario x 3 zone); righe con 1v1: {int(V[DUEL[0]].notna().sum())} su {len(V)}", flush=True)

# ---------- walk-forward, residuals and honest gain ----------
fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
THR = float(norm.isf(0.05 / 2 / len(DUEL)))
TOPK = (6, 15)
Z = {c: zs(V[c]).fillna(0.0) for c in DUEL}

def stat_fit(t, past, nxt, add):
    ok = Y[t][past.index].notna(); pv, yv = past[ok], Y[t][past.index][ok]
    m = model.StatModel(t); feats = m.feats
    if t in model.WIDE_STATS:
        feats = [c for c in pv.columns if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_")) and c not in PF and pv[c].dtype.kind in "fiu" and pv[c].notna().mean() > .5] \
                + ["is_home", "lam_base", "o_lam_base"] + model.CONTEXT_STATS.get(t, [])
    m.feats = list(dict.fromkeys(feats + add))
    m.mu, m.sd = pv[m.feats].mean(), pv[m.feats].std().replace(0, 1)
    m.m = model.PoissonRegressor(alpha=m.alpha / len(pv) * 100, max_iter=500).fit(m._X(pv), yv)
    mu = m.m.predict(m._X(pv))
    return m.predict(nxt), float(max(np.mean((yv - mu) ** 2) / np.mean(mu), 1.0))

def goal_fit(past, nxt, add):
    class G(model.GoalModel):
        cols = model.GOAL_FEATS + add
        def _X(self, V_): return ((V_[self.cols].astype(float) - self.mu) / self.sd).fillna(0.0).clip(-4, 4)
        def fit(self, V_):
            V_ = V_[V_.finished & V_.goals.notna() & V_.lam_base.notna()]
            self.mu, self.sd = V_[self.cols].mean(), V_[self.cols].std().replace(0, 1)
            self.m = model.PoissonRegressor(alpha=self.alpha / len(V_) * 100, max_iter=500).fit(self._X(V_), V_.goals.values / V_.lam_base.values, sample_weight=V_.lam_base.values); return self
    return G().fit(past).predict(nxt), 1.0

def fit(t, past, nxt, add): return goal_fit(past, nxt, add) if t == "goals" else stat_fit(t, past, nxt, add)

def scan(R):
    """Correlation of every 1v1 feature with the residual r of R (rows indexed like V), and the sign agreement across seasons."""
    Zm = pd.DataFrame({c: Z[c].reindex(R.index) for c in DUEL}); r = R.r.values; n = len(R)
    rr = (Zm.values * r[:, None]).mean(0) / (Zm.values.std(0) + 1e-9) / (r.std() + 1e-9)
    agree = np.zeros(len(DUEL)); ns = 0
    for s in sorted(R.season.unique()):
        mk = (R.season == s).values
        if mk.sum() < 150: continue
        agree += np.sign((Zm.values[mk] * r[mk][:, None]).mean(0)); ns += 1
    return pd.DataFrame({"feature": DUEL, "corr": rr, "z": np.abs(rr) * np.sqrt(n), "agree": np.abs(agree), "seasons": ns})

report = {"thr": THR, "n_features": len(DUEL), "targets": {}}
for t in ["goals"] + list(model.STAT_TARGETS):
    base_rows, new_rows, chosen = [], {k: [] for k in TOPK}, {}
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        s = nxt.season.mode().iloc[0]
        if s not in chosen:                                   # selection from the residuals of the seasons BEFORE s
            earlier = pd.concat(base_rows) if base_rows else pd.DataFrame()
            earlier = earlier[earlier.season < s] if len(earlier) else earlier
            chosen[s] = {k: [] for k in TOPK}
            if len(earlier) > 1500:
                sc = scan(earlier); sc = sc[sc.z > THR].sort_values("z", ascending=False)
                for k in TOPK: chosen[s][k] = sc.feature.tolist()[:k]
        p0, d0 = fit(t, past, nxt, [])
        r0 = pd.DataFrame({"p": p0, "disp": d0}, index=nxt.index).join(V[["season"]]).join(Y[t].rename("y"))
        r0["r"] = (r0.y - r0.p) / np.sqrt(r0.p * r0.disp); base_rows.append(r0.dropna(subset=["y"]))
        for k in TOPK:
            if chosen[s][k]:
                p1, _ = fit(t, past, nxt, chosen[s][k]); new_rows[k].append(pd.DataFrame({f"p{k}": p1}, index=nxt.index))
    B = pd.concat(base_rows); out = {"gain": {}}
    for k in TOPK:
        N = pd.concat(new_rows[k]) if new_rows[k] else pd.DataFrame({f"p{k}": []})
        Bk = B.join(N); Bk[f"p{k}"] = Bk[f"p{k}"].fillna(Bk.p)
        e0, e1 = (Bk.p - Bk.y) ** 2, (Bk[f"p{k}"] - Bk.y) ** 2; d = e0 - e1
        per = pd.DataFrame({"s": Bk.season, "a": e0, "b": e1}).groupby("s").apply(lambda x: (1 - x.b.mean() / x.a.mean()) * 100)
        out["gain"][k] = {"err0": float(e0.mean()), "err1": float(e1.mean()), "pct": float(d.mean() / e0.mean() * 100), "z": float(d.mean() / (d.std() / np.sqrt(len(d)))),
                          "seasons": {s_: round(float(v), 3) for s_, v in per.items()}, "better": int((per > 0).sum()), "tested": int((per != 0).sum())}
    sc = scan(B); top = sc.sort_values("z", ascending=False).head(12)
    out["top"] = [{"feature": r_.feature, "corr": round(float(r_["corr"]), 4), "z": round(float(r_.z), 2), "agree": int(r_.agree), "seasons": int(r_.seasons)} for _, r_ in top.iterrows()]
    out["n_above"] = int((sc.z > THR).sum()); out["chosen"] = {s_: v for s_, v in chosen.items()}
    report["targets"][t] = out
    print(f"\n=== {t}: {out['n_above']} variabili 1v1 sopra soglia (|z|>{THR:.1f}; per caso ne attendo ~{len(DUEL) * 0.0027:.1f} sopra 3)", flush=True)
    for k in TOPK:
        g = out["gain"][k]
        print(f"   +{k:2d} variabili scelte sulle stagioni precedenti: errore {g['err0']:.4f} -> {g['err1']:.4f} ({g['pct']:+.2f}%, z {g['z']:+.1f}) | stagioni migliori {g['better']}/{g['tested']}: " + " ".join(f"{s_} {v:+.2f}" for s_, v in g["seasons"].items()), flush=True)
    for x in out["top"][:6]: print(f"      {x['feature']:40s} corr {x['corr']:+.3f} z {x['z']:.1f}  stessa direzione in {x['agree']}/{x['seasons']} stagioni")
os.makedirs(DATA, exist_ok=True)
def clean(o):   # NaN / Infinity are not valid JSON: the browser would reject the whole page
    if isinstance(o, float): return o if np.isfinite(o) else None
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o
json.dump(clean(report), open(f"{DATA}/study_duels.json", "w"), indent=1, default=float, allow_nan=False)
print("\nscritto", f"{DATA}/study_duels.json\nFINE")

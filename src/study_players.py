"""Do the per-player statistics of the starting XI (crosses, tackles, duels, aerials, dribbles, key passes, ...) explain what the team models miss?

  python src/study_players.py data       (needs data/dataset.pkl)

For every team-match, each statistic is summed over the 11 starters using each player's exponentially weighted per-90 rate computed strictly
BEFORE the match (features.build: pre_<stat>), times his expected minutes. Also the same for the OPPOSING XI (what the other eleven concede /
is good at). Those XI profiles are tested:
  1. against the Pearson residual of the production count models (corners, shots, shots on target, yellow cards, fouls, offsides) and of the
     goal model, with a Bonferroni threshold and the sign agreement across seasons;
  2. as extra features, season by season, in a walk-forward: the squared error of the model with and without them."""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM, REFIT_DAYS = "2021-2022", 42
RATES = ["npxg", "xa", "sh", "sot", "kp", "box_touches", "yc", "fouls", "fouled", "def_actions", "tackles", "interceptions", "recoveries", "aerials", "dribbles"]
D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
V = model.team_view(M, T, model.lineup_strength(P, M), model.referee_table(M, T))
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
Y = {"goals": V.goals}
for t in model.STAT_TARGETS: Y[t] = pd.Series([tm[t].get((m, s), np.nan) if t in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index)

# ---------- XI profiles from the player table ----------
extra_cols = ["crosses", "duels_won", "duels_lost", "dispossessed", "final_third", "touches", "clearances", "blocks"]
for c in extra_cols:                                   # not in features.PLAYER_RATES: same per-90 EW rate, built here and shifted by one match
    w = P["min"] / 90
    alpha = 1 - 0.5 ** (1 / 10)
    num = (P[c].fillna(0) * w).groupby(P["player_id"], sort=False).transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean())
    den = (P[c].notna() * w).groupby(P["player_id"], sort=False).transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean())
    P["pre_" + c] = (num / den.replace(0, np.nan)).groupby(P["player_id"], sort=False).shift(1)
ALLR = RATES + extra_cols
st = P[P.starter & (P.comp == "serie_a") & P.venue.notna()].copy()
for c in ALLR: st["xi_" + c] = st["pre_" + c]          # per-90 rate of each starter before the match; the XI profile is the sum over the 11
xi =st.groupby(["match_id", "venue"])[["xi_" + c for c in ALLR]].sum(min_count=6).reset_index().rename(columns={"venue": "side"})
V = V.merge(xi, on=["match_id", "side"], how="left")
opp = xi.rename(columns={c: "o_" + c for c in xi.columns if c.startswith("xi_")}); opp["side"] = opp["side"].map({"home": "away", "away": "home"})
V = V.merge(opp, on=["match_id", "side"], how="left")
PF = [c for c in V.columns if c.startswith(("xi_", "o_xi_")) and c.split("xi_", 1)[1] in ALLR]
print(f"{len(PF)} profili dell'undici ({len(ALLR)} statistiche x propria/avversaria); righe con profilo: {int(V[PF[0]].notna().sum())} su {len(V)}", flush=True)

def zs(s): s = s.astype(float); return ((s - s.mean()) / (s.std() + 1e-9)).clip(-4, 4)

fin = V[V.finished & V.goals.notna()]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
rows, preds = [], {}
for t in ["goals"] + list(model.STAT_TARGETS):
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        if t == "goals": mdl = model.GoalModel().fit(past); p, disp = mdl.predict(nxt), 1.0
        else: mdl = model.StatModel(t).fit(past, Y[t][past.index]); p, disp = mdl.predict(nxt), mdl.dispersion
        parts.append(pd.DataFrame({"p": p, "disp": disp}, index=nxt.index))
    R = pd.concat(parts).join(V[["season", "date"]]).join(Y[t].rename("y")).dropna(subset=["y"])
    R["r"] = (R.y - R.p) / np.sqrt(R.p * R.disp); preds[t] = R
    for c in PF:
        s = pd.concat([V[c].reindex(R.index), R.r, R.season], axis=1, keys=["x", "r", "s"]).dropna()
        if len(s) < 500 or s.x.std() == 0: continue
        z = zs(s.x); rr = np.corrcoef(z, s.r)[0, 1]
        by = [np.sign(np.corrcoef(z[s.s == k], s.r[s.s == k])[0, 1]) for k in sorted(s.s.unique()) if (s.s == k).sum() > 150]
        rows.append((t, c, len(s), rr, abs(rr) * np.sqrt(len(s)), max(sum(v > 0 for v in by), sum(v < 0 for v in by)), len(by)))
    print("residui", t, len(R), flush=True)
S = pd.DataFrame(rows, columns=["target", "feature", "n", "corr", "z", "seasons_same_sign", "seasons"]).sort_values("z", ascending=False)
thr = float(norm.isf(0.05 / 2 / len(S)))
S.to_csv(f"{DATA}/study_players.csv", index=False)
print(f"\ntest {len(S)}, soglia Bonferroni |z|>{thr:.1f}; per caso con |z|>3 ne attendo ~{len(S) * 0.0027:.1f}; sopra soglia: {int((S.z > thr).sum())}")
for t in ["goals"] + list(model.STAT_TARGETS):
    s = S[S.target == t].head(6)
    print(f"\n{t}\n" + s.assign(corr=s['corr'].round(3), z=s.z.round(1)).drop(columns="target").to_string(index=False))

# ---------- walk-forward with the best XI features added ----------
print("\n=== aggiungendo i migliori profili al modello (stesso walk-forward, errore quadratico) ===", flush=True)
for t in ["goals"] + list(model.STAT_TARGETS):
    best = S[(S.target == t) & (S.z > thr)].feature.tolist()[:6]
    if not best: print(t, ": nessun profilo sopra soglia, nessun test"); continue
    for c in best: V["cand:" + c] = V[c]
    add = ["cand:" + c for c in best]; R0 = preds[t]; rows2 = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished & V.goals.notna()]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        if t == "goals":
            class G(model.GoalModel):
                def _X(self, V_): return ((V_[model.GOAL_FEATS + add].astype(float) - self.mu) / self.sd).fillna(0.0).clip(-4, 4)
                def fit(self, V_):
                    V_ = V_[V_.finished & V_.goals.notna() & V_.lam_base.notna()]; cols = model.GOAL_FEATS + add
                    self.mu, self.sd = V_[cols].mean(), V_[cols].std().replace(0, 1)
                    self.m = model.PoissonRegressor(alpha=self.alpha / len(V_) * 100, max_iter=500).fit(self._X(V_), V_.goals.values / V_.lam_base.values, sample_weight=V_.lam_base.values); return self
            p = G().fit(past).predict(nxt)
        else:
            m = model.StatModel(t); m.feats = list(dict.fromkeys(m.feats + add))
            ok = Y[t][past.index].notna(); pv = past[ok]; yv = Y[t][past.index][ok]
            if t in model.WIDE_STATS: m.feats = [c for c in pv.columns if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_")) and pv[c].dtype.kind in "fiu" and pv[c].notna().mean() > .5] + ["is_home", "lam_base", "o_lam_base"] + model.CONTEXT_STATS.get(t, []) + add
            m.mu, m.sd = pv[m.feats].mean(), pv[m.feats].std().replace(0, 1)
            m.m = model.PoissonRegressor(alpha=m.alpha / len(pv) * 100, max_iter=500).fit(m._X(pv), yv); p = m.predict(nxt)
        rows2.append(pd.DataFrame({"p1": p}, index=nxt.index))
    R1 = R0.join(pd.concat(rows2)); R1 = R1.dropna(subset=["p1"])
    e0, e1 = (R1.p - R1.y) ** 2, (R1.p1 - R1.y) ** 2; d = e0 - e1
    per = pd.DataFrame({"s": R1.season, "a": e0, "b": e1}).groupby("s").apply(lambda x: (1 - x.b.mean() / x.a.mean()) * 100)
    print(f"{t:9s} +{len(best)} profili: errore {e0.mean():.4f} -> {e1.mean():.4f} ({d.mean() / e0.mean() * 100:+.2f}%, z {d.mean() / (d.std() / np.sqrt(len(d))):+.1f}) | stagioni migliori {int((per > 0).sum())}/{len(per)}: " + " ".join(f"{v:+.2f}" for v in per.values), flush=True)
print("FINE")

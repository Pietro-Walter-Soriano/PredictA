"""Where do the count models (corners, shots, shots on target, cards, fouls, offsides) still go wrong?

  python src/study_resid.py data     (needs data/dataset.pkl)

1. Walk-forward predictions of the CURRENT StatModel (refit every REFIT_DAYS on strictly earlier matches).
2. Pearson residual r = (y - p) / sqrt(p * dispersion) for every team-match of the test seasons.
3. Candidate features (game context, calendar, weather, formations, strength gap, rest, plus every style column the model does not use)
   and "error persistence" features (what the model got wrong so far for this team / this opponent / this referee) are correlated
   with r. A candidate counts if |z| passes a Bonferroni threshold AND the sign agrees in most seasons."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features, model

DATA = sys.argv[1]
TEST_FROM, REFIT_DAYS = "2021-2022", 42
D = pd.read_pickle(f"{DATA}/dataset.pkl")
M, T, P, _ = features.build(D)
V = model.team_view(M, T, model.lineup_strength(P, M), model.referee_table(M, T))
V["o_lam_base"] = V.groupby("match_id")["lam_base"].transform(lambda s: s[::-1].values)
tm = T.set_index(["match_id", "venue"])
Y = {t: pd.Series([tm[t].get((m, s), np.nan) if t in tm else np.nan for m, s in zip(V.match_id, V.side)], index=V.index) for t in model.STAT_TARGETS}

# ---------- candidate features, all known before kick-off ----------
X = pd.DataFrame(index=V.index)
X["lam_gap"] = V.lam_base - V.o_lam_base                       # favourite (+) / underdog (-)
X["lam_gap_abs"] = X.lam_gap.abs()                             # mismatch: the weaker side chases the game
X["lam_sum"] = V.lam_base + V.o_lam_base                       # expected total goals
X["lam_gap_sq"] = X.lam_gap ** 2
Mx = M.set_index("match_id")
for c in ("temp", "wind", "rain", "round"): X[c] = V.match_id.map(Mx[c]).astype(float)
X["hour"] = V.date.dt.hour + V.date.dt.minute / 60
X["night"] = (V.date.dt.hour >= 18).astype(float)
X["dow"] = V.date.dt.dayofweek; X["sunday"] = (X.dow == 6).astype(float); X["midweek"] = X.dow.isin([1, 2, 3]).astype(float)
X["month"] = V.date.dt.month
X["season_progress"] = X["round"] / 38
form = np.where(V.side == "home", V.match_id.map(Mx.home_formation), V.match_id.map(Mx.away_formation))
oform = np.where(V.side == "home", V.match_id.map(Mx.away_formation), V.match_id.map(Mx.home_formation))
f3 = lambda a: pd.Series(a, index=V.index).astype(str).str.startswith("3").astype(float)
f4 = lambda a: pd.Series(a, index=V.index).astype(str).str.startswith("4").astype(float)
X["back3"], X["o_back3"], X["back4"], X["o_back4"] = f3(form), f3(oform), f4(form), f4(oform)
X["back3_vs_back4"] = X.back3 * X.o_back4
X["mv_ratio"] = np.log((V.xi_mv + 5) / (V.o_xi_mv + 5)); X["mv_sum"] = np.log(V.xi_mv + V.o_xi_mv + 10)
Vs = V.sort_values("date")
for who, name in (("team", "rest"), ("opp", "o_rest")):
    X[name] = Vs.groupby(who)["date"].diff().dt.total_seconds().div(86400).clip(upper=14).reindex(V.index)
X["rest_gap"] = X.rest - X.o_rest
X["short_rest"] = (X.rest <= 3.5).astype(float)
X["derby_proxy"] = ((V.team.map(V.groupby("team").size()) > 0) & (V.xi_mv > V.xi_mv.quantile(.75)) & (V.o_xi_mv > V.o_xi_mv.quantile(.75))).astype(float)
X["both_top"] = X.derby_proxy
X["ref_experience"] = np.log1p(V.ref_n)
X["is_home"] = V.is_home
# interactions between own style and what the opponent allows (the model has both columns, not their product)
for c in ("corners", "shots", "sot", "fouls", "yc", "offsides"):
    if f"s_{c}" in V and f"o_s_opp_{c}" in V: X[f"x_{c}"] = V[f"s_{c}"] * V[f"o_s_opp_{c}"] / (V[f"s_{c}"].mean() + 1e-9)
# every numeric style / lineup / referee column (the scan tells which of them add anything beyond the production list)
for c in V.columns:
    if c.startswith(("s_", "o_s_", "xi_", "o_xi_", "ref_")) and V[c].dtype.kind in "fiu" and V[c].notna().mean() > .5: X["col:" + c] = V[c]

def z(s):
    s = s.astype(float); s = (s - s.mean()) / (s.std() + 1e-9); return s.clip(-4, 4)

fin = V[V.finished]; test = fin[fin.season >= TEST_FROM].sort_values("date")
cuts = pd.date_range(test.date.min(), test.date.max() + pd.Timedelta(days=REFIT_DAYS), freq=f"{REFIT_DAYS}D")
rows = []
for t in model.STAT_TARGETS:
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        past = V[(V.date < a) & V.finished]; nxt = V[(V.date >= a) & (V.date < b) & V.finished]
        if nxt.empty or len(past) < 300 or Y[t][nxt.index].notna().sum() == 0: continue
        m = model.StatModel(t).fit(past, Y[t][past.index])
        parts.append(pd.DataFrame({"p": m.predict(nxt), "disp": m.dispersion}, index=nxt.index))
    R = pd.concat(parts).join(Y[t].rename("y")).dropna(subset=["y"])
    R["r"] = (R.y - R.p) / np.sqrt(R.p * R.disp)
    R = R.join(V[["match_id", "season", "date", "team", "opp", "referee", "side"]])
    print(f"\n=== {t}: {len(R)} team-match, mean residual {R.r.mean():+.3f}, sd {R.r.std():.3f}", flush=True)
    # error persistence, strictly from earlier rows: team, opponent (what the opponent concedes), referee (both sides together)
    R = R.sort_values(["date", "match_id", "side"])
    for who, key in (("team", "team"), ("opp", "opp"), ("ref", "referee")):
        for hl in (6, 15):
            e = R.groupby(key)["r"].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3).mean())
            n = R.groupby(key)["r"].transform(lambda s: s.shift(1).expanding().count())
            R[f"persist:{who}_hl{hl}"] = e * n / (n + 4)       # shrink towards 0 with 4 pseudo-matches
    cand = pd.concat([R[[c for c in R.columns if c.startswith("persist:")]], X.loc[R.index]], axis=1)
    for c in cand.columns:
        s = pd.concat([cand[c], R.r, R.season], axis=1, keys=["x", "r", "s"]).dropna()
        if len(s) < 300 or s.x.std() == 0: continue
        zz = z(s.x); rr = np.corrcoef(zz, s.r)[0, 1]
        by = [np.sign(np.corrcoef(zz[s.s == k], s.r[s.s == k])[0, 1]) for k in sorted(s.s.unique()) if (s.s == k).sum() > 100]
        agree = max(sum(1 for v in by if v > 0), sum(1 for v in by if v < 0))
        rows.append((t, c, len(s), rr, abs(rr) * np.sqrt(len(s)), agree, len(by)))
S = pd.DataFrame(rows, columns=["target", "feature", "n", "corr", "z", "seasons_same_sign", "seasons"]).sort_values("z", ascending=False)
thr = 4.4
S.to_csv(f"{DATA}/study_resid.csv", index=False)
print(f"\ntest totali {len(S)}, soglia Bonferroni |z|>{thr}; attesi per caso con |z|>3: ~{len(S) * 0.0027:.0f}")
for t in model.STAT_TARGETS:
    s = S[S.target == t].head(8)
    print(f"\n{t}\n" + s.assign(corr=s["corr"].round(3), z=s.z.round(1)).to_string(index=False, header=True))
print("\nsopra soglia:", int((S.z > thr).sum()))

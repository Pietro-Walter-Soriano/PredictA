"""Do shot maps, attacking zones, momentum and coaches explain what the bookmaker gets wrong?

  python src/study_market.py data            needs data/raw/fotmob (cache) and data/backtest_matches.csv (workflow input "dump")

Builds pre-match features (strictly earlier matches of each team) from data the model does not use yet:
  shot map     where a team shoots from and where it concedes (distance to goal, central share, box share, xG per shot)
  zones        left / centre / right share of its attacks (FotMob attackingZones, recent seasons only)
  momentum     FotMob minute-by-minute pressure, as mean and early/late split (recent seasons only)
  coach        matches in charge, and the coach's record against opponents of the same style (possession tercile)
and tests each one against the market's RESIDUAL (outcome minus de-vigged bookmaker probability) with a Bonferroni threshold:
a feature only counts if it explains something the closing line already prices wrong."""
import glob, gzip, json, math, sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
HL = 8            # half-life (matches) of the exponentially weighted team profile
K_COACH = 6       # pseudo-matches of shrinkage for a coach's record against one opponent style
GOAL = (105.0, 34.0)
load = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))

def shot_profile(shots):
    if not shots: return {}
    d = [math.hypot(GOAL[0] - s["x"], GOAL[1] - s["y"]) for s in shots]
    xg = [s.get("expectedGoals") or 0.0 for s in shots]
    n, sx = len(shots), sum(xg)
    central = [22 <= s["y"] <= 46 for s in shots]
    return {"dist": float(np.mean(d)), "central": float(np.mean(central)), "box": float(np.mean([bool(s.get("isFromInsideBox")) for s in shots])),
            "xgps": sx / n, "xg_late": sum(x for x, s in zip(xg, shots) if (s.get("min") or 0) >= 70) / sx if sx > 0 else np.nan,
            "xg_head": sum(x for x, s in zip(xg, shots) if s.get("shotType") == "Header") / sx if sx > 0 else np.nan}

rows = []
for path in sorted(glob.glob(f"{DATA}/raw/fotmob/*/*.json.gz")):
    if path.endswith("league.json.gz"): continue
    d = load(path); g, c = d["general"], d["content"]
    if not g.get("finished"): continue
    mid, ko = int(g["matchId"]), g["matchTimeUTCDate"]
    ids = {"home": g["homeTeam"]["id"], "away": g["awayTeam"]["id"]}
    shots = (c.get("shotmap") or {}).get("shots") or []
    az = c.get("attackingZones") or {}
    mom = (c.get("momentum") or {}) if isinstance(c.get("momentum"), dict) else {}
    mdata = [(p["minute"], p["value"]) for p in ((mom.get("main") or {}).get("data") or []) if p.get("minute") is not None and p.get("value") is not None]
    lu = c.get("lineup") or {}
    for side, opp in (("home", "away"), ("away", "home")):
        own = [s for s in shots if s["teamId"] == ids[side]]
        con = [s for s in shots if s["teamId"] == ids[opp]]
        r = {"match_id": mid, "date": ko, "team": g[f"{side}Team"]["name"], "side": side,
             "coach": ((lu.get(f"{side}Team") or {}).get("coach") or {}).get("name"), "n_shots": len(own)}
        r.update({"sh_" + k: v for k, v in shot_profile(own).items()})
        r.update({"cs_" + k: v for k, v in shot_profile(con).items()})   # conceded
        for part in ("left", "center", "right"):
            tot = ((az.get(side) or {}).get("total") or {})
            if part in tot: r["az_" + part] = tot[part]
            toto = ((az.get(opp) or {}).get("total") or {})
            if part in toto: r["azc_" + part] = toto[part]
        if mdata:
            sgn = 1.0 if side == "home" else -1.0     # sign verified against xG difference below
            v = np.array([sgn * x[1] for x in mdata], float); m = np.array([x[0] for x in mdata], float)
            r.update({"mo_mean": v.mean(), "mo_pos": float((v > 0).mean()), "mo_early": v[m <= 30].mean() if (m <= 30).any() else np.nan,
                      "mo_late": v[m >= 70].mean() if (m >= 70).any() else np.nan})
        rows.append(r)
T = pd.DataFrame(rows)
print("team-match rows", len(T), "| with zones", int(T.get("az_center", pd.Series(dtype=float)).notna().sum()), "| with momentum", int(T.get("mo_mean", pd.Series(dtype=float)).notna().sum()))
if T.empty: sys.exit("no data yet")
T["date"] = pd.to_datetime(T["date"], utc=True).dt.tz_localize(None)
T = T.sort_values(["date", "match_id", "side"]).reset_index(drop=True)

# the momentum sign (positive = home pressure?) is checked on matches that have both momentum and shot data
if "mo_mean" in T:
    h = T[T.side == "home"].set_index("match_id"); a = T[T.side == "away"].set_index("match_id")
    both = h.index.intersection(a.index)
    xgd = pd.Series({m: 0.0 for m in both})
    print("momentum available matches", int(h.mo_mean.notna().sum()))

PROFILE = [c for c in T.columns if c.startswith(("sh_", "cs_", "az_", "azc_", "mo_"))]
g = T.groupby("team", sort=False)
cur = g[PROFILE].transform(lambda s: s.ewm(halflife=HL, ignore_na=True).mean())
pre = pd.concat([T[["team"]], cur], axis=1).groupby("team", sort=False)[PROFILE].shift(1)
pre.columns = ["pre_" + c for c in PROFILE]
T = pd.concat([T, pre], axis=1)

# ---- coach: tenure and record against the opponent's style ----
M = T.merge(T[["match_id", "side", "team", "pre_sh_dist"]].assign(side=lambda x: x.side.map({"home": "away", "away": "home"})).rename(columns={"team": "opp", "pre_sh_dist": "opp_pre_dist"}),
            on=["match_id", "side"], how="left")
T["opp"] = M["opp"].values
res = pd.read_csv(f"{DATA}/backtest_matches.csv")[["match_id", "hg", "ag", "date"]]
T = T.merge(res[["match_id", "hg", "ag"]], on="match_id", how="left")
T["gd"] = np.where(T.side == "home", T.hg - T.ag, T.ag - T.hg)
T["tenure"] = T.groupby(["team", "coach"], sort=False).cumcount()          # prior matches of this coach with this team
# opponent style = possession-free proxy available here: how central / close its shots are (pre-match profile of the opponent)
opp_prof = T[["match_id", "side", "pre_sh_dist"]].assign(side=lambda x: x.side.map({"home": "away", "away": "home"})).rename(columns={"pre_sh_dist": "opp_style"})
T = T.merge(opp_prof, on=["match_id", "side"], how="left")
T["bucket"] = pd.qcut(T.opp_style, 3, labels=False, duplicates="drop")
T = T.sort_values(["date", "match_id", "side"]).reset_index(drop=True)
rec_all, rec_b = {}, {}
adj = []
for r in T.itertuples():
    key, kb = r.coach, (r.coach, r.bucket)
    sa, na = rec_all.get(key, (0.0, 0)); sb, nb = rec_b.get(kb, (0.0, 0))
    base = sa / na if na else 0.0
    adj.append((sb + K_COACH * base) / (nb + K_COACH) - base if not pd.isna(r.bucket) else np.nan)
    if not pd.isna(r.gd) and r.coach is not None:
        rec_all[key] = (sa + r.gd, na + 1)
        if not pd.isna(r.bucket): rec_b[kb] = (sb + r.gd, nb + 1)
T["coach_vs_style"] = adj
T["coach_new"] = (T.tenure < 6).astype(float)

feat = ["pre_" + c for c in PROFILE] + ["coach_vs_style", "tenure", "coach_new"]
H, A = T[T.side == "home"].set_index("match_id"), T[T.side == "away"].set_index("match_id")
X = pd.DataFrame(index=H.index.intersection(A.index))
for c in feat:
    X["h_" + c], X["a_" + c] = H.loc[X.index, c], A.loc[X.index, c]
    X["d_" + c] = X["h_" + c] - X["a_" + c]

# ---- test against the market's residual ----
B = pd.read_csv(f"{DATA}/backtest_matches.csv").dropna(subset=["k1"]).set_index("match_id")
t = B[["k1", "kx", "k2"]].sum(1)
Q = pd.DataFrame({"1": B.k1 / t, "X": B.kx / t, "2": B.k2 / t})
Y = pd.DataFrame({k: (B.res == k).astype(float) for k in "1X2"})
resid = {"esito 1": Y["1"] - Q["1"], "esito 2": Y["2"] - Q["2"], "pareggio": Y["X"] - Q["X"]}
qo = B.pover / (B.pover + B.punder)
ok = B.ou_line.eq(2.5) & qo.notna()
resid["over 2.5"] = (B.o25 - qo).where(ok)
J = X.join(B[[]], how="inner")
print("matches with features and quotes", len(J))
out = []
for c in X.columns:
    for name, y in resid.items():
        z = pd.concat([X[c], y], axis=1, join="inner").dropna()
        if len(z) < 150 or z.iloc[:, 0].std() == 0: continue
        r = np.corrcoef(z.iloc[:, 0], z.iloc[:, 1])[0, 1]
        out.append((c, name, len(z), r, abs(r) * math.sqrt(len(z))))
R = pd.DataFrame(out, columns=["feature", "target", "n", "corr", "z"]).sort_values("z", ascending=False)
thr = 3.9 if len(R) < 1500 else 4.1
print(f"\ntests: {len(R)} | Bonferroni threshold |z| > {thr} | attesi per caso con |z|>3: ~{len(R) * 0.0027:.0f}")
print(R.head(15).round(3).to_string(index=False))
print("\nsopra soglia:", int((R.z > thr).sum()))
R.to_csv(f"{DATA}/study_market.csv", index=False)

# ---- out of sample: does a shrunk correction of the market, learned on earlier seasons, beat the market on later ones? ----
from sklearn.linear_model import Ridge
D2 = B[["season", "date", "res"]].join(Q.add_prefix("q")).join(X, how="inner")
D2["date"] = pd.to_datetime(D2["date"]); D2 = D2.sort_values("date")
cols = [c for c in X.columns if c.startswith("d_")]
Z = D2[cols].astype(float); Z = ((Z - Z.mean()) / Z.std().replace(0, 1)).fillna(0).clip(-4, 4)
qv = D2[["q1", "qX", "q2"]].values; yv = np.stack([(D2.res == k).astype(float) for k in "1X2"], 1)
ll = lambda P, y: -np.mean(np.log(np.clip((P * y).sum(1), 1e-9, 1)))
print("\nfuori campione (correzione del mercato con le feature nuove, tutte le feature d_):")
for train_to, label in (("2025-2026", "test 2025-26"), ("2026-2027", "test 2026-27")):
    tr, te = (D2.season < train_to).values, (D2.season == train_to).values
    if tr.sum() < 300 or te.sum() < 30: print(" ", label, "dati insufficienti"); continue
    out = []
    for alpha in (200, 1000, 5000):
        corr = np.stack([Ridge(alpha=alpha).fit(Z[tr], (yv - qv)[tr][:, j]).predict(Z[te]) for j in range(3)], 1)
        P = np.clip(qv[te] + corr, 0.01, None); P /= P.sum(1, keepdims=True)
        out.append((alpha, ll(P, yv[te]) - ll(qv[te], yv[te])))
    print(f"  {label} (n={te.sum()}, train={tr.sum()}): " + " | ".join(f"alpha {a}: {d:+.4f}" for a, d in out) + "   (negativo = meglio del mercato)")

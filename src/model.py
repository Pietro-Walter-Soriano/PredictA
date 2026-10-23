"""Match model: base Elo-xG goal rates + learned adjustments (style matchup, lineup, referee) + per-stat count models.

All learned models are Poisson GLMs fitted only on matches strictly before the ones they predict.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

RHO = -0.06
MAXG = 10
GOAL_STYLE = ["poss", "ppda", "deep", "long_balls", "crosses", "box_touches", "xg_set", "shot_xg_fastbreak", "aerials"]
STAT_TARGETS = {  # target column on team_match -> style features of the team (s_) and of the opponent (o_)
    "corners": ["s_corners", "s_crosses", "s_poss", "s_box_touches", "o_s_opp_corners", "o_s_poss", "o_s_corners"],
    "shots": ["s_shots", "s_poss", "s_box_touches", "o_s_opp_shots", "o_s_ppda"],
    "sot": ["s_sot", "s_shots", "o_s_opp_shots", "o_s_saves"],
    "yc": ["s_yc", "s_fouls", "o_s_opp_yc", "s_tackles", "ref_yc", "o_s_dribbles"],
    "fouls": ["s_fouls", "s_tackles", "o_s_opp_fouls", "o_s_dribbles", "ref_fouls"],
    "offsides": ["s_offsides", "o_s_ppda", "s_long_balls"],
}


def dixon_coles(lh, la, rho=RHO):
    i = np.arange(MAXG + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    return M / M.sum()


def outcome_probs(M):
    return {"1": np.tril(M, -1).sum(), "X": np.trace(M), "2": np.triu(M, 1).sum()}


def lineup_strength(P, matches):
    """Per match/side: expected attacking output (npxG+xA per 90 * expected minutes) of the XI vs the team's recent XIs,
    average pre-match FotMob rating and form of the XI, and its fatigue (all competitions, strictly before kick-off)."""
    key = P.set_index(["match_id", "player_id"])
    last = P.sort_values("date").groupby("player_id").tail(1).set_index("player_id")  # latest state for upcoming matches
    apps = {pid: (g.date.values, g["min"].values) for pid, g in P[P["min"] > 0][["player_id", "date", "min"]].groupby("player_id")}
    def fatigue(pid, ko):
        d, m = apps.get(pid, (np.array([], dtype="datetime64[ns]"), np.array([])))
        before = d < np.datetime64(ko)
        d, m = d[before], m[before]
        if not len(d): return np.nan, 0.0, 0
        rest = (np.datetime64(ko) - d[-1]) / np.timedelta64(1, "D")
        return min(rest, 14.0), float(m[d >= np.datetime64(ko) - np.timedelta64(14, "D")].sum()), int((d >= np.datetime64(ko) - np.timedelta64(10, "D")).sum())
    rows = []
    for r in matches.itertuples():
        for side in ("home", "away"):
            xi = r.lineup[side]["starters"]
            if len(xi) < 11:
                rows.append({"match_id": r.match_id, "side": side}); continue
            att, rat, frm, mv, known = 0.0, [], [], 0.0, 0
            fat = [fatigue(s["id"], r.date) for s in xi]
            for s in xi:
                if (r.match_id, s["id"]) in key.index:
                    p = key.loc[(r.match_id, s["id"])]
                    p = p.iloc[0] if isinstance(p, pd.DataFrame) else p
                    a90, rt, f5 = (p.get("pre_npxg"), p.get("pre_xa")), p.get("pre_rating"), p.get("pre_form5")
                elif s["id"] in last.index:  # upcoming match: player's state after his latest game
                    p = last.loc[s["id"]]
                    a90, rt, f5 = (p.get("cur_npxg"), p.get("cur_xa")), p.get("cur_rating"), p.get("cur_form5")
                else:
                    a90, rt, f5 = (None, None), None, None
                if a90[0] is not None and not pd.isna(a90[0]):
                    att += (a90[0] or 0) + (a90[1] or 0); known += 1
                if rt is not None and not pd.isna(rt): rat.append(rt)
                if f5 is not None and not pd.isna(f5) and rt is not None and not pd.isna(rt): frm.append(f5 - rt)
                mv += (s.get("mv") or 0) / 1e6
            rows.append({"match_id": r.match_id, "side": side, "xi_att": att * 11 / max(known, 1), "xi_known": known,
                         "xi_rating": np.mean(rat) if rat else np.nan, "xi_form": np.mean(frm) if frm else 0.0, "xi_mv": mv,
                         "xi_rest": np.nanmean([x[0] for x in fat]) if any(not np.isnan(x[0]) for x in fat) else np.nan,
                         "xi_min14": np.mean([x[1] for x in fat]), "xi_games10": np.mean([x[2] for x in fat]),
                         "xi_short_rest": float(np.mean([x[0] <= 3.5 for x in fat if not np.isnan(x[0])] or [0]))})
    return pd.DataFrame(rows)


def team_view(M, T, L, refs):
    """One row per (match, side) with the base rate, own style (s_), opponent style (o_s_), lineup and referee features."""
    sty = T[["match_id", "venue"] + [c for c in T.columns if c.startswith("s_")]]
    # upcoming matches have no team_match row: use each team's state after its latest match (cur_ columns)
    last = T.sort_values("date").groupby("team").tail(1).set_index("team")
    cur = [c for c in T.columns if c.startswith("cur_")]
    up = []
    for r in M[~M.finished].itertuples():
        for side in ("home", "away"):
            t = getattr(r, side)
            if t in last.index:
                up.append({"match_id": r.match_id, "venue": side, **{"s_" + c[4:]: last.at[t, c] for c in cur}})
    if up: sty = pd.concat([sty, pd.DataFrame(up)[sty.columns.intersection(pd.DataFrame(up).columns)]], ignore_index=True)
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        v = M[["match_id", "season", "date", "finished", "referee"]].copy()
        v["side"] = side
        v["is_home"] = 1.0 if side == "home" else 0.0
        v["lam_base"] = M["lam_h_base" if side == "home" else "lam_a_base"]
        v["team"], v["opp"] = M[side], M[opp]
        v["goals"] = M["hg" if side == "home" else "ag"]
        out.append(v)
    V = pd.concat(out, ignore_index=True)
    V = V.merge(sty.rename(columns={"venue": "side"}), on=["match_id", "side"], how="left")
    o = sty.rename(columns={"venue": "side", **{c: "o_" + c for c in sty.columns if c.startswith("s_")}})
    o["side"] = o["side"].map({"home": "away", "away": "home"})
    V = V.merge(o, on=["match_id", "side"], how="left")
    V = V.merge(L, on=["match_id", "side"], how="left")
    Lo = L.rename(columns={c: "o_" + c for c in L.columns if c.startswith("xi_")}); Lo["side"] = Lo["side"].map({"home": "away", "away": "home"})
    V = V.merge(Lo, on=["match_id", "side"], how="left")
    V = V.merge(refs, on="match_id", how="left")
    # lineup attack relative to the team's own recent XIs (EW, pre-match)
    V = V.sort_values("date")
    V["xi_att_rel"] = np.log((V["xi_att"] + 0.05) / (V.groupby("team")["xi_att"].transform(lambda s: s.shift(1).ewm(halflife=6).mean()) + 0.05))
    V["xi_rating_rel"] = V["xi_rating"] - V.groupby("team")["xi_rating"].transform(lambda s: s.shift(1).ewm(halflife=6).mean())
    V["o_xi_rating_rel"] = V["o_xi_rating"] - V.groupby("opp")["o_xi_rating"].transform(lambda s: s.shift(1).ewm(halflife=6).mean())
    return V.sort_values(["date", "match_id", "side"]).reset_index(drop=True)


def referee_table(M, T):
    """Pre-match EW referee card/foul tendency, relative to league average."""
    tm = T.groupby("match_id").agg(yc=("yc", "sum"), fouls=("fouls", "sum")).reset_index()
    R = M[["match_id", "date", "referee"]].merge(tm, on="match_id", how="left").sort_values("date")
    lg_yc, lg_f = tm.yc.mean(), tm.fouls.mean()
    g = R.groupby("referee")
    R["ref_n"] = g.cumcount()
    k = 6  # shrink with 6 pseudo-matches at league average
    for c, lg in (("yc", lg_yc), ("fouls", lg_f)):
        s = g[c].transform(lambda x: x.shift(1).fillna(0).cumsum()); n = g[c].transform(lambda x: x.shift(1).notna().cumsum())
        R["ref_" + c] = (s + k * lg) / (n + k) / lg
    R.loc[R.referee.isna(), ["ref_yc", "ref_fouls"]] = 1.0
    return R[["match_id", "ref_n", "ref_yc", "ref_fouls"]]


# fatigue of both XIs (all competitions): helps the goal model in backtest; stat models are better without it
FATIGUE_FEATS = ["xi_rest", "xi_min14", "xi_short_rest", "o_xi_rest", "o_xi_min14", "o_xi_short_rest"]
GOAL_FEATS = ["xi_att_rel", "xi_rating_rel", "o_xi_rating_rel", "xi_form"] + \
             [f"s_{c}" for c in GOAL_STYLE] + [f"o_s_{c}" for c in GOAL_STYLE] + ["o_s_opp_npxg", "s_opp_npxg"] + FATIGUE_FEATS


class GoalModel:
    """log(goals) = log(lam_base) + w·z(features); fitted on past matches only."""

    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def _X(self, V):
        X = V[GOAL_FEATS].astype(float)
        return ((X - self.mu) / self.sd).fillna(0.0).clip(-4, 4)

    def fit(self, V):
        V = V[V.finished & V.goals.notna() & V.lam_base.notna()]
        self.mu, self.sd = V[GOAL_FEATS].mean(), V[GOAL_FEATS].std().replace(0, 1)
        y = V.goals.values / V.lam_base.values  # offset trick: model the ratio with sample weight lam_base
        self.m = PoissonRegressor(alpha=self.alpha / len(V) * 100, max_iter=500).fit(self._X(V), y, sample_weight=V.lam_base.values)
        return self

    def predict(self, V):
        return V.lam_base.values * self.m.predict(self._X(V))


class StatModel:
    def __init__(self, target, alpha=1.0):
        self.target, self.feats, self.alpha = target, STAT_TARGETS[target] + ["is_home", "lam_base", "o_lam_base"], alpha

    def _X(self, V):
        return ((V[self.feats].astype(float) - self.mu) / self.sd).fillna(0.0).clip(-4, 4)

    def fit(self, V, y):
        ok = y.notna()
        V, y = V[ok], y[ok]
        self.mu, self.sd = V[self.feats].mean(), V[self.feats].std().replace(0, 1)
        self.m = PoissonRegressor(alpha=self.alpha / len(V) * 100, max_iter=500).fit(self._X(V), y)
        resid_var = np.mean((y - self.m.predict(self._X(V))) ** 2)
        self.dispersion = max(resid_var / np.mean(self.m.predict(self._X(V))), 1.0)  # >1: overdispersed vs Poisson
        return self

    def predict(self, V):
        return self.m.predict(self._X(V))


def count_over(mean, line, dispersion=1.0):
    """P(X > line) with a negative binomial of the given dispersion (Poisson when dispersion=1)."""
    k = int(math.floor(line))
    if dispersion <= 1.0001:
        return 1 - poisson.cdf(k, mean)
    from scipy.stats import nbinom
    n = mean / (dispersion - 1); p = n / (n + mean)
    return 1 - nbinom.cdf(k, n, p)


def market_lambdas(p1, px, p2, ou_line=None, p_over=None):
    """Goal rates (home, away) whose Dixon-Coles probabilities best match the de-vigged 1X2 (and O/U if given)."""
    from scipy.optimize import minimize
    s = p1 + px + p2
    tgt = np.array([p1, px, p2]) / s
    ou = None
    if ou_line is not None and p_over is not None and not pd.isna(ou_line) and not pd.isna(p_over):
        ou = (float(ou_line), float(p_over))
    i = np.arange(MAXG + 1); tot = i[:, None] + i[None, :]
    def err(x):
        lh, la = np.exp(x)
        S = dixon_coles(lh, la)
        e = np.sum((np.array(list(outcome_probs(S).values())) - tgt) ** 2)
        if ou: e += (S[tot > ou[0]].sum() - ou[1]) ** 2
        return e
    r = minimize(err, np.log([1.4, 1.1]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9})
    return tuple(np.exp(r.x))

"""Pre-match features computed strictly from information available before kick-off (no leakage).

Team strength: exponentially-weighted, opponent-adjusted attack/defence ratings on a blend of goals, npxG and xGOT.
Style: exponentially-weighted per-match style metrics (possession, pressing, directness, crossing, set pieces, ...).
Player: exponentially-weighted per-90 rates (npxG, xA, shots, cards, fouls, FotMob rating) + recent form + minutes load.
"""
import math
import numpy as np
import pandas as pd

STYLE = ["poss", "ppda", "deep", "long_balls", "crosses", "aerials", "corners", "box_touches", "xg_set", "shot_xg_fastbreak",
         "shot_xg_head", "fouls", "yc", "tackles", "interceptions", "dribbles", "shots", "sot", "saves", "offsides"]
PLAYER_RATES = ["npxg", "xa", "sh", "sot", "kp", "box_touches", "yc", "fouls", "fouled", "def_actions", "tackles", "interceptions",
                "recoveries", "aerials", "dribbles", "saves", "conceded", "xgot_faced"]
SEASON_REGRESS = 0.70   # share of last season's rating kept over the summer (rest goes back to league mean)


class TeamRatings:
    """Sequential Elo-like update of log attack/defence on a target goal measure; all updates use only past matches."""

    def __init__(self, half_life=14, w_goals=0.3, w_xg=0.5, w_xgot=0.2, home_adv=0.10, promoted_att=-0.18, promoted_def=0.18):
        self.k = 1 - 0.5 ** (1 / half_life)
        self.w = (w_goals, w_xg, w_xgot)
        self.home_adv = home_adv
        self.promoted = (promoted_att, promoted_def)
        self.att, self.dfn, self.n = {}, {}, {}
        self.mean_goals = 1.35
        self.season = None

    def new_season(self, season, teams):
        if self.season is not None:
            known = set(self.att)
            for t in list(self.att):
                self.att[t] *= SEASON_REGRESS; self.dfn[t] *= SEASON_REGRESS
            for t in teams:
                if t not in known:
                    self.att[t], self.dfn[t] = self.promoted
        self.season = season

    def expected(self, home, away):
        for t in (home, away):
            if t not in self.att:
                self.att[t], self.dfn[t] = self.promoted if self.season is not None and self.n else (0.0, 0.0)
                self.n[t] = 0
        lh = self.mean_goals * math.exp(self.att[home] + self.dfn[away] + self.home_adv)
        la = self.mean_goals * math.exp(self.att[away] + self.dfn[home] - self.home_adv)
        return lh, la

    def target(self, g, xg, xgot):
        parts = [(self.w[0], g), (self.w[1], xg), (self.w[2], xgot)]
        parts = [(w, v) for w, v in parts if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(w * v for w, v in parts) / sum(w for w, _ in parts)

    def update(self, home, away, th, ta):
        lh, la = self.expected(home, away)
        k = self.k
        eh, ea = math.log((th + 0.1) / (lh + 0.1)), math.log((ta + 0.1) / (la + 0.1))
        self.att[home] += k * eh; self.dfn[away] += k * eh
        self.att[away] += k * ea; self.dfn[home] += k * ea
        self.mean_goals += 0.002 * ((th + ta) / 2 - self.mean_goals)
        self.n[home] = self.n.get(home, 0) + 1; self.n[away] = self.n.get(away, 0) + 1


def build(D, half_life=14, style_hl=8, player_hl=10, **rating_kw):
    M, T, P = D["matches"].copy(), D["team_match"].copy(), D["player_match"].copy()
    M = M.sort_values("date").reset_index(drop=True)
    # ---- dynamic team ratings ----
    tr = TeamRatings(half_life=half_life, **rating_kw)
    tm = T.set_index(["match_id", "venue"])
    lam = []
    for r in M.itertuples():
        if r.season != tr.season:
            tr.new_season(r.season, set(M.loc[M.season == r.season, "home"]) | set(M.loc[M.season == r.season, "away"]))
        lh, la = tr.expected(r.home, r.away)
        lam.append((lh, la, tr.att[r.home], tr.dfn[r.home], tr.att[r.away], tr.dfn[r.away], tr.n.get(r.home, 0), tr.n.get(r.away, 0)))
        if r.finished and (r.match_id, "home") in tm.index:
            h, a = tm.loc[(r.match_id, "home")], tm.loc[(r.match_id, "away")]
            th = tr.target(r.hg, h.get("npxg"), h.get("xgot"))
            ta = tr.target(r.ag, a.get("npxg"), a.get("xgot"))
            tr.update(r.home, r.away, th, ta)
    M[["lam_h_base", "lam_a_base", "att_h", "def_h", "att_a", "def_a", "n_h", "n_a"]] = np.array(lam)
    # ---- team style: EW over each team's matches; "cur_" includes the match, "s_" is the pre-match value ----
    T = T.sort_values("date").reset_index(drop=True)
    cols = STYLE + ["npxg", "gf", "ga"] + [c for c in T.columns if c.startswith("opp_") and c[4:] in STYLE + ["npxg"]]
    cur = T.groupby("team", sort=False)[cols].transform(lambda s: s.ewm(halflife=style_hl, ignore_na=True).mean())
    cur.columns = ["cur_" + c for c in cols]
    pre = pd.concat([T[["team"]], cur], axis=1).groupby("team", sort=False)[list(cur.columns)].shift(1)
    pre.columns = ["s_" + c[4:] for c in cur.columns]
    T = pd.concat([T, cur, pre], axis=1)
    # ---- player per-90 rates and form, over ALL competitions (Serie A + cups): "cur_" includes the match, "pre_" is pre-match ----
    P = P.sort_values("date").reset_index(drop=True)
    P["min"] = P["min"].fillna(0).clip(lower=0)
    w = P["min"] / 90
    alpha = 1 - 0.5 ** (1 / player_hl)
    gp = P.groupby("player_id", sort=False)
    ew = lambda col: col.groupby(P["player_id"], sort=False).transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean())
    den_r = ew(P["rating"].notna() * w)
    for c in PLAYER_RATES:  # each rate only over the minutes where that stat was recorded
        P["cur_" + c] = ew(P[c].fillna(0) * w) / ew(P[c].notna() * w).replace(0, np.nan)
    P["cur_rating"] = ew(P["rating"].fillna(0) * w) / den_r.replace(0, np.nan)
    P["cur_form5"] = gp["rating"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    P["cur_start_rate5"] = gp["starter"].transform(lambda s: s.astype(float).rolling(5, min_periods=1).mean())
    P["cur_min_start"] = P["min"].where(P["starter"]).groupby(P["player_id"]).transform(lambda s: s.ewm(halflife=6, ignore_na=True).mean().ffill())
    P["cur_min_sub"] = P["min"].where(~P["starter"] & (P["min"] > 0)).groupby(P["player_id"]).transform(lambda s: s.ewm(halflife=6, ignore_na=True).mean().ffill())
    P["cur_apps"] = gp.cumcount() + 1
    P["cur_minutes"] = gp["min"].cumsum()
    cur = [c for c in P.columns if c.startswith("cur_")]
    pre = P.groupby("player_id", sort=False)[cur].shift(1)
    pre.columns = [c.replace("cur_", "pre_") for c in cur]
    P = pd.concat([P, pre], axis=1)
    P["last_date"] = gp["date"].shift(1)
    return M, T, P, tr

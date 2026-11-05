"""Player against player: the direct opponent of every outfield player and what the pairing says about the match.

Both starting elevens are laid on the same pitch (the away team mirrored, FotMob verticalLayout) and the outfield players are matched one to
one by minimum total distance (Hungarian assignment). A pair is a 'flank' duel when the player stands wide, else a 'centre' duel.
For every pair: an own statistic (recent per-90 rate before the match) times an opponent statistic; summed over the pairs of the team,
for all pairs / flank pairs / centre pairs. 16 x 12 x 3 = 576 columns named duel:<zone>:<own>*<opponent>.
Used by model.StatModel as a second, heavily penalised stage (src/study_duels2.py: yellow cards -0.7% of squared error, 5 seasons of 6).
For matches not played yet the players' latest state (cur_) replaces the pre-match one (pre_), as in model.lineup_strength."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

S1 = ["dribbles", "kp", "sh", "sot", "npxg", "xa", "crosses", "fouled", "box_touches", "final_third", "aerials", "dispossessed", "fouls", "tackles", "duels_won", "duels_lost"]
S2 = ["tackles", "interceptions", "duels_lost", "duels_won", "def_actions", "fouls", "aerials", "clearances", "blocks", "recoveries", "dribbles", "yc"]
ALLS = sorted(set(S1) | set(S2))
FLANK = 0.2                                     # |x - 0.5| above this = wide player
ZONES = (("all", 6), ("flank", 2), ("centre", 2))   # zone, minimum number of pairs with both statistics known
NAMES = [f"{a}*{b}" for a in S1 for b in S2]
COLUMNS = [f"duel:{z}:{n}" for z, _ in ZONES for n in NAMES]


def pairs_of(lineup):
    L = {s: [p for p in lineup[s]["starters"] if p.get("pos") != 11 and p.get("x") is not None] for s in ("home", "away")}
    if min(len(L["home"]), len(L["away"])) < 8: return None
    A = np.array([[p["x"], p["y"]] for p in L["home"]]); B = np.array([[1 - p["x"], 1 - p["y"]] for p in L["away"]])
    ia, ib = linear_sum_assignment(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2))
    return [(L["home"][i], L["away"][j], abs(A[i, 0] - 0.5) > FLANK) for i, j in zip(ia, ib)]


def add(V, M, P):
    """V (one row per match and side) with the 576 duel columns merged on (match_id, side); rows without a usable lineup get NaN."""
    if COLUMNS[0] in V.columns: return V
    rate = P[["pre_" + c for c in ALLS]].to_numpy(float)
    row = {(m, p): i for i, (m, p) in enumerate(zip(P.match_id.values, P.player_id.values))}
    last = P.sort_values("date").groupby("player_id").tail(1).set_index("player_id")[["cur_" + c for c in ALLS]]
    lastv = dict(zip(last.index, last.to_numpy(float)))
    col = {c: i for i, c in enumerate(ALLS)}; i1 = [col[c] for c in S1]; i2 = [col[c] for c in S2]

    def vec(mid, pid, idx, finished):
        k = row.get((mid, pid))
        if k is not None: return rate[k, idx]
        if not finished and pid in lastv: return lastv[pid][idx]
        return np.full(len(idx), np.nan)

    out = []
    for r in M.itertuples():
        pr = pairs_of(r.lineup)
        if pr is None: continue
        for side in ("home", "away"):
            acc = {z: np.zeros(len(NAMES)) for z, _ in ZONES}; cnt = {z: np.zeros(len(NAMES)) for z, _ in ZONES}
            for ph, pa, wide in pr:
                own, opp = (ph, pa) if side == "home" else (pa, ph)
                prod = np.outer(vec(r.match_id, own["id"], i1, r.finished), vec(r.match_id, opp["id"], i2, r.finished)).ravel()
                ok = ~np.isnan(prod)
                for z in ("all", "flank" if wide else "centre"):
                    acc[z][ok] += prod[ok]; cnt[z][ok] += 1
            vals = np.concatenate([np.where(cnt[z] >= need, acc[z], np.nan) for z, need in ZONES])
            out.append(np.concatenate([[r.match_id, 0 if side == "home" else 1], vals]))
    D = pd.DataFrame(np.array(out), columns=["match_id", "side_i"] + COLUMNS)
    D["match_id"] = D.match_id.astype(M.match_id.dtype); D["side"] = np.where(D.side_i == 0, "home", "away"); D = D.drop(columns="side_i")
    return V.merge(D, on=["match_id", "side"], how="left")

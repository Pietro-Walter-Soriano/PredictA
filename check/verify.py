"""Independent consistency checks on every forecast JSON (throwaway review aid)."""
import glob, json, math, sys
from scipy.stats import poisson
fails = []
def ok(c, m):
    if not c: fails.append(m)
for f in sorted(glob.glob(sys.argv[1] + "/2*.json")):
    d = json.load(open(f)); n = f.split("/")[-1]; lh, la = d["lambda"]["home"], d["lambda"]["away"]
    p = d["p"]; ok(abs(p["1"] + p["X"] + p["2"] - 1) < 2e-3, f"{n} 1X2 sum")
    ok(abs(sum(d["markets"]["goals_dist"]) - 1) < 3e-3, f"{n} goals dist sum")
    eg = sum(k * v for k, v in enumerate(d["markets"]["goals_dist"][:6])) + 6.5 * d["markets"]["goals_dist"][6]
    ok(abs(eg - lh - la) < 0.2, f"{n} E[goals] {eg:.2f} vs {lh+la:.2f}")
    ok(abs(d["markets"]["cs"]["home"] - math.exp(-la)) < 0.03, f"{n} clean sheet")
    ok(abs(d["markets"]["btts"] - (1 - math.exp(-lh)) * (1 - math.exp(-la))) < 0.03, f"{n} btts")
    if d["lambda_market"]:  # final rate = geometric blend of model and market
        w = d["model_weight"]
        ok(abs(lh - d["lambda_model"]["home"] ** w * d["lambda_market"]["home"] ** (1 - w)) < 2e-3, f"{n} blend home")
    for side, lam in (("home", lh), ("away", la)):
        pl = [x for x in d["teams"][side]["players"] if x["props"]]
        ok(abs(sum(x["props"]["exp_goals"] for x in pl) - lam) < 0.02, f"{n} {side} scorer sum {sum(x['props']['exp_goals'] for x in pl):.3f} vs {lam}")
        ok(abs(sum(x["exp_min"] for x in d["teams"][side]["players"]) - 990) <= 12, f"{n} {side} minutes")
        ok(sum(x["starter"] for x in d["teams"][side]["players"]) == 11, f"{n} {side} XI size")
    c = d["lines"]["corners"]
    if c["disp"] <= 1.0001: ok(abs(c["over"]["9.5"] - (1 - poisson.cdf(9, c["mean"]))) < 1e-3, f"{n} corners poisson")
    else: ok(c["over"]["9.5"] < 1 and c["over"]["7.5"] > c["over"]["11.5"], f"{n} corners monotone")
print("FAIL", fails) if fails else print("PASS", len(glob.glob(sys.argv[1] + "/2*.json")), "files")

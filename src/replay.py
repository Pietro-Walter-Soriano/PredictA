"""Rebuild the archived forecasts of every finished round of the current season with the current engine.
Each round is forecast as of 2 hours before its first kick-off, hiding everything that happened later.
Used after a change of model or data source, so that round grades stay comparable."""
import os, shutil, subprocess, sys
import pandas as pd

DATA = sys.argv[1]
PY = sys.executable
M = pd.read_pickle(f"{DATA}/dataset.pkl")["matches"]
season = M.season.max()
cur = M[M.season == season]
tmp = f"{DATA}/.replay"
for rnd, g in cur.groupby("round"):
    if not g.finished.all(): continue
    as_of = (g.date.min() - pd.Timedelta(hours=2)).isoformat()
    shutil.rmtree(f"{DATA}/ledger/{season}/R{int(rnd):02d}", ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    env = {**os.environ, "AS_OF": as_of, "ONLY_ROUND": str(int(rnd))}
    subprocess.run([PY, "-I", "src/predict.py", DATA, tmp], env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, "-I", "check/verify.py", tmp], check=True)
    subprocess.run([PY, "-I", "src/ledger.py", tmp, f"{DATA}/ledger"], check=True)
    print("replayed round", int(rnd), "as of", as_of, flush=True)
shutil.rmtree(tmp, ignore_errors=True)

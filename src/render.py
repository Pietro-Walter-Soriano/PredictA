"""Embed index, all match JSONs and the round report cards into a single self-contained HTML (opens from file://)."""
import glob, json, os, sys
tpl, data_dir, out, rounds = sys.argv[1:5]
idx = json.load(open(f"{data_dir}/index.json"))
matches = {m["file"]: json.load(open(f"{data_dir}/{m['file']}.json")) for m in idx["matches"]}
safe = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
open(out, "w").write(open(tpl).read().replace("__INDEX__", safe(idx)).replace("__MATCHES__", safe(matches)).replace("__ROUNDS__", safe(json.load(open(rounds)) if os.path.exists(rounds) else {"rounds": [], "total": None})))
print(out, os.path.getsize(out) // 1024, "KB")

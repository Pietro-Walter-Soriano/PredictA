"""Write REPORT.md: grade and KPIs per round + upcoming forecasts, readable straight on GitHub."""
import json, os, sys
from datetime import datetime

ROUNDS, INDEX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(ROUNDS)) if os.path.exists(ROUNDS) else {"rounds": [], "total": None}
I = json.load(open(INDEX)) if os.path.exists(INDEX) else {"matches": [], "generated": None}
pct = lambda x: "–" if x is None else f"{x * 100:.0f}%"
num = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
COMP = ["1X2 vs storico", "1X2 vs bookmaker", "Mercati gol", "Linee statistiche", "Marcatori"]

out = ["# PredictA — Serie A", "", f"Aggiornato: {(I.get('generated') or '')[:16].replace('T', ' ')} UTC · "
       "dashboard completa: https://pietro-walter-soriano.github.io/PredictA/ (oppure *Actions → ultimo run → Artifacts*).", ""]
t = R.get("total")
if t:
    out += [f"## Voto complessivo: **{R['total_grade']}/100** su {t['n']} partite", "",
            "| " + " | ".join(COMP) + " |", "|" + "---|" * len(COMP),
            "| " + " | ".join(str(R["total_components"].get(c, "–")) for c in COMP) + " |", "",
            f"Favorito indovinato {pct(t['fav_hit'])} · log-loss 1X2 {num(t['logloss'])} (storico {num(t['baseline_logloss'])}"
            + (f", bookmaker {num(t['market']['logloss'])}" if t.get("market") else "") + f") · risultato esatto {pct(t['top1_hit'])} "
            f"(top-3 {pct(t['top3_hit'])}) · scommesse di valore {t['value']['won']}/{t['value']['bets']}, P&L {t['value']['pnl']:+.2f} u", "",
            "50 = come il riferimento ingenuo, 75 = errore più basso del 10%. Dettagli nel README.", "",
            "## Giornate", "", "| Giornata | Voto | " + " | ".join(COMP) + " | Favorito | Log-loss | Value P&L | Tipo |",
            "|---|---|" + "---|" * len(COMP) + "---|---|---|---|"]
    for r in reversed(R["rounds"]):
        f = r["final"]
        out.append(f"| {r['season']} R{r['round']} | **{r['grade']}** | " + " | ".join(str(r["components"].get(c, "–")) for c in COMP)
                   + f" | {pct(f['fav_hit'])} | {num(f['logloss'])} | {f['value']['pnl']:+.2f} | {'ricostruita' if r['replay'] else 'live'}"
                   + ("" if r["complete"] else " (in corso)") + " |")
    out.append("")
if I.get("matches"):
    out += ["## Prossime partite", "", "| Data (UTC) | Partita | 1 | X | 2 | Gol attesi | Over 2.5 | Gol entrambe | Più probabile | Formazioni |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m in I["matches"]:
        out.append(f"| {m['date'][:16].replace('T', ' ')} | {m['home']} – {m['away']} | {pct(m['p']['1'])} | {pct(m['p']['X'])} | {pct(m['p']['2'])} | "
                   f"{m['lambda']['home']:.2f}–{m['lambda']['away']:.2f} | {pct(m['o25'])} | {pct(m['btts'])} | {m['top']['s']} ({pct(m['top']['p'])}) | "
                   f"{'ufficiali' if m['lineup_type'] == 'standard' else 'probabili'} |")
    out.append("")
out += ["---", "Strumento di analisi statistica, non un consiglio di scommessa."]
open(OUT, "w").write("\n".join(out) + "\n")
print("report", OUT)

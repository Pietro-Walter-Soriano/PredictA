# Serie A forecast

Previsioni per ogni partita di Serie A (1X2, risultato esatto, gol, angoli, cartellini, tiri, marcatori) partendo dalle
formazioni, e una pagella oggettiva per ogni giornata conclusa. Gira in locale, nessun servizio a pagamento.

## Installazione (Linux, macOS o Windows con WSL)

Serve Python 3.10 o superiore.

```bash
git clone <repo> calcio-forecast && cd calcio-forecast
./setup.sh          # crea .venv, installa le dipendenze, crea .env da .env.example
./run.sh            # primo giro: scarica l'archivio 2020-21 -> oggi (~1 ora, ~80 MB), poi prevede e valuta
```

Apri `site/serie-a.html` nel browser: scheda **Prossime partite** e scheda **Pagelle giornate**.

## Chiavi e fonti opzionali (`.env`)

| Variabile | Cosa abilita |
|---|---|
| `ODDS_API_KEY` | quote pre-partita (Pinnacle quando c'è) da the-odds-api.com, 1 richiesta per giro |
| `PUBLISH_DIR` | copia della dashboard in un'altra cartella dopo ogni giro |

Le quote storiche di football-data.co.uk (Pinnacle, media di mercato, Bet365) si scaricano da sole. Se il sito non è
raggiungibile, metti a mano i CSV in `data/manual/` con il nome `I1_2021.csv`, `I1_2122.csv`, ... `I1_2627.csv`
(da `https://www.football-data.co.uk/mmz4281/<codice>/I1.csv`).

Priorità delle quote: Pinnacle chiusura > media di mercato > Bet365 > The Odds API > ESPN/DraftKings.

## Esecuzione automatica su GitHub (attiva)

`.github/workflows/forecast.yml` si sveglia ogni 10 minuti e `src/plan.py` decide se c'è qualcosa da fare (GitHub può
ritardare i cron di qualche minuto: il piano ragiona sull'orologio reale, non sul tick):

- **50 minuti prima** del calcio d'inizio, se FotMob ha già la formazione ufficiale (11 titolari per squadra): previsione rifatta con quella;
- **30 minuti prima**, se la formazione ufficiale non c'era: previsione rifatta con la formazione probabile;
- **partita finita**: scarica i dati finali, calcola il risultato, valuta la previsione e inserisce i KPI nella scheda della giornata
  (*Pagelle giornate*, giornata "in corso n/N"), partita per partita; dopo ~20 minuti un secondo passaggio, perché FotMob completa xG, tiri
  e statistiche dei giocatori un po' dopo il fischio finale;
- altrimenti un giro completo (con backtest) circa ogni ora, come prima.

I giri a tempo (previsione, risultato) saltano il backtest (`FAST=true` in `run.sh`) e usano `data/calibration.json` già committato;
la richiesta a The Odds API parte solo quando si rifà una previsione. Lo stato del piano è in `data/schedule.json`.
Ogni giro committa archivio, pagelle e [`REPORT.md`](REPORT.md) e aggiorna la dashboard su GitHub Pages (anche negli *Artifacts*).
Run manuale: *Actions → forecast → Run workflow* (`force` = giro completo comunque). Per le quote The Odds API aggiungi il secret
`ODDS_API_KEY` (*Settings → Secrets and variables → Actions*). Una sola macchina deve scrivere sul repository: se usi GitHub Actions,
in locale lancia `run.sh` solo per guardare, senza fare push.

## Esecuzione automatica in locale (alternativa)

`run.sh` si può lanciare quando vuoi e si ferma da solo se è già in corso. Ogni ora va bene:

- Linux/macOS, `crontab -e`: `7 9-23 * * * /percorso/calcio-forecast/run.sh >> /percorso/calcio-forecast/data/run.log 2>&1`
- Windows: Utilità di pianificazione che lancia `wsl.exe -- bash -lc "~/calcio-forecast/run.sh >> ~/calcio-forecast/data/run.log 2>&1"`

A ogni giro: aggiorna i dati, riaddestra, prevede le partite dei prossimi 10 giorni, archivia le previsioni fatte prima
del calcio d'inizio (`data/ledger/`), valuta le giornate concluse (`data/rounds.json`) e ricostruisce la dashboard.

## Come è valutato (voto 0-100)

Media di 5 componenti; ognuna confronta l'errore delle nostre probabilità con quello di un riferimento ingenuo
(50 = uguale, 75 = errore più basso del 10%):

| Componente | Riferimento |
|---|---|
| 1X2 vs storico | frequenze 1 / X / 2 delle stagioni passate |
| 1X2 vs bookmaker | quote di chiusura senza margine, sulle stesse partite |
| Mercati gol | frequenza storica di Over 1.5/2.5/3.5, gol entrambe, porte inviolate |
| Linee statistiche | frequenza storica su linee fisse: angoli 9.5, gialli 4.5, tiri in porta 8.5, tiri 24.5, falli 24.5 |
| Marcatori | gol della squadra divisi per minuti attesi dei giocatori |

Oltre al voto: favorito indovinato, log-loss, RPS, risultato esatto, calibrazione, XI indovinato, simulazione di
puntate a quota equa dove il modello supera il bookmaker di almeno 3 punti.

## Struttura

| File | Ruolo |
|---|---|
| `src/fetch_fotmob.py` | FotMob: Serie A dal 2020-21 (statistiche squadra e giocatore, voti, tiri, formazioni, arbitro, meteo) + coppe dei club italiani |
| `src/fetch_aux.py` | Understat (PPDA, passaggi profondi), ESPN (calendario, quote DraftKings) |
| `src/fetch_odds.py` | football-data.co.uk, The Odds API |
| `src/dataset.py` | normalizza tutto in `data/dataset.pkl` (partite, squadra-partita, giocatore-partita) |
| `src/features.py` | rating dinamici, stile di gioco, stato dei giocatori su tutte le competizioni; solo dati pre-partita |
| `src/model.py` | modello gol (rating + stile + XI + stanchezza), modelli per le statistiche, quote -> gol attesi |
| `src/backtest.py` | walk-forward dal 2021-22; calibra il peso modello/mercato (`data/calibration.json`) |
| `src/tune.py` | ottimizzazione degli iperparametri sulle stagioni passate (da lanciare a mano) |
| `src/predict.py` | previsioni per partita (`site/data/*.json`) |
| `src/ledger.py`, `src/evaluate.py` | archivio delle previsioni (`data/ledger/<stagione>/Rxx/<id>_first|final.json`) e pagelle |
| `src/plan.py` | pianificatore dello scheduler: T-50 / T-30 / fine partita / giro orario (`data/schedule.json`) |
| `check/verify.py` | controlli di coerenza sulle previsioni |

Ricostruire una giornata passata con i soli dati disponibili prima (formazioni ufficiali comprese):

```bash
AS_OF=2026-10-16T16:45:00 ONLY_ROUND=7 .venv/bin/python -I src/predict.py data /tmp/replay
.venv/bin/python -I src/ledger.py /tmp/replay data/ledger
```

## Avvertenza

Strumento di analisi statistica. Non è un consiglio di scommessa.

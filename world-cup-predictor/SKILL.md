---
name: world-cup-predictor
description: Statistical match predictions for the 2026 FIFA World Cup (and any international football match). Builds Elo ratings from 150 years of results, converts them into win/draw/loss probabilities and likely scorelines, layers on current squad news, logs every prediction, and tracks accuracy against real results. Use this skill whenever the user asks who will win a World Cup match, wants match predictions or odds, asks "who's going to win the World Cup", mentions simulating the tournament, wants to check how past predictions did, or asks about team strength ratings — even if they don't say the word "predict".
---

# World Cup Predictor

Predict international football matches from statistics, not vibes. The core is
a two-model ensemble built from every men's international since 1872 (~49,000
matches): an Elo rating system and an independent Maher attack/defence Poisson
model, blended at equal weight (validated walk-forward on WC 1998–2022;
the blend beats either model alone on total Brier). When the two models
disagree by more than 10 points the prediction is flagged. Bookmaker odds,
when recorded, are scored as a benchmark the model must beat — never as an
input. On top of the statistical baseline sits a qualitative layer
(injuries, suspensions, form) that you research and apply with documented
reasoning. Every prediction is logged; every real result is fed back in.

All scripts are plain Python 3 (stdlib only). Run them from the project
directory (the user's working directory, e.g. `/workspace/WorldCup`), so that
`data/` and `predictions/` accumulate there across sessions — they are the
system's memory. Refer to scripts by absolute path under this skill's
`scripts/` directory.

## Session workflow (run every match day)

Work through these steps in order. Steps 1–3 are the feedback loop the whole
system depends on — skipping them means predicting from stale ratings and
never learning from mistakes.

### 1. Refresh the data — every run, no exceptions

```bash
python3 <skill>/scripts/fetch_data.py
```

The dataset routinely lags 1–2 days behind reality, which during a tournament
means it is missing exactly the matches that matter most. So after fetching,
check for completed-but-unscored fixtures (mind the user's timezone — a match
"today" in the data may already be finished where the user is):

```bash
awk -F, -v today=$(date +%F) '$4=="NA" && $1<=today && $6=="FIFA World Cup"' data/results.csv
```

For every fixture that has actually finished, web search the final score
(e.g. "Mexico South Africa World Cup final score June 11 2026") and append a
row to `data/manual_results.csv` — exact same columns and team names as
`data/results.csv` (e.g. "United States", "South Korea"; check the file if
unsure). The dedupe logic handles overlap once the upstream dataset catches
up. Never invent a score from memory: if the search doesn't confirm it,
leave the fixture unresolved and tell the user.

### 2. Rebuild ratings — both models

```bash
python3 <skill>/scripts/elo.py
python3 <skill>/scripts/maher.py
```

Takes seconds. `elo.py` recomputes every team's Elo from full history and
refits the Elo→goals calibration. `maher.py` refits the attack/defence
strengths (time-decayed, shrunk toward the mean for low-data teams) that form
the other half of the ensemble. This is how yesterday's results change
today's predictions. `maher.py` preserves the stored ensemble blend weight.

### 3. Resolve past predictions

```bash
python3 <skill>/scripts/record.py
```

Matches logged predictions against real results, computes outcome accuracy
and Brier scores. Read the output: if accuracy is drifting or specific kinds
of predictions keep missing (e.g. underdogs in knockouts, or matches where
you applied a qualitative adjustment), say so to the user and factor it into
how aggressively you adjust in step 5. The `[adjusted]` tag in the report
shows whether your qualitative adjustments are helping or hurting.

### 4. Find today's fixtures

Upcoming fixtures are already in `data/results.csv` with `NA` scores —
World Cup matches appear with `tournament == "FIFA World Cup"`:

```bash
grep ",NA,NA," data/results.csv | grep "FIFA World Cup" | head -20
```

Confirm against a quick web search if the user cares about exact kickoff
times or if a fixture looks missing.

### 5. Predict each match

Statistical baseline first:

```bash
python3 <skill>/scripts/predict.py "Brazil" "Morocco" --neutral --stage "Group C" --date 2026-06-13
```

Flags that matter:
- `--neutral` for matches where neither side is a host. Hosts (United States,
  Mexico, Canada) playing in their own country get home advantage — pass them
  as the first (home) team without `--neutral`. The `neutral` column in
  `results.csv` tells you which case you're in.
- `--knockout` from the round of 32 onward — adds advance probabilities
  (extra time/penalties modelled).

Useful extras on the same command:
- `--city "Mexico City"` — flags high-altitude venues (visiting lowland teams
  historically suffer there). The fixture row in `results.csv` has the city.
- Output automatically includes head-to-head record and each side's rest days.

Then the player layer. Web search for each side: confirmed/probable lineups,
injuries and suspensions, recent form and morale. For a *confirmed* absence
of a player, quantify it from data instead of guessing:

```bash
python3 <skill>/scripts/players.py "France"                      # scoring profile, top-3 reliance
python3 <skill>/scripts/players.py "France" --player "Mbappé"    # one player's goal share
```

then apply it directly — the player's recent goal share is converted into an
expected-goals discount (a replacement covers ~60% of the loss):

```bash
python3 <skill>/scripts/predict.py "France" "Senegal" --neutral \
  --missing-home "Mbappé" --date 2026-06-14 --log
```

This only covers scoring. For absences that matter through defence or
creation (keeper, centre-back, playmaker), or for non-player news (internal
crisis, key tactical change), use the bounded manual adjustment instead:

```bash
python3 <skill>/scripts/predict.py "Brazil" "Morocco" --neutral --stage "Group C" \
  --date 2026-06-13 --adjust-home -4 \
  --adjust-reason "Starting CB suspended (2nd yellow), backup is uncapped" \
  --log
```

Manual adjustments are percentage points on win probability, capped at ±10,
reason required — it's logged so step 3 can tell you whether your adjustments
beat the raw model. A missing rotation player ~1–2 points; a missing
world-class keeper 3–6; a genuine crisis up to 10. Decide using what Elo
*can't* see: recent results are priced in, yesterday's training injury is
not. No solid news → no adjustment; log the baseline run.

Log exactly one final prediction per match (only the last `--log` run, not
the baseline-then-adjusted pair).

The output now also includes the head-to-head record between the two teams —
use it as colour in the report, not as a probability input (Elo already
encodes those results).

**Market benchmark** (do it whenever odds are findable — it is the model's
scoreboard): while searching for team news, web search current decimal odds
for the match (e.g. "Brazil Morocco odds") and pass them on the logged run:

```bash
python3 <skill>/scripts/predict.py "Brazil" "Morocco" --neutral \
  --market "2.05,3.40,3.90" --date 2026-06-13 --log
```

The odds are de-vigged and stored alongside the model's probabilities;
`record.py` then scores model vs market vs a 50/50 blend on every resolved
match, and the dashboard shows the running comparison. The market is a
benchmark, never an input: `predicted_outcome` stays model-only. If the
market disagrees with the model by more than ~10 points on an outcome, say
so in the report and try to explain the gap (markets price lineup news and
public sentiment; the model prices long-run results).

The prediction output also shows both ensemble components (Elo vs Maher).
A "models disagree" flag means the two independent methods read the fixture
differently — present the prediction with extra caution and say why if the
news research suggests a reason.

### 6. Report to the user

For each match, present:

```
## <Home> vs <Away> — <stage>, <date>
**Prediction: <most likely outcome> (<prob>)** — most likely score <X-Y>
| | <Home> | Draw | <Away> |
|Probability| .. | .. | .. |
Elo: <home> (<rating>) vs <away> (<rating>), gap <+N> incl. advantage
Key factors: <2-4 bullets: Elo gap, host advantage, injuries/news found,
              relevant head-to-head or tournament history>
Adjustment: <what and why, or "none — no material squad news">
```

Also show the running accuracy from step 3 ("Model record so far: 9/13
outcomes, Brier 0.41") so the user always knows how much to trust it.

### 7. Regenerate the dashboard

```bash
python3 <skill>/scripts/dashboard.py
```

Writes `predictions/dashboard.html` — a self-contained page with upcoming
predictions (probability bars), resolved predictions vs reality, the model's
calibration curve, championship odds, and the Elo top 20 with movement since
the last run. Always regenerate it at the end of a session (it shows stale
data otherwise) and give the user the absolute file path to open.

## Tournament odds (on request or at stage transitions)

```bash
python3 <skill>/scripts/simulate.py -n 50000
```

Simulates the remaining tournament 50,000 times from the current state:
played group matches keep real scores, the rest are sampled from the same
Elo+Maher ensemble the match predictions use. Groups are auto-derived from
the fixture list. Knockout pairing is
approximated by performance seeding (see the script docstring); say so when
presenting champion odds. Worth re-running after each matchday completes —
odds shifting is the story users want.

## Standalone mode (no AI needed)

The entire statistical pipeline runs without Claude:

```bash
python3 <skill>/scripts/run_all.py --days 2 --simulate
```

fetch → Elo → Maher → resolve → predict & log upcoming fixtures → simulate →
dashboard, all in one command; cron-able for daily automation.

`start.bat` (Windows) / `start.sh` (Mac/Linux) in the project root run the
pipeline once and then serve the dashboard via `scripts/serve.py`, whose
POST /rerun endpoint backs the dashboard's "Re-run predictions" button. The
server tries port 8000 first and walks a fallback list (8080, 8765, 8888,
9090, then OS-assigned) because Windows frequently reserves 8000 for Hyper-V
(WinError 10013); it prints and opens whichever URL it bound. The dashboard also displays the exact prompt to
paste into Claude for a full update (news, injuries, lineups) — if a user
pastes something like it, run the complete session workflow above. If the user
asks "do I need Claude to run this", explain the split: scripts produce the
statistical baseline by themselves; a Claude session adds the things that
need judgment and live information — filling the dataset's 1–2 day lag from
news, injury/lineup research, market comparison, and the narrative report.

## Backtesting (after any model change)

```bash
python3 <skill>/scripts/backtest.py
```

Replays the 2014/2018/2022 World Cups walk-forward (predict each match from
ratings as they stood, then update). Reference performance (192 matches):
ensemble Brier 0.5800 vs Elo-only 0.5826 vs Maher-only 0.5839 vs 0.667 for
guessing; ~57% outcome accuracy. If a change makes these worse, revert it.
Quote these numbers when the user asks "how good is this model, really?"

- `--models` — side-by-side Elo vs Maher vs ensemble table per tournament.
- `--tune` — grid-search home advantage, K scale, Dixon-Coles ρ.
- `--tune-maher` — grid-search the Maher decay/prior/friendly-weight knobs.
- `--tune-blend [--write]` — re-derive the ensemble weight; `--write` stores
  it in `data/maher_ratings.json`.

All tuning trains on WC 1998–2018 and validates on held-out 2022. Adopt new
values only if train AND test both improve; a train win with a test loss is
noise. Last full run: decay was weakly identified (per-year winners split),
so the pre-registered default 0.20/yr stays; the blend weight is held at the
equal-weight 0.5 because the 1998–2022 panel could not reliably distinguish
weights in [0, 0.5] and equal weighting avoids tuning toward any one
tournament.

## First-time setup

If `data/elo_ratings.json` doesn't exist, run steps 1–2 and show the user the
top-15 Elo table plus calibration constants before predicting anything.

## When things go wrong

- **Team name not found**: `predict.py` suggests close matches. The dataset
  uses common English names ("South Korea", "United States", "DR Congo").
- **Dataset URL unreachable**: predictions still work from the existing
  `data/elo_ratings.json`; warn the user that ratings are as of its
  `latest_match` date.
- **Group derivation fails** in `simulate.py`: write `data/tournament.json`
  with `{"groups": [["Team1","Team2","Team3","Team4"], ...]}` (12 groups)
  from a web search of the official draw.

## Methodology details

Read `references/methodology.md` when the user asks *why* a prediction looks
the way it does, challenges the model, or wants the math (K factors, goal
multipliers, calibration, Brier scoring, known limitations).

## Scheduling

The skill itself runs on demand. If the user wants automatic match-day runs,
suggest `/schedule` (cloud, cron-based) or `/loop` (this session) wrapping a
prompt like "run the world cup predictor session workflow for today's
matches".

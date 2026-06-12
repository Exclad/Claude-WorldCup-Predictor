# World Cup Predictor

A data-driven statistical forecasting system for international football. Built for the 2026 FIFA World Cup — predicts match outcomes, simulates tournament brackets, tracks prediction accuracy, and serves an interactive dashboard.

No external dependencies. Pure Python 3 stdlib.

---

## Features

- **Dual-model ensemble**: Elo ratings + Maher Poisson model blended geometrically
- **Full scoreline matrix**: Dixon-Coles corrected Poisson distribution — not just win/draw/loss
- **Player absence quantification**: Data-derived xG discounts from 47,000+ goal records
- **Tournament simulation**: 50,000 Monte Carlo runs for championship/stage odds
- **Backtest validated**: Walk-forward on 2014/2018/2022 World Cups (0.580 Brier, ~57% accuracy)
- **Live accuracy tracking**: Every prediction logged and scored against actual results
- **Market comparison**: De-vigged bookmaker odds displayed alongside model probabilities
- **Interactive dashboard**: Self-contained HTML with upcoming predictions, model record, Elo rankings, championship odds
- **Local server**: `POST /rerun` endpoint to re-run pipeline from the browser

---

## How It Works

### Simple Version

Think of it like a sophisticated betting system built on historical data, not intuition.

Every international match since 1872 is used to assign each team a **rating** — a number that goes up when you beat strong teams and goes down when you lose to weak ones. The bigger the rating gap between two teams, the more likely the stronger team wins.

But one number isn't enough. A team can be strong overall but leak goals defensively. So a second model independently measures each team's **attack strength** and **defensive weakness**, then predicts how many goals each side is likely to score.

Both models produce expected goals (like "France 1.8 goals, Morocco 0.9 goals"). We combine them, then use probability math to turn those numbers into a full grid of every possible scoreline — 0-0, 1-0, 2-1, 3-2, etc. — each with a probability. Add up all the cells where France score more than Morocco, and that's France's win probability.

For knockout matches, if the game could go to penalties, we factor in each team's historical shootout record.

We track every prediction we make and score it against what actually happened. That keeps the model honest.

### Detailed Technical Version

#### Data Pipeline

The pipeline runs in sequence: fetch → elo → maher → record → predict → simulate → dashboard.

**Data sources** (`fetch_data.py`): Downloads three CSVs from `martj42/international_results` on GitHub:
- `results.csv` — 49,478 international matches since 1872
- `goalscorers.csv` — 47,000+ goal-by-goal records (scorer, minute, own goal, penalty flag)
- `shootouts.csv` — ~1,000 penalty shootout results

A `manual_results.csv` file handles the 1–2 day upstream lag during live tournaments.

---

#### Model 1: Elo Rating System (`elo.py`)

Sequential Bayesian-style update applied to every match in chronological order.

```
Expected result:  We = 1 / (1 + 10^(-dr/400))
                  dr = home_elo - away_elo + home_advantage (if not neutral)

Rating update:    R' = R + K × G × (W - We)
```

**K factors** (match importance):
| Tournament | K |
|---|---|
| FIFA World Cup | 60 |
| Continental finals (Euro, Copa América, AFCON, etc.) | 50 |
| Qualification | 40 |
| Other tournaments | 30 |
| Friendlies | 20 |

**Goal margin multiplier G**:
- 0–1 goal difference: 1.0
- 2 goals: 1.5
- 3+ goals: (11 + goal_diff) / 8

**Home advantage**: +100 Elo points added to `dr` for non-neutral venues.

**Inactivity decay**: Teams inactive 4+ years regress 20% toward 1500 (squad turnover proxy).

**Calibration constants** (fitted by least-squares on post-1990 data at every run):
- `goal_diff_per_elo` — slope of actual goal difference vs Elo gap (~0.00545)
- `base_goals` — average total goals in near-equal matches (~2.6)
- `knockout_goals_factor` — measured from WC 1986+, knockout matches average ~3% more goals

---

#### Model 2: Maher Attack/Defence Poisson Model (`maher.py`)

Fits independent multiplicative factors per team. Based on Dixon & Coles (1997) / Maher (1982).

```
λ_home = μ × att_home × def_away × γ
λ_away = μ × att_away × def_home

where:
  μ = average goals per team per match (~2.5 / 2)
  att_t, def_t = multiplicative strength factors (fitted)
  γ = home factor (~1.25)
```

Fitted by iterative proportional scaling (~60 iterations until convergence).

**Anti-overfitting measures**:

| Technique | Value | Rationale |
|---|---|---|
| Exponential time decay | 0.20/year (half-life ~3.5yr) | Recent matches more predictive; pre-validated via grid-search |
| Shrinkage prior | 8 pseudo-matches at average | Pulls low-data teams toward mean; prevents extreme extrapolation |
| Friendly weight | 1.0 (not downweighted) | Validated: friendlies carry full signal vs 0.5 in every backtest |
| Window | 25 years max | Decay weight negligible beyond this; removes Victorian-era noise |

---

#### Ensemble Blend (`predict.py`)

Both models produce expected goals (λ). Combined via geometric mean:

```
λ_final_h = λ_elo_h^w × λ_maher_h^(1-w)
λ_final_a = λ_elo_a^w × λ_maher_a^(1-w)

w = 0.5 (equal weight)
```

Weight chosen by walk-forward grid-search on 2006–2018 World Cups, validated on 2022. All weights in [0, 0.5] differ by <0.001 Brier — statistically indistinguishable. Equal weight is the minimax-regret pre-registered default.

If the two models disagree by >10pp on any outcome probability, the prediction is flagged `models_disagree: true`.

---

#### Scoreline Distribution: Poisson + Dixon-Coles Correction

```
P(h, a) = Poisson(h; λ_h) × Poisson(a; λ_a) × τ(h, a)
```

The Dixon-Coles correction τ adjusts low-score cells (0-0, 1-0, 0-1, 1-1) which independent Poissons systematically misprice:

```
τ(0,0) = 1 - λ_h × λ_a × ρ
τ(1,0) = 1 + λ_a × ρ
τ(0,1) = 1 + λ_h × ρ
τ(1,1) = 1 - ρ
τ(h,a) = 1 otherwise

ρ = -0.10 (classic 1997 empirical value)
```

This shifts ~1–2pp from wins/losses into draws, where the data says it belongs.

Outcome probabilities: sum all grid cells (up to 10×10) where h>a (home win), h=a (draw), h<a (away win).

---

#### Knockout Advance Probability

For matches that can go to extra time and penalties:

```
lean = 0.5 + (We - 0.5) × 0.33
P(advance | draw) = lean × shoot_rate_h / (lean × shoot_rate_h + (1-lean) × shoot_rate_a)
```

Shootout win rates are shrunk toward 50% with a prior of 8 pseudo-trials, preventing small-sample extremes.

---

#### Player Absence Quantification (`players.py`)

Uses `goalscorers.csv` to compute each player's share of their team's goals over a rolling 2-year window (excluding own goals).

```
xG_discount = goal_share × 0.40
```

Where 0.40 means a replacement recovers 60% of the absent player's contribution. Discount capped at 50% per team.

Example: Mbappé at ~23% of France's goals → ~4pp win probability reduction if absent. This matches empirical expectation, not narrative.

---

#### Qualitative Adjustments

Manual win-probability shifts bounded at ±10pp. Applied as:

```
home_win += N; away_win -= N; renormalize (preserves draw probability)
```

A reason string is required and logged, enabling post-hoc accuracy audit of manual calls. Typical magnitudes: 1–2pp for rotation player absent, 3–6pp for world-class absence, up to 10pp for squad crisis.

---

#### Backtest: Walk-Forward Validation (`backtest.py`)

Replays 2014, 2018, 2022 World Cups chronologically. For each match: predict from ratings as they stood that day (no future leakage), log Brier, update ratings, continue.

| Tournament | n | Elo Brier | Maher Brier | Ensemble Brier |
|---|---|---|---|---|
| 2014 | 64 | 0.537 | 0.586 | 0.558 |
| 2018 | 64 | 0.582 | 0.569 | 0.572 |
| 2022 | 64 | 0.629 | 0.597 | 0.610 |
| **Total** | **192** | 0.583 | 0.584 | **0.580** |

- Uniform guess Brier: 0.667
- Bookmakers: ~0.55–0.57
- Outcome accuracy: ~57%

Hyperparameter tuning rule: adopt new parameters only if **both** train AND test Brier improve. Train win + test loss = overfit to noise.

---

#### Tournament Simulation (`simulate.py`)

Runs the remaining bracket 50,000 times. Completed matches keep real scores; unplayed matches sample scorelines from the Elo+Maher ensemble. Groups derived from fixture cliques; knockout bracket seeded by group stage performance. Hosts (USA, Mexico, Canada) retain home advantage throughout.

Output: per-team probability of reaching R32, R16, QF, SF, Final, Champion.

---

## Project Structure

```
WorldCup/
├── start.sh                    # Mac/Linux: full pipeline + serve dashboard
├── start.bat                   # Windows equivalent
├── data/
│   ├── results.csv             # 49K+ matches since 1872 (auto-downloaded)
│   ├── goalscorers.csv         # 47K+ goal records (auto-downloaded)
│   ├── shootouts.csv           # Penalty records (auto-downloaded)
│   ├── manual_results.csv      # User-entered recent results (tournament lag)
│   ├── elo_ratings.json        # Current Elo state
│   ├── elo_history.json        # Timestamped Elo snapshots (for dashboard movement)
│   ├── maher_ratings.json      # Maher model + ensemble weight
│   └── tournament.json         # Optional: manual tournament group/bracket structure
├── predictions/
│   ├── log.json                # All predictions + resolved actuals
│   ├── tournament_odds.json    # Monte Carlo stage reach probabilities
│   └── dashboard.html          # Self-contained HTML dashboard
└── world-cup-predictor/
    ├── SKILL.md                # Complete session workflow reference
    ├── references/
    │   └── methodology.md      # Full mathematical methodology
    └── scripts/
        ├── run_all.py          # Automated full-pipeline orchestrator
        ├── fetch_data.py       # Download datasets from GitHub
        ├── elo.py              # Build/rebuild Elo ratings
        ├── maher.py            # Fit Maher attack/defence model
        ├── predict.py          # Predict a single match
        ├── record.py           # Resolve predictions vs actual results
        ├── simulate.py         # Monte Carlo tournament simulation
        ├── dashboard.py        # Generate HTML dashboard
        ├── players.py          # Player scoring profiles and absence impact
        ├── serve.py            # Local HTTP server with /rerun endpoint
        └── backtest.py         # Walk-forward validation and hyperparameter tuning
```

---

## Setup

No package installation required. Requires Python 3.6+.

```bash
git clone <repo>
cd WorldCup

# Download data and run full pipeline
python3 world-cup-predictor/scripts/run_all.py --days 2 --simulate
```

Or use the launchers:

```bash
./start.sh        # Mac/Linux — runs pipeline then opens dashboard in browser
start.bat         # Windows
```

---

## Usage

### Full automated pipeline

```bash
python3 world-cup-predictor/scripts/run_all.py --days 2 --simulate
```

Runs in sequence: fetch → elo → maher → record → predict upcoming → simulate → dashboard.

### Predict a single match

```bash
python3 world-cup-predictor/scripts/predict.py <home> <away> [FLAGS]
```

Key flags:

| Flag | Description |
|---|---|
| `--neutral` | No home advantage |
| `--knockout` | Knockout match (adds advance/penalties probability) |
| `--date YYYY-MM-DD` | Match date (required for logging) |
| `--stage "Group A"` | Tournament stage |
| `--city "Mexico City"` | Venue (flags altitude >1200m) |
| `--missing-home "Player"` | Confirmed absent — applies data-derived xG discount |
| `--missing-away "Player"` | Same for away team |
| `--adjust-home N` | Manual win prob shift in pp (capped ±10) |
| `--adjust-reason "text"` | Required reason for any adjustment |
| `--market "2.05,3.40,3.90"` | Decimal odds for market comparison (never fed to model) |
| `--log` | Append to predictions/log.json |

**Example**:

```bash
python3 world-cup-predictor/scripts/predict.py Brazil Morocco \
  --neutral --stage "Group C" --date 2026-06-13 \
  --missing-home "Vinicius" --market "1.85,3.40,4.20" --log
```

### Player absence analysis

```bash
python3 world-cup-predictor/scripts/players.py "France" --player "Mbappé"
```

Shows goal share over last 2 years and estimated xG discount if absent.

### Backtest and hyperparameter tuning

```bash
# Walk-forward validation on 2014–2022
python3 world-cup-predictor/scripts/backtest.py

# Grid-search Elo hyperparameters (train 2006–2018, test 2022)
python3 world-cup-predictor/scripts/backtest.py --tune

# Grid-search Maher hyperparameters
python3 world-cup-predictor/scripts/backtest.py --tune-maher

# Choose ensemble blend weight (--write stores result)
python3 world-cup-predictor/scripts/backtest.py --tune-blend --write

# Side-by-side Elo vs Maher vs ensemble comparison
python3 world-cup-predictor/scripts/backtest.py --models
```

### Dashboard server

```bash
python3 world-cup-predictor/scripts/serve.py
# Opens http://localhost:8000 in browser
# POST /rerun triggers a fresh pipeline run from the browser's "Re-run" button
```

---

## Dashboard

The dashboard (`predictions/dashboard.html`) is fully self-contained — no CDN, no external requests.

**Sections**:
- **Model record** — outcomes correct, mean Brier score, vs market comparison
- **Calibration table** — stated confidence vs actual hit rate (5 buckets, requires ≥5 resolved)
- **Upcoming predictions** — probability bars, expected goals, most likely scoreline, Elo gaps, model disagreement flags, market odds
- **Previous matches** — predicted vs actual score, hit/miss, Brier per match, exact scoreline hits
- **Championship odds** — top 12 teams, Champion/Final/SF %, Monte Carlo error band
- **Elo top 20** — rating, movement since last snapshot, last match date

---

## Data Sources

| Dataset | Source | Update frequency |
|---|---|---|
| Match results | [martj42/international_results](https://github.com/martj42/international_results) | Daily (1–2 day lag) |
| Goal scorers | Same repo | Daily |
| Penalty shootouts | Same repo | Daily |
| Recent match stub | `data/manual_results.csv` (user-maintained) | Manual during tournaments |

---

## Limitations

- **Bookmaker gap**: 0.580 Brier vs ~0.55–0.57 for markets. Difference is bookmakers have access to squad data, transfer intel, and real-money calibration signal.
- **Lineup data**: No confirmed XI until 60–75 minutes before kickoff. Player layer only accounts for confirmed absences, not tactical changes.
- **2026 format**: 48-team format is unprecedented. Knockout calibration was fitted on 32-team era data — treat championship odds as provisional until real bracket loads.
- **Penalties**: Approximate coin flips. Modelled as strength-weighted with shrinkage, but genuine uncertainty is high.
- **Football ceiling**: Even bookmakers can't beat ~55% accuracy. Low-scoring sport with high variance — randomness is irreducible.

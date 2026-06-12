# Methodology

How the predictor works, why each piece exists, and what its limits are.

## 1. Data

`martj42/international_results` — every men's full international since 1872,
~49,500 rows: date, teams, score, tournament, city, country, neutral flag.
Updated by its maintainer every few days; during tournaments
`data/manual_results.csv` bridges the gap. Upcoming fixtures appear in the
same file with `NA` scores, which is how the skill knows the schedule.

Friendlies and qualifiers are included deliberately: they carry less weight
(lower K), but excluding them throws away most of what we know about teams
that rarely reach major tournaments.

## 2. Elo ratings (`elo.py`)

Sequential Elo in the style of eloratings.net:

- Expected result: `We = 1 / (1 + 10^(-dr/400))` where `dr` is the rating gap
  plus **+100 for the home team** in non-neutral matches.
- Update: `R' = R + K * G * (W - We)` with W ∈ {1, 0.5, 0}.
- **K by importance**: World Cup finals 60, continental finals 50,
  qualifiers 40, other tournaments 30, friendlies 20. A World Cup upset moves
  ratings three times more than a friendly upset.
- **Margin multiplier G**: 1 for ≤1-goal margins, 1.5 for 2 goals,
  `(11+gd)/8` for 3+. Winning 4–0 says more than winning 1–0.
- **Recency**: sequential updating means the rating *is* a recency-weighted
  summary — each new result pulls the rating toward current strength, and
  older matches' influence decays with every update since. Additionally,
  teams inactive 4+ years regress 20% toward 1500 (squads turn over).

This covers the article's "recent games count more" requirement without an
explicit decay window: a match last month typically shifts a rating ~10–40
points; a 2014 match's contribution has been overwritten by everything since.

## 3. From Elo gap to scoreline (`predict.py`)

Two constants fitted from post-1990 matches at every `elo.py` run:

- `goal_diff_per_elo`: least-squares slope (through origin) of actual goal
  difference vs pre-match Elo gap. Turns "Brazil is +180 on Morocco" into
  "expect ~0.8 more goals".
- `base_goals`: average total goals in matches between near-equal teams
  (|gap| < 100). Anchors how many goals a typical match has.

Each team gets a Poisson rate: `λ_home = base/2 + expected_gd/2`,
`λ_away = base/2 − expected_gd/2` (floored at 0.15). The full 11×11 scoreline
matrix gives P(win), P(draw), P(loss) and the most likely scores.

Independent Poissons have a known bias: they underpredict draws, because real
scorelines are slightly negatively correlated at low scores (teams at 0-0 or
1-1 late in a match settle for the point). The **Dixon–Coles correction**
(ρ = −0.10, the classic fitted value from the 1997 paper) reweights the
0-0, 1-0, 0-1 and 1-1 cells to fix this — it typically moves 1–2 percentage
points from the win/loss outcomes onto the draw, which matters because draws
are exactly what group-stage predictions get wrong most.

Knockout draws: advance probability = `P(win) + P(draw) × pen_share`, where
`pen_share = 0.5 + (We − 0.5) × 0.33` — penalties are mostly a coin flip,
slightly leaning to the stronger team.

## 3b. The second model: Maher attack/defence (`maher.py`)

Elo compresses a team into one number, which forces an assumption: a +200
team is +200 in both attack and defence. The Maher (1982) model drops that
assumption by fitting two multiplicative strengths per team on goals scored:

    λ_home = μ · att_home · def_away · γ      (γ ≈ 1.28, home factor)
    λ_away = μ · att_away · def_home

fitted by weighted maximum likelihood (iterative proportional scaling) with
three anti-overfitting measures:

- **Time decay** (Dixon & Coles 1997): a match `y` years old gets weight
  `exp(−0.20·y)` (half-life ≈ 3.5 years), so the fit tracks current squads.
- **Shrinkage / partial pooling**: every team gets 8 pseudo-matches of
  exactly average performance, pulling thin records toward att = def = 1.
  Without it, isolated low-data teams get absurd ratings from a few results
  against equally poorly-estimated opponents.
- **Friendly weighting**: checked out-of-sample; full weight (1.0) beat 0.5
  in every tournament tested, so friendlies count fully.

### The ensemble

The final expected goals are the geometric blend of the two models' lambdas:
`λ = λ_elo^w · λ_maher^(1−w)` with **w = 0.5**. The weight was evaluated
walk-forward on WC 1998–2022: Maher-pure won 4 of 7 tournaments, Elo-pure 2,
and mean Brier for w ∈ [0, 0.5] differed by under 0.001 — statistically
indistinguishable. Equal weight is the minimax-regret choice across the panel
and was the pre-registered default, so nothing was tuned toward any single
tournament. On 2014–2022 (192 matches) the ensemble scores Brier 0.5800 vs
0.5826 (Elo) and 0.5839 (Maher).

When the two models differ by more than 10 percentage points on any outcome
the prediction carries a `models_disagree` flag: two independent reads of the
same data conflict, so the stated confidence is optimistic.

## 3c. The market benchmark (`--market`)

Bookmaker closing odds are the academic gold standard for unbiased football
probabilities — they aggregate every public model plus inside information.
This system uses them as the **scoreboard, never an input**: decimal odds
passed via `--market H,D,A` are de-vigged (implied probabilities normalised
to remove the bookmaker margin, typically 4–7%) and stored with the
prediction. `record.py` then scores model vs market vs a 50/50 log-pool
blend on every resolved match. If the model cannot beat the market's Brier
over the tournament, that is reported honestly — beating 0.667 guessing is
easy, beating the market is the real test.

## 4. Qualitative adjustment layer

Elo knows results; it does not know that the first-choice keeper got injured
in training yesterday. The adjustment flags exist for exactly that class of
information — *news the rating cannot contain yet*.

Guard rails, and why:
- **Capped at ±10 percentage points**: history says lineup news rarely swings
  a match more than that; bigger swings are usually narrative, not signal.
- **Reason required and logged**: `record.py` tags resolved predictions that
  were adjusted, so over time there is an answer to "do my adjustments beat
  the raw model?" If they don't, stop making them.
- **Don't double-count form**: recent wins/losses are already in the rating.
  Adjust only for things Elo can't see — injuries, suspensions, unusual
  travel/rest gaps, confirmed heavy squad rotation.

### Player layer (`players.py`, `--missing-*`)

The goalscorers dataset (47,000+ scorer records) gives each team's actual
scoring distribution. A confirmed absence is priced as:
`xG_discount = player_goal_share × 0.40` — the 0.40 reflects that a
substitute replaces the player, recovering roughly 60% of the lost output
(consistent with published star-absence studies). Capped at 50% total per
team. Limits: goal share misses defensive/creative value (a keeper's absence
shows up as 0%), so the manual adjustment flag exists alongside it. The
"top-3 reliance" metric flags teams structurally vulnerable to one injury.

### Shootout history

Penalty shootouts aren't quite coin flips: some national sides are
persistently better (preparation, culture, keeper specialization). Knockout
draw resolution combines the Elo strength lean with each team's historical
shootout win rate, shrunk toward 50% with a prior weight of 8 shootouts so
small samples can't dominate (a team with 2/2 wins gets 60%, not 100%).

### Knockout scoring factor

Measured from WC 1986+ (group stage = first N matches per edition): knockout
matches average ~3% *more* goals than group matches (`knockout_goals_factor`
in the calibration). Caveat: extra-time goals are included in recorded
scores, which inflates the knockout side; the honest conclusion is "no
meaningful difference", and the measured factor is applied as-is rather than
a folk-wisdom "knockouts are cagey" discount the data doesn't support.

### Hyperparameter tuning (`backtest.py --tune`)

Home advantage, K scale, and ρ were grid-searched with a train/test split
(2006–2018 train, 2022 held out). Result: K scale flips sign between train
and test (overfitting signal), home advantage 60 vs 100 differs by ~0.003
Brier (noise at n=64). The eloratings.net-standard defaults survived an
honest attempt to beat them — that's the unbiased way to choose parameters.

### Rest days and altitude

Both are reported as context, not folded into the probability — there's no
clean calibration for them in this dataset. Rest differential ≥3 days in a
compressed tournament, or a lowland team visiting Mexico City (2240 m),
justifies a small manual adjustment.

### Context the numbers don't capture

- **Head-to-head**: `predict.py` reports the historical record between the two
  sides. It is presented as narrative context only — those results are already
  inside both Elo ratings, so feeding it back into the probability would
  double-count.
- **Market comparison**: bookmaker odds are the strongest public forecast of a
  football match (they aggregate lineup news, weather, and sharp money). The
  workflow compares the model against the market when odds are available and
  flags gaps >10pp. The model is deliberately *not* moved toward the market —
  a logged disagreement is testable later; a blended number is not.

## 5. Feedback loop (`record.py`)

Every logged prediction stores the full probability vector. Once the real
result lands:

- **Outcome accuracy**: did the highest-probability outcome happen? Crude but
  legible. A good model on World Cup group stages lands roughly 55–65%
  (draws are inherently hard — they're rarely the single most likely outcome
  even when 25% likely).
- **Brier score** (3-outcome): `Σ (p_i − o_i)²`. 0 perfect, 0.667 for the
  uniform-guess baseline, 2 worst. This is the honest metric: it rewards
  calibrated probabilities, not lucky picks. Falling below ~0.58 average
  beats naive guessing; ~0.45 is strong for football.

The result also enters `results.csv`/`manual_results.csv`, so the next
`elo.py` run folds it into the ratings — the "add results back into the next
analysis" loop the user asked for is the Elo update itself.

## 6. Tournament simulation (`simulate.py`)

The article's "play the tournament 50,000 times". Each run samples every
remaining match from the Poisson model (played matches keep real scores),
applies 2026 rules (12 groups of 4; top two plus 8 best thirds advance), and
plays out the knockout. Champion odds = wins / runs. Sampling error at 50k
runs is ±0.2pp for a 10% team — fine for headline odds.

**Known approximation**: the round-of-32 bracket is built by seeding
qualifiers on group-stage performance rather than FIFA's fixed
group-letter/third-place allocation chart. This perturbs individual R32
matchups but moves championship probabilities very little (strong teams must
beat strong teams eventually either way). For exact bracket fidelity, supply
`bracket_pairs` in `data/tournament.json`.

## 7. Validation (`backtest.py`)

Walk-forward replay of the 2014/2018/2022 World Cups: every match predicted
from the ratings as they stood that day, then the real result applied. No
information from the future leaks into any prediction. The Maher model is
refit once at each tournament's first match (squad strength barely moves in
four weeks); Elo updates match by match. Current Brier scores
(`backtest.py --models`):

| WC    | n   | Elo    | Maher  | Ensemble |
|-------|-----|--------|--------|----------|
| 2014  | 64  | 0.537  | 0.586  | 0.558    |
| 2018  | 64  | 0.582  | 0.569  | 0.572    |
| 2022  | 64  | 0.629  | 0.597  | 0.610    |
| total | 192 | 0.583  | 0.584  | **0.580**|

The ensemble is never the best model in any single year — that is the point.
Each component wins where the other fails, and the blend is best on total
while avoiding either's worst year. Uniform guessing scores 0.667; bookmakers
land around 0.55–0.57 Brier on the same tournaments. 2022 was an upset-heavy
outlier for every model. Rerun after changing K factors, ρ, calibration,
home advantage, or any Maher knob — if these numbers degrade, the change was
a bad idea regardless of how clever it felt.

### Considered and rejected (deliberately)

Documented so they aren't "discovered" again and bolted on without evidence:

- **Modelling rest days numerically**: group-stage rest is nearly uniform
  and the fitted effect size on past WCs was indistinguishable from noise;
  an unvalidated knob is a bias vector. Reported as context only.
- **Head-to-head as a probability input**: Elo already contains every past
  meeting; counting them again double-weights old results. Reported as
  colour only.
- **Picking the Maher decay the 2022 test set preferred** (0.05–0.1/yr):
  train years preferred fast decay, 2022 preferred slow, per-year winners
  split 4–3. Choosing the test set's favourite is itself overfitting, so the
  pre-registered 0.20/yr default stands.
- **Blending market odds into the prediction**: would make the scoreboard
  meaningless (you can't beat a benchmark you copied). The 50/50 blend is
  logged and scored, but the headline prediction stays model-only.
- **xG / player-rating models**: would likely improve accuracy but need
  shot-level data with no reliable free source for internationals.

## 8. Honest limitations

State these when a user pushes on a surprising number:

- National-team Elo lags real strength changes around generational shifts
  (a golden cohort retiring all at once).
- Player-level data is not modelled numerically — only via the bounded
  adjustment layer. (xG-based or player-rating models exist but need data
  sources this skill doesn't bundle.)
- Penalties are nearly random; knockout odds past the quarterfinals carry
  wide error bars.
- Probabilities are honest, not certain: the favourite losing a 65/35 match
  is the model working, 35% of the time.

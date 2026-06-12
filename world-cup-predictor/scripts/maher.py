#!/usr/bin/env python3
"""Maher (1982) attack/defence Poisson model — the second, independent model.

Where Elo compresses a team into one number, this fits two: an attack and a
defence strength per team, by weighted maximum likelihood on goals scored.

  lambda_home = mu * att_home * def_away * gamma   (gamma = home factor,
  lambda_away = mu * att_away * def_home            1 on neutral ground)

Anti-overfitting measures, all chosen on 2006-2018 World Cups and validated
out-of-sample on 2022 via backtest.py --tune-maher (never on the data being
predicted):
- Time decay: a match `y` years old gets weight exp(-decay * y), so the fit
  tracks current squads instead of decades-old history (Dixon & Coles 1997).
- Shrinkage / partial pooling: every team gets `prior_n` pseudo-matches of
  exactly average performance, pulling low-data teams toward att=def=1
  instead of letting 10 lucky matches produce an extreme rating.
- Friendly down-weighting: friendlies use rotated squads; they enter the fit
  at reduced weight.
- Window: matches older than WINDOW_YEARS are excluded outright (their decay
  weight is negligible; skipping them is pure speed).

Fitting is iterative proportional scaling on the Poisson likelihood (each
parameter updated by the ratio of observed to expected goals, with the
pseudo-counts added to both sides), which converges in a few dozen rounds.

Writes data/maher_ratings.json: mu, gamma, per-team att/def, and the ensemble
blend weight (set by backtest.py --tune-blend; predict.py and simulate.py
read it). Stdlib only.
"""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from elo import load_matches  # noqa: E402

# Defaults checked out-of-sample by backtest.py --tune-maher (train WC
# 1998-2018, test 2022). friendly_weight=1.0 beat 0.5 in every year, so it
# was adopted. decay was weakly identified (per-year winners split 4-3
# between fast and slow), so the pre-registered 0.20 stays — choosing the
# value the 2022 test set happened to like would itself be overfitting.
# prior_n had negligible effect; 8 keeps shrinkage for low-data teams.
DECAY_PER_YEAR = 0.20      # half-life ~3.5 years
PRIOR_N = 8.0              # pseudo-matches of average performance per team
FRIENDLY_WEIGHT = 1.0
WINDOW_YEARS = 25
ITERATIONS = 60


def fit(matches, as_of=None, decay=DECAY_PER_YEAR, prior_n=PRIOR_N,
        friendly_weight=FRIENDLY_WEIGHT, window_years=WINDOW_YEARS):
    """Fit the model on matches strictly before `as_of` (default: today).

    Returns dict with mu, gamma, and {team: {att, def, weight}} where weight
    is the team's effective (decayed) match count — a data-volume indicator.
    """
    as_of = as_of or date.today().isoformat()
    ref = datetime.strptime(as_of, "%Y-%m-%d").date()
    cutoff = f"{ref.year - window_years}-01-01"

    rows = []  # (weight, home, away, hs, as, neutral)
    for m in matches:
        if not (cutoff <= m["date"] < as_of):
            continue
        d = datetime.strptime(m["date"], "%Y-%m-%d").date()
        w = 2.718281828 ** (-decay * (ref - d).days / 365.25)
        if "friendly" in m["tournament"].lower():
            w *= friendly_weight
        if w < 0.005:
            continue
        rows.append((w, m["home"], m["away"], m["hs"], m["as"], m["neutral"]))
    if not rows:
        raise ValueError(f"No matches in window before {as_of}")

    teams = {t for _, h, a, *_ in rows for t in (h, a)}
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    w_total = sum(w for w, *_ in rows)
    goals_total = sum(w * (hs + as_) for w, _, _, hs, as_, _ in rows)
    mu = goals_total / (2 * w_total)  # average goals per team per match
    gamma = 1.25
    prior_goals = prior_n * mu  # pseudo-counts added to observed and expected

    for _ in range(ITERATIONS):
        scored = {t: prior_goals for t in teams}
        exp_scored = {t: prior_goals for t in teams}
        conceded = {t: prior_goals for t in teams}
        exp_conceded = {t: prior_goals for t in teams}
        for w, h, a, hs, as_, neutral in rows:
            g = 1.0 if neutral else gamma
            lam_h = mu * att[h] * dfn[a] * g
            lam_a = mu * att[a] * dfn[h]
            scored[h] += w * hs;  exp_scored[h] += w * lam_h
            scored[a] += w * as_; exp_scored[a] += w * lam_a
            conceded[a] += w * hs;  exp_conceded[a] += w * lam_h
            conceded[h] += w * as_; exp_conceded[h] += w * lam_a
        for t in teams:
            att[t] *= scored[t] / exp_scored[t]
            dfn[t] *= conceded[t] / exp_conceded[t]
        # renormalise so mean att = mean def = 1; mu and gamma absorb scale
        att_mean = sum(att.values()) / len(att)
        dfn_mean = sum(dfn.values()) / len(dfn)
        for t in teams:
            att[t] /= att_mean
            dfn[t] /= dfn_mean
        # refit mu and gamma exactly against the updated strengths
        total_obs = total_base = home_obs = home_base = 0.0
        for w, h, a, hs, as_, neutral in rows:
            base_h = att[h] * dfn[a]          # lam_h / (mu*gamma_term)
            base_a = att[a] * dfn[h]
            if neutral:
                total_base += w * (base_h + base_a)
            else:
                total_base += w * (base_h * gamma + base_a)
                home_obs += w * hs
                home_base += w * base_h
            total_obs += w * (hs + as_)
        mu = total_obs / total_base
        gamma = home_obs / (mu * home_base) if home_base else gamma

    eff = {t: prior_n for t in teams}
    for w, h, a, *_ in rows:
        eff[h] += w; eff[a] += w
    return {"as_of": as_of, "mu": round(mu, 4), "gamma": round(gamma, 4),
            "decay_per_year": decay, "prior_n": prior_n,
            "friendly_weight": friendly_weight,
            "matches_used": len(rows),
            "teams": {t: {"att": round(att[t], 4), "def": round(dfn[t], 4),
                          "eff_matches": round(eff[t], 1)}
                      for t in sorted(teams, key=lambda t: -(att[t] / dfn[t]))}}


def lambdas(model, home, away, neutral):
    """Expected goals for a fixture under a fitted model."""
    th, ta = model["teams"][home], model["teams"][away]
    g = 1.0 if neutral else model["gamma"]
    return (model["mu"] * th["att"] * ta["def"] * g,
            model["mu"] * ta["att"] * th["def"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--decay", type=float, default=DECAY_PER_YEAR)
    parser.add_argument("--prior-n", type=float, default=PRIOR_N)
    parser.add_argument("--friendly-weight", type=float, default=FRIENDLY_WEIGHT)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    matches = load_matches(data_dir)
    if not matches:
        print("No matches found. Run fetch_data.py first.", file=sys.stderr)
        return 1
    model = fit(matches, decay=args.decay, prior_n=args.prior_n,
                friendly_weight=args.friendly_weight)

    # keep the ensemble blend weight if one was already validated
    dest = data_dir / "maher_ratings.json"
    if dest.exists():
        old = json.loads(dest.read_text(encoding="utf-8"))
        if "ensemble" in old:
            model["ensemble"] = old["ensemble"]
    dest.write_text(json.dumps(model, indent=1, ensure_ascii=False),
                    encoding="utf-8")

    print(f"Maher fit on {model['matches_used']} weighted matches "
          f"(decay {args.decay}/yr, prior {args.prior_n}, "
          f"friendlies x{args.friendly_weight}). mu={model['mu']:.3f}, "
          f"home gamma={model['gamma']:.3f}. Saved {dest}")
    print("\nTop 15 by att/def ratio:")
    for i, (t, v) in enumerate(list(model["teams"].items())[:15], 1):
        print(f"{i:3d}. {t:25s} att {v['att']:.3f}  def {v['def']:.3f}  "
              f"(eff n={v['eff_matches']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

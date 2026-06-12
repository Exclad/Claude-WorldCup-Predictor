#!/usr/bin/env python3
"""Player-level analysis from the goalscorers dataset.

Shows a team's scoring distribution over a recent window: who actually
produces the goals, and how reliant the team is on its top scorers. This
turns "their star is injured" from a hunch into a number — predict.py uses
the same data via --missing-home/--missing-away to discount a team's
expected goals when a named player is out.

Usage:
  players.py "Brazil"                  scoring profile, last 2 years
  players.py "Brazil" --years 4        wider window
  players.py "Brazil" --player "Name"  one player's share + suggested impact
"""
import argparse
import csv
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

# A missing scorer is replaced by a substitute, not by nothing; historical
# studies of star absences suggest the team loses roughly 40% of the missing
# player's direct goal contribution.
REPLACEMENT_LOSS = 0.40


def team_goals(team, data_dir, years):
    cutoff = (date.today() - timedelta(days=int(years * 365))).isoformat()
    path = Path(data_dir) / "goalscorers.csv"
    if not path.exists():
        sys.exit(f"{path} not found — run fetch_data.py first.")
    scorers = Counter()
    total = pens = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["team"] != team or r["date"] < cutoff:
                continue
            if r["own_goal"].strip().upper() == "TRUE":
                continue
            total += 1
            if r["penalty"].strip().upper() == "TRUE":
                pens += 1
            scorers[r["scorer"]] += 1
    return scorers, total, pens


def player_share(team, player, data_dir, years=2):
    """Goal share for one player; tolerant of partial name matches."""
    scorers, total, _ = team_goals(team, data_dir, years)
    if total == 0:
        return None, 0, total
    if player in scorers:
        return player, scorers[player] / total, total
    matches = [s for s in scorers if player.lower() in s.lower()]
    if len(matches) == 1:
        return matches[0], scorers[matches[0]] / total, total
    if len(matches) > 1:
        sys.exit(f"Ambiguous player '{player}' for {team}: {matches}")
    return None, 0.0, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team")
    parser.add_argument("--player", help="show one player's share and impact")
    parser.add_argument("--years", type=float, default=2)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    scorers, total, pens = team_goals(args.team, args.data_dir, args.years)
    if total == 0:
        sys.exit(f"No goals recorded for '{args.team}' in the last {args.years} years "
                 "— check the team name against data/results.csv.")

    if args.player:
        name, share, total = player_share(args.team, args.player, args.data_dir, args.years)
        if name is None:
            print(f"'{args.player}' has no goals for {args.team} in the window — "
                  "absence likely matters via defense/creation, not scoring; "
                  "use a small manual adjustment instead.")
            return 0
        print(f"{name}: {scorers[name]}/{total} of {args.team}'s goals "
              f"({share:.0%}) over last {args.years:g} years")
        print(f"If absent: expected-goals discount ~{share * REPLACEMENT_LOSS:.0%} "
              f"(pass --missing-* '{name}' to predict.py to apply it)")
        return 0

    top = scorers.most_common(10)
    top3_share = sum(n for _, n in scorers.most_common(3)) / total
    print(f"{args.team} — {total} goals, {len(scorers)} scorers, "
          f"last {args.years:g} years")
    print(f"Top-3 reliance: {top3_share:.0%} of all goals"
          + ("  (high — vulnerable to absences)" if top3_share > 0.5 else ""))
    if total >= 10 and pens / total > 0.25:
        print(f"Penalty dependence: {pens}/{total} goals from the spot "
              f"({pens/total:.0%}) — penalty volume is luck-heavy and tends to "
              "regress; treat their attack as slightly weaker than raw goals suggest")
    for name, n in top:
        print(f"  {n:3d} ({n/total:4.0%})  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

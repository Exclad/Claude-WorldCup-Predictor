#!/usr/bin/env python3
"""Build Elo ratings for national teams from the full match history.

Method (based on eloratings.net's World Football Elo Ratings):
- Every team starts at 1500 and is updated sequentially through history, so
  recent matches naturally dominate a team's current rating.
- K factor scales with match importance (World Cup 60 ... friendly 20).
- Margin-of-victory multiplier rewards big wins.
- +100 Elo home advantage for the home team in non-neutral matches.
- Teams inactive for 4+ years are regressed 20% toward 1500 before their next
  match (squads turn over; stale ratings shouldn't persist at full strength).

Also fits two calibration constants from post-1990 data, used by predict.py to
turn an Elo gap into expected goals:
- slope: expected goal difference per Elo point (least squares through origin)
- base_goals: average total goals in matches between near-equal teams

Reads data/results.csv plus optional data/manual_results.csv (same columns,
for results newer than the dataset). Writes data/elo_ratings.json.
"""
import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

START_ELO = 1500.0
HOME_ADV = 100.0
INACTIVITY_YEARS = 4
INACTIVITY_REGRESSION = 0.20
CALIBRATION_FROM_YEAR = 1990


def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "friendly" in t:
        return 20
    if "qualification" in t:
        return 40
    if "fifa world cup" in t:
        return 60
    continental = ("uefa euro", "copa américa", "copa america", "african cup",
                   "africa cup", "afc asian cup", "gold cup", "confederations",
                   "nations league finals", "oceania")
    if any(c in t for c in continental):
        return 50
    return 30


def goal_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def load_matches(data_dir: Path):
    matches = []
    seen = set()
    for name in ("results.csv", "manual_results.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["home_score"] in ("", "NA") or row["away_score"] in ("", "NA"):
                    continue  # fixture without a result yet
                key = (row["date"], row["home_team"], row["away_team"])
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "date": row["date"],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "hs": int(float(row["home_score"])),
                    "as": int(float(row["away_score"])),
                    "tournament": row["tournament"],
                    "neutral": row["neutral"].strip().upper() == "TRUE",
                })
    matches.sort(key=lambda m: m["date"])
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    matches = load_matches(data_dir)
    if not matches:
        print("No matches found. Run fetch_data.py first.", file=sys.stderr)
        return 1

    ratings, info = {}, {}
    cal_xy, cal_xx = 0.0, 0.0          # goal-diff vs elo-diff regression
    close_goals, close_n = 0, 0        # total goals between near-equal teams
    wc_by_year = {}                    # WC finals matches, for knockout factor

    for m in matches:
        d = datetime.strptime(m["date"], "%Y-%m-%d").date()
        for team in (m["home"], m["away"]):
            if team not in ratings:
                ratings[team] = START_ELO
                info[team] = {"matches": 0, "last_match": m["date"]}
            else:
                last = datetime.strptime(info[team]["last_match"], "%Y-%m-%d").date()
                if (d - last).days > INACTIVITY_YEARS * 365:
                    ratings[team] += (START_ELO - ratings[team]) * INACTIVITY_REGRESSION

        dr = ratings[m["home"]] - ratings[m["away"]]
        if not m["neutral"]:
            dr += HOME_ADV
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))

        gd = m["hs"] - m["as"]
        if d.year >= CALIBRATION_FROM_YEAR:
            cal_xy += dr * gd
            cal_xx += dr * dr
            if abs(dr) < 100:
                close_goals += m["hs"] + m["as"]
                close_n += 1
        if m["tournament"] == "FIFA World Cup" and d.year >= 1986:
            wc_by_year.setdefault(d.year, []).append(m["hs"] + m["as"])

        result = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = k_factor(m["tournament"]) * goal_multiplier(gd) * (result - we)
        ratings[m["home"]] += delta
        ratings[m["away"]] -= delta
        for team in (m["home"], m["away"]):
            info[team]["matches"] += 1
            info[team]["last_match"] = m["date"]

    # knockout matches are cagier than group matches; measure by how much.
    # Matches arrive date-sorted, so per edition the group stage is simply the
    # first N matches (36 for 24-team WCs 1986-94, 48 for 32-team 1998-2022,
    # 72 for 48-team 2026+).
    grp_g = grp_n = ko_g = ko_n = 0
    for year, goals in wc_by_year.items():
        n_group = 36 if year <= 1994 else (48 if year <= 2022 else 72)
        grp_g += sum(goals[:n_group]); grp_n += len(goals[:n_group])
        ko_g += sum(goals[n_group:]); ko_n += len(goals[n_group:])
    knockout_factor = (ko_g / ko_n) / (grp_g / grp_n) if ko_n and grp_n else 1.0

    out = {
        "updated": date.today().isoformat(),
        "latest_match": matches[-1]["date"],
        "total_matches": len(matches),
        "calibration": {
            "goal_diff_per_elo": cal_xy / cal_xx if cal_xx else 0.0045,
            "base_goals": close_goals / close_n if close_n else 2.6,
            "knockout_goals_factor": round(knockout_factor, 4),
            "home_advantage_elo": HOME_ADV,
        },
        "ratings": {
            team: {"elo": round(ratings[team], 1), **info[team]}
            for team in sorted(ratings, key=lambda t: -ratings[t])
        },
    }
    dest = data_dir / "elo_ratings.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # snapshot history so the dashboard can show rating movement over time
    hist_path = data_dir / "elo_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    today = date.today().isoformat()
    history = [h for h in history if h["date"] != today]
    history.append({"date": today, "latest_match": matches[-1]["date"],
                    "ratings": {t: round(r, 1) for t, r in ratings.items()}})
    hist_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

    print(f"Rated {len(ratings)} teams from {len(matches)} matches "
          f"(latest: {matches[-1]['date']}). Saved {dest}")
    print(f"Calibration: {out['calibration']['goal_diff_per_elo']:.5f} goal-diff/Elo, "
          f"{out['calibration']['base_goals']:.2f} base goals")
    print("\nTop 15:")
    for i, (team, r) in enumerate(list(out["ratings"].items())[:15], 1):
        print(f"{i:3d}. {team:25s} {r['elo']:7.1f}  ({r['matches']} matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

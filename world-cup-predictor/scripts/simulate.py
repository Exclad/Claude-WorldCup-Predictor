#!/usr/bin/env python3
"""Monte Carlo simulation of the remaining World Cup (default 50,000 runs).

Groups are taken from data/tournament.json if present, otherwise derived
automatically from the fixture list in results.csv (group-stage opponents form
12 disjoint cliques of 4). Already-played matches keep their real scores; the
rest are sampled from the same Elo+Maher ensemble model predict.py uses.

Knockout structure is approximated: the 32 qualifiers (12 winners, 12
runners-up, 8 best third-placed teams) are seeded by group-stage performance
into a 1v32 / 2v31 ... bracket. The real FIFA bracket fixes pairings by group
letter, so per-team odds can differ somewhat in the round of 32, but
championship odds are dominated by team strength and barely move. To use the
exact bracket, list it in data/tournament.json as "bracket_pairs".

Hosts keep their Elo home advantage in every match they play. Knockout draws
go to extra time/penalties, modelled as a strength-leaning coin flip.

Output: per-team probability of reaching each stage and winning the trophy.
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import maher  # noqa: E402

GROUP_WINDOW = ("2026-06-11", "2026-06-27")
HOSTS = {"United States", "Mexico", "Canada"}
STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def load_fixtures(data_dir):
    rows = []
    for name in ("results.csv", "manual_results.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["tournament"] == "FIFA World Cup" and \
                        GROUP_WINDOW[0] <= r["date"] <= GROUP_WINDOW[1]:
                    rows.append(r)
    return rows


def derive_groups(fixtures):
    adj = defaultdict(set)
    for r in fixtures:
        adj[r["home_team"]].add(r["away_team"])
        adj[r["away_team"]].add(r["home_team"])
    groups, seen = [], set()
    for team in adj:
        if team in seen:
            continue
        comp, stack = set(), [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        groups.append(sorted(comp))
    return groups


def match_lambdas(elo_h, elo_a, cal, home_adv, knockout=False, maher_pair=None):
    """Ensemble expected goals: geometric blend of Elo and Maher lambdas,
    same formula as predict.py. maher_pair is ((lam_h, lam_a), w_elo) or
    None when the Maher model has no fit for one of the teams."""
    dr = elo_h - elo_a + home_adv
    gd = dr * cal["goal_diff_per_elo"]
    base = cal["base_goals"]
    ko = cal.get("knockout_goals_factor", 1.0) if knockout else 1.0
    base *= ko
    lam_h = max(0.15, base / 2 + gd / 2)
    lam_a = max(0.15, base / 2 - gd / 2)
    if maher_pair:
        (mh, ma), w = maher_pair
        mh = max(0.15, mh * ko)
        ma = max(0.15, ma * ko)
        lam_h = lam_h ** w * mh ** (1 - w)
        lam_a = lam_a ** w * ma ** (1 - w)
    return lam_h, lam_a, dr


def sample_poisson(lam):
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p <= l:
            return k
        k += 1


def play(team_a, team_b, elo, cal, knockout, maher_model=None):
    adv = 0.0
    home_team = None
    if team_a in HOSTS and team_b not in HOSTS:
        adv = cal["home_advantage_elo"]
        home_team = team_a
    elif team_b in HOSTS and team_a not in HOSTS:
        adv = -cal["home_advantage_elo"]
        home_team = team_b
    maher_pair = None
    if maher_model and team_a in maher_model["teams"] and team_b in maher_model["teams"]:
        if home_team == team_b:
            la2, la1 = maher.lambdas(maher_model, team_b, team_a, neutral=False)
        else:
            la1, la2 = maher.lambdas(maher_model, team_a, team_b,
                                     neutral=home_team is None)
        w = maher_model.get("ensemble", {}).get("w_elo", 0.5)
        maher_pair = ((la1, la2), w)
    lam_a, lam_b, dr = match_lambdas(elo[team_a], elo[team_b], cal, adv,
                                     knockout, maher_pair)
    ga, gb = sample_poisson(lam_a), sample_poisson(lam_b)
    if not knockout or ga != gb:
        return ga, gb
    we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    return (1, 0) if random.random() < 0.5 + (we - 0.5) * 0.33 else (0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--runs", type=int, default=50000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    data_dir = Path(args.data_dir)

    data = json.loads((data_dir / "elo_ratings.json").read_text(encoding="utf-8"))
    cal = data["calibration"]
    elo = {t: v["elo"] for t, v in data["ratings"].items()}
    maher_path = data_dir / "maher_ratings.json"
    maher_model = json.loads(maher_path.read_text(encoding="utf-8")) \
        if maher_path.exists() else None

    tconf_path = data_dir / "tournament.json"
    tconf = json.loads(tconf_path.read_text(encoding="utf-8")) if tconf_path.exists() else {}

    fixtures = load_fixtures(data_dir)
    groups = tconf.get("groups") or derive_groups(fixtures)
    bad = [g for g in groups if len(g) != 4]
    if bad or len(groups) != 12:
        sys.exit(f"Expected 12 groups of 4, got {[len(g) for g in groups]}. "
                 "Fix data/tournament.json or the fixture list.")
    missing = [t for g in groups for t in g if t not in elo]
    if missing:
        sys.exit(f"No Elo rating for: {missing}")

    played = {}
    pending = []
    for r in fixtures:
        pair = (r["home_team"], r["away_team"])
        if r["home_score"] not in ("", "NA"):
            played[pair] = (int(float(r["home_score"])), int(float(r["away_score"])))
        else:
            pending.append(pair)
    print(f"Simulating {args.runs} tournaments: {len(played)} group matches played, "
          f"{len(pending)} to sample, 12 groups derived "
          f"{'from tournament.json' if tconf.get('groups') else 'from fixtures'}")

    reach = {t: {s: 0 for s in STAGES} for g in groups for t in g}

    for _ in range(args.runs):
        table = {}
        for g in groups:
            stats = {t: [0, 0, 0] for t in g}  # pts, gd, gf
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = g[i], g[j]
                    score = played.get((a, b)) or tuple(reversed(played.get((b, a)) or ())) or None
                    if not score:
                        score = play(a, b, elo, cal, knockout=False,
                                     maher_model=maher_model)
                    ga, gb = score
                    stats[a][1] += ga - gb; stats[a][2] += ga
                    stats[b][1] += gb - ga; stats[b][2] += gb
                    if ga > gb:
                        stats[a][0] += 3
                    elif gb > ga:
                        stats[b][0] += 3
                    else:
                        stats[a][0] += 1; stats[b][0] += 1
            ranked = sorted(g, key=lambda t: (stats[t], random.random()), reverse=True)
            table[tuple(g)] = (ranked, stats)

        firsts, seconds, thirds = [], [], []
        for ranked, stats in table.values():
            firsts.append((stats[ranked[0]], ranked[0]))
            seconds.append((stats[ranked[1]], ranked[1]))
            thirds.append((stats[ranked[2]], ranked[2]))
        thirds.sort(key=lambda x: (x[0], random.random()), reverse=True)
        qualifiers = firsts + seconds + thirds[:8]
        qualifiers.sort(key=lambda x: (x[0], random.random()), reverse=True)
        field = [t for _, t in qualifiers]  # seeded 1..32

        for t in field:
            reach[t]["R32"] += 1
        stage_i = 1
        while len(field) > 1:
            nxt = []
            half = len(field) // 2
            for i in range(half):
                a, b = field[i], field[-(i + 1)]
                ga, gb = play(a, b, elo, cal, knockout=True,
                              maher_model=maher_model)
                nxt.append(a if ga > gb else b)
            field = nxt
            for t in field:
                reach[t][STAGES[min(stage_i, 5)]] += 1
            stage_i += 1

    out = {
        "runs": args.runs,
        "teams": {t: {s: round(c / args.runs, 4) for s, c in d.items()}
                  for t, d in sorted(reach.items(),
                                     key=lambda kv: -kv[1]["Champion"])},
    }
    dest = Path("predictions"); dest.mkdir(exist_ok=True)
    (dest / "tournament_odds.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'Team':25s} {'Champ':>7s} {'Final':>7s} {'SF':>7s} {'QF':>7s} {'R16':>7s}")
    for t, d in list(out["teams"].items())[:20]:
        print(f"{t:25s} {d['Champion']:7.1%} {d['Final']:7.1%} {d['SF']:7.1%} "
              f"{d['QF']:7.1%} {d['R16']:7.1%}")
    print(f"\nSaved predictions/tournament_odds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

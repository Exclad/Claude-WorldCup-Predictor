#!/usr/bin/env python3
"""Predict a single match from the Elo + Maher ensemble.

Pipeline:
1. Elo gap (plus home advantage unless --neutral) -> expected goals via the
   calibrated slope and base-goals constants.
2. Maher attack/defence model (data/maher_ratings.json) -> a second,
   independent expected-goals pair for the same fixture.
3. Ensemble: geometric blend of the two lambda pairs, with the weight chosen
   on 2006-2018 World Cups and validated on 2022 (backtest.py --tune-blend).
   If the two models disagree by more than 10 points on any outcome, the
   prediction is flagged: treat the stated confidence as optimistic.
4. Full Poisson scoreline matrix with a Dixon-Coles low-score correction
   (independent Poissons underpredict draws; the correction shifts mass onto
   0-0 and 1-1) -> P(home win), P(draw), P(away win) and likely scorelines.
   Also reports the head-to-head record from the dataset for context.
5. --knockout also reports advance probabilities (draws resolved by extra
   time/penalties, modelled as a slightly strength-weighted coin flip).
6. --market H,D,A (decimal bookmaker odds) records the de-vigged market
   probabilities alongside the model's. The market is the benchmark the
   model must beat, not an input: predicted_outcome stays model-only.
   A 50/50 log-pool blend is also logged so record.py can score all three.

Optionally logs the prediction to predictions/log.json (use --log) so that
record.py can later compare it against the real result and track accuracy.

The probabilities here are the statistical baseline. Qualitative factors
(injuries, suspensions, form) are applied on top by the skill workflow and
recorded via --adjust-home/--adjust-away with --adjust-reason.
"""
import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from players import player_share, REPLACEMENT_LOSS  # noqa: E402
import maher as maher_mod  # noqa: E402

MAX_GOALS = 10
MIN_LAMBDA = 0.15
DC_RHO = -0.10  # Dixon-Coles low-score correlation (typical fitted value)
MAX_MISSING_DISCOUNT = 0.50  # a team never loses more than half its attack
DEFAULT_W_ELO = 0.5  # used only if no validated blend weight is stored
DISAGREE_PP = 0.10  # flag when the two models differ this much on an outcome

# 2026 venue altitudes (metres). Altitude over ~1200m measurably hurts
# visiting lowland teams (thinner air, ball flight, fatigue).
VENUE_ALTITUDE = {"Mexico City": 2240, "Zapopan": 1566, "Guadalajara": 1566,
                  "Monterrey": 540, "Guadalupe": 540}


def shootout_rates(data_dir):
    """Shrunk historical shootout win rates per team (prior 0.5, weight 8)."""
    import csv
    path = Path(data_dir) / "shootouts.csv"
    wins, total = {}, {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for t in (r["home_team"], r["away_team"]):
                    total[t] = total.get(t, 0) + 1
                w = r["winner"]
                wins[w] = wins.get(w, 0) + 1
    def rate(team):
        n = total.get(team, 0)
        return (wins.get(team, 0) + 4) / (n + 8), n
    return rate


def last_match_before(team, match_date, data_dir):
    import csv
    best = None
    for name in ("results.csv", "manual_results.csv"):
        path = Path(data_dir) / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["home_score"] in ("", "NA"):
                    continue
                if team in (r["home_team"], r["away_team"]) and r["date"] < match_date:
                    if best is None or r["date"] > best:
                        best = r["date"]
    return best


def poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dixon_coles_tau(h, a, lam_h, lam_a, rho=DC_RHO):
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1 + lam_a * rho
    if h == 0 and a == 1:
        return 1 + lam_h * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def head_to_head(home, away, data_dir, limit=5):
    """Last meetings between the two teams, from the results data."""
    import csv
    meetings = []
    for name in ("results.csv", "manual_results.csv"):
        path = Path(data_dir) / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["home_score"] in ("", "NA"):
                    continue
                if {r["home_team"], r["away_team"]} == {home, away}:
                    meetings.append((r["date"], r["home_team"],
                                     int(float(r["home_score"])),
                                     int(float(r["away_score"])), r["away_team"]))
    meetings.sort()
    w = d = l = 0
    for _, ht, hs, as_, _ in meetings:
        diff = hs - as_ if ht == home else as_ - hs
        if diff > 0:
            w += 1
        elif diff == 0:
            d += 1
        else:
            l += 1
    return {"total": len(meetings),
            "record": f"{home} {w}W {d}D {l}L",
            "recent": [f"{dt}: {ht} {hs}-{as_} {at}"
                       for dt, ht, hs, as_, at in meetings[-limit:]]}


def outcome_probs(lam_h, lam_a):
    """(P(home), P(draw), P(away)) from the Dixon-Coles-corrected matrix."""
    w = d = l = 0.0
    for h in range(MAX_GOALS + 1):
        ph = poisson(h, lam_h)
        for a in range(MAX_GOALS + 1):
            p = ph * poisson(a, lam_a) * dixon_coles_tau(h, a, lam_h, lam_a)
            if h > a:
                w += p
            elif h == a:
                d += p
            else:
                l += p
    s = w + d + l
    return w / s, d / s, l / s


def devig(odds):
    """Decimal odds (H,D,A) -> fair probabilities and the bookmaker margin."""
    implied = [1.0 / o for o in odds]
    s = sum(implied)
    return [p / s for p in implied], s - 1.0


def find_team(name, ratings):
    if name in ratings:
        return name
    low = {t.lower(): t for t in ratings}
    if name.lower() in low:
        return low[name.lower()]
    sugg = [t for t in ratings if name.lower() in t.lower()][:5]
    sys.exit(f"Team '{name}' not found in ratings."
             + (f" Did you mean: {', '.join(sugg)}?" if sugg else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", help="home team (or first team if --neutral)")
    parser.add_argument("away", help="away team")
    parser.add_argument("--neutral", action="store_true", help="no home advantage")
    parser.add_argument("--knockout", action="store_true",
                        help="knockout match: also report advance probabilities")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log", action="store_true",
                        help="append prediction to predictions/log.json")
    parser.add_argument("--date", default=None, help="match date YYYY-MM-DD (for the log)")
    parser.add_argument("--stage", default="", help="e.g. 'Group A', 'Round of 32' (for the log)")
    parser.add_argument("--adjust-home", type=float, default=0.0,
                        help="qualitative win-prob adjustment for home team, in "
                             "percentage points (e.g. -3 for a missing striker)")
    parser.add_argument("--adjust-away", type=float, default=0.0)
    parser.add_argument("--adjust-reason", default="",
                        help="why the adjustment was applied (required if adjusting)")
    parser.add_argument("--missing-home", action="append", default=[],
                        metavar="PLAYER", help="confirmed-absent home player; "
                        "expected goals reduced by his data-derived goal share")
    parser.add_argument("--missing-away", action="append", default=[], metavar="PLAYER")
    parser.add_argument("--city", default=None,
                        help="venue city, to flag altitude effects (e.g. 'Mexico City')")
    parser.add_argument("--market", default=None, metavar="H,D,A",
                        help="decimal bookmaker odds home,draw,away — logged as "
                             "the benchmark the model is scored against")
    args = parser.parse_args()

    if (args.adjust_home or args.adjust_away) and not args.adjust_reason:
        sys.exit("Adjustments require --adjust-reason explaining the evidence.")

    data = json.loads((Path(args.data_dir) / "elo_ratings.json").read_text(encoding="utf-8"))
    ratings, cal = data["ratings"], data["calibration"]

    home = find_team(args.home, ratings)
    away = find_team(args.away, ratings)
    eh, ea = ratings[home]["elo"], ratings[away]["elo"]

    dr = eh - ea + (0 if args.neutral else cal["home_advantage_elo"])
    we = 1.0 / (1.0 + 10 ** (-dr / 400.0))

    exp_gd = dr * cal["goal_diff_per_elo"]
    base = cal["base_goals"]
    ko_factor = cal.get("knockout_goals_factor", 1.0) if args.knockout else 1.0
    base *= ko_factor
    lam_eh = max(MIN_LAMBDA, base / 2 + exp_gd / 2)
    lam_ea = max(MIN_LAMBDA, base / 2 - exp_gd / 2)

    # second model: Maher attack/defence (independent view of the same fixture)
    maher_path = Path(args.data_dir) / "maher_ratings.json"
    maher_model = json.loads(maher_path.read_text(encoding="utf-8")) \
        if maher_path.exists() else None
    components = None
    if maher_model and home in maher_model["teams"] and away in maher_model["teams"]:
        lam_mh, lam_ma = maher_mod.lambdas(maher_model, home, away, args.neutral)
        lam_mh = max(MIN_LAMBDA, lam_mh * ko_factor)
        lam_ma = max(MIN_LAMBDA, lam_ma * ko_factor)
        w_elo = maher_model.get("ensemble", {}).get("w_elo", DEFAULT_W_ELO)
        lam_h = lam_eh ** w_elo * lam_mh ** (1 - w_elo)
        lam_a = lam_ea ** w_elo * lam_ma ** (1 - w_elo)
        pe = outcome_probs(lam_eh, lam_ea)
        pm = outcome_probs(lam_mh, lam_ma)
        spread = max(abs(a - b) for a, b in zip(pe, pm))
        components = {
            "elo": {"home_win": round(pe[0], 4), "draw": round(pe[1], 4),
                    "away_win": round(pe[2], 4)},
            "maher": {"home_win": round(pm[0], 4), "draw": round(pm[1], 4),
                      "away_win": round(pm[2], 4),
                      "expected_goals": [round(lam_mh, 2), round(lam_ma, 2)]},
            "w_elo": w_elo,
            "max_disagreement": round(spread, 3),
            "models_disagree": spread > DISAGREE_PP,
        }
    else:
        lam_h, lam_a = lam_eh, lam_ea  # Elo only; Maher has no fit for a team

    # data-derived discount for confirmed-absent players
    missing_report = []
    for side, team, names in (("home", home, args.missing_home),
                              ("away", away, args.missing_away)):
        discount = 0.0
        for raw in names:
            name, share, _ = player_share(team, raw, args.data_dir)
            loss = share * REPLACEMENT_LOSS
            discount += loss
            missing_report.append({"side": side, "player": name or raw,
                                   "goal_share": round(share, 3),
                                   "xg_discount": round(loss, 3)})
        discount = min(discount, MAX_MISSING_DISCOUNT)
        if side == "home":
            lam_h = max(MIN_LAMBDA, lam_h * (1 - discount))
        else:
            lam_a = max(MIN_LAMBDA, lam_a * (1 - discount))

    p_win = p_draw = p_loss = 0.0
    scorelines = []
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson(h, lam_h) * poisson(a, lam_a) \
                * dixon_coles_tau(h, a, lam_h, lam_a)
            scorelines.append((p, h, a))
            if h > a:
                p_win += p
            elif h == a:
                p_draw += p
            else:
                p_loss += p
    total = p_win + p_draw + p_loss
    p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total

    # bounded qualitative adjustment: shift win probs, keep draw, renormalise
    adj_h = max(-10, min(10, args.adjust_home)) / 100.0
    adj_a = max(-10, min(10, args.adjust_away)) / 100.0
    if adj_h or adj_a:
        p_win = max(0.01, p_win + adj_h - adj_a)
        p_loss = max(0.01, p_loss + adj_a - adj_h)
        s = p_win + p_draw + p_loss
        p_win, p_draw, p_loss = p_win / s, p_draw / s, p_loss / s

    scorelines.sort(reverse=True)
    top = [{"score": f"{h}-{a}", "prob": round(p / total, 4)} for p, h, a in scorelines[:5]]

    result = {
        "date": args.date or date.today().isoformat(),
        "stage": args.stage,
        "home": home, "away": away,
        "neutral": args.neutral, "knockout": args.knockout,
        "elo": {"home": eh, "away": ea, "diff_with_adv": round(dr, 1)},
        "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
        "probabilities": {"home_win": round(p_win, 4), "draw": round(p_draw, 4),
                          "away_win": round(p_loss, 4)},
        "top_scorelines": top,
        "predicted_outcome": max([("home_win", p_win), ("draw", p_draw),
                                  ("away_win", p_loss)], key=lambda x: x[1])[0],
        "adjustment": ({"home_pp": args.adjust_home, "away_pp": args.adjust_away,
                        "reason": args.adjust_reason}
                       if (adj_h or adj_a) else None),
        "missing_players": missing_report or None,
        "head_to_head": head_to_head(home, away, args.data_dir),
        "components": components,
    }

    # bookmaker odds: the benchmark, never an input to predicted_outcome
    if args.market:
        odds = [float(x) for x in args.market.split(",")]
        if len(odds) != 3 or min(odds) <= 1.0:
            sys.exit("--market needs three decimal odds > 1.0: home,draw,away")
        fair, margin = devig(odds)
        # 50/50 log opinion pool of model and market, scored separately
        pool = [(p * q) ** 0.5 for p, q in zip((p_win, p_draw, p_loss), fair)]
        s = sum(pool)
        result["market"] = {
            "odds": odds, "overround": round(margin, 4),
            "probabilities": {"home_win": round(fair[0], 4),
                              "draw": round(fair[1], 4),
                              "away_win": round(fair[2], 4)},
            "blend": {"home_win": round(pool[0] / s, 4),
                      "draw": round(pool[1] / s, 4),
                      "away_win": round(pool[2] / s, 4)},
        }

    # context: rest days and altitude (reported, not folded into the number)
    match_date = args.date or date.today().isoformat()
    rest = {}
    for side, team in (("home", home), ("away", away)):
        last = last_match_before(team, match_date, args.data_dir)
        if last:
            days = (date.fromisoformat(match_date) - date.fromisoformat(last)).days
            rest[side] = days
    result["rest_days"] = rest or None
    alt = VENUE_ALTITUDE.get(args.city or "", 0)
    result["altitude_m"] = alt if alt >= 1200 else None

    if args.knockout:
        # strength lean, then tilt by each side's historical shootout record
        lean = 0.5 + (we - 0.5) * 0.33
        rate = shootout_rates(args.data_dir)
        sh, nh = rate(home)
        sa, na = rate(away)
        pen_home = (lean * sh) / (lean * sh + (1 - lean) * sa)
        result["advance"] = {"home": round(p_win + p_draw * pen_home, 4),
                             "away": round(p_loss + p_draw * (1 - pen_home), 4),
                             "shootout_history": {home: f"{sh:.0%} shrunk ({nh} shootouts)",
                                                  away: f"{sa:.0%} shrunk ({na} shootouts)"}}

    venue = "neutral venue" if args.neutral else f"{home} at home"
    print(f"\n{home} ({eh:.0f}) vs {away} ({ea:.0f}) — {venue}")
    print(f"Elo gap incl. advantage: {dr:+.0f}; "
          f"ensemble expected goals {lam_h:.2f} : {lam_a:.2f}")
    if components:
        c = components
        print(f"Elo model:   {c['elo']['home_win']:.0%}/{c['elo']['draw']:.0%}/"
              f"{c['elo']['away_win']:.0%}  ·  Maher model: "
              f"{c['maher']['home_win']:.0%}/{c['maher']['draw']:.0%}/"
              f"{c['maher']['away_win']:.0%}  (w_elo={c['w_elo']})")
        if c["models_disagree"]:
            print(f"!! Models disagree by {c['max_disagreement']:.0%} — "
                  "treat the confidence below as optimistic")
    else:
        print("Maher ratings unavailable for this pairing — Elo only")
    print(f"P({home} win)  {p_win:6.1%}")
    print(f"P(draw)        {p_draw:6.1%}")
    print(f"P({away} win)  {p_loss:6.1%}")
    if args.market and result.get("market"):
        mk = result["market"]["probabilities"]
        bl = result["market"]["blend"]
        print(f"Market (de-vigged): {mk['home_win']:.0%}/{mk['draw']:.0%}/"
              f"{mk['away_win']:.0%}  ·  model+market blend: "
              f"{bl['home_win']:.0%}/{bl['draw']:.0%}/{bl['away_win']:.0%}")
    if args.knockout:
        print(f"Advance: {home} {result['advance']['home']:.1%} / "
              f"{away} {result['advance']['away']:.1%}")
    print("Most likely scorelines: "
          + ", ".join(f"{s['score']} ({s['prob']:.1%})" for s in top[:3]))
    h2h = result["head_to_head"]
    if h2h["total"]:
        print(f"Head-to-head ({h2h['total']} meetings): {h2h['record']}; last: "
              + "; ".join(h2h["recent"][-3:]))
    else:
        print("Head-to-head: first ever meeting")
    for m in missing_report:
        print(f"Missing ({m['side']}): {m['player']} — {m['goal_share']:.0%} of recent "
              f"goals, xG discounted {m['xg_discount']:.0%}")
    if rest:
        print(f"Rest days: {home} {rest.get('home', '?')}, {away} {rest.get('away', '?')}")
    if result["altitude_m"]:
        print(f"Altitude: {args.city} at {result['altitude_m']}m — historically "
              "tough on visiting lowland teams; consider in the qualitative layer")
    if result["adjustment"]:
        print(f"Adjustment applied: {result['adjustment']}")

    if args.log:
        log_path = Path("predictions/log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        result["logged_at"] = date.today().isoformat()
        result["actual"] = None
        log.append(result)
        log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Logged to {log_path} ({len(log)} predictions on record)")

    print()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

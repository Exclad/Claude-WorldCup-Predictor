#!/usr/bin/env python3
"""Walk-forward backtest: how would this model have done at past World Cups?

Replays all of history chronologically, maintaining Elo ratings exactly as
elo.py does. Whenever a match belongs to a target tournament (default: the
2014, 2018 and 2022 World Cup finals), the model first predicts it from the
ratings as they stood that day — no peeking — records accuracy and Brier,
and only then updates the ratings with the real result.

Baselines for comparison:
- uniform guess: Brier 0.667
- "coin-flip favourite": pick the higher-Elo side with naive 45/27.5/27.5

This is the evidence that the system works (or a regression alarm when the
model is changed). Run after any change to K factors, calibration or the
Poisson layer.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from elo import (START_ELO, HOME_ADV, k_factor, goal_multiplier,  # noqa: E402
                 load_matches)
import maher  # noqa: E402

DC_RHO = -0.10
MAX_GOALS = 10
TRAIN_YEARS, TEST_YEARS = {1998, 2002, 2006, 2010, 2014, 2018}, {2022}


def replay(matches, years, slope, base, home_adv=HOME_ADV, k_scale=1.0, rho=DC_RHO):
    """Walk-forward replay; returns {year: [(hit, brier), ...]}."""
    global DC_RHO
    old_rho, DC_RHO = DC_RHO, rho
    ratings = {}
    evals = {y: [] for y in years}
    try:
        for m in matches:
            year = int(m["date"][:4])
            for t in (m["home"], m["away"]):
                ratings.setdefault(t, START_ELO)
            dr = ratings[m["home"]] - ratings[m["away"]]
            if not m["neutral"]:
                dr += home_adv
            gd = m["hs"] - m["as"]
            if m["tournament"] == "FIFA World Cup" and year in years:
                w, d, l = predict_probs(dr, slope, base)
                outcome = "w" if gd > 0 else ("d" if gd == 0 else "l")
                obs = {"w": 0, "d": 0, "l": 0}
                obs[outcome] = 1
                brier = (w - obs["w"]) ** 2 + (d - obs["d"]) ** 2 + (l - obs["l"]) ** 2
                pick = max((("w", w), ("d", d), ("l", l)), key=lambda x: x[1])[0]
                evals[year].append((pick == outcome, brier))
            we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
            result = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            delta = k_scale * k_factor(m["tournament"]) * goal_multiplier(gd) * (result - we)
            ratings[m["home"]] += delta
            ratings[m["away"]] -= delta
    finally:
        DC_RHO = old_rho
    return evals


def tune(matches, slope, base):
    """Grid-search hyperparameters on 2006-2018 WCs, validate on 2022.

    The split matters: picking parameters on the same tournaments you report
    is overfitting. Train Brier chooses the candidate; the 2022 column is the
    honest out-of-sample check.
    """
    train_years, test_years = {2006, 2010, 2014, 2018}, {2022}
    print(f"{'home_adv':>8s} {'k_scale':>8s} {'rho':>7s} {'train Brier':>12s} {'test Brier':>11s}")
    results = []
    for home_adv in (60, 80, 100, 120):
        for k_scale in (0.8, 1.0, 1.2):
            for rho in (-0.05, -0.10, -0.15):
                ev = replay(matches, train_years | test_years, slope, base,
                            home_adv, k_scale, rho)
                tr = [b for y in train_years for _, b in ev[y]]
                te = [b for y in test_years for _, b in ev[y]]
                tr_b = sum(tr) / len(tr)
                te_b = sum(te) / len(te)
                results.append((tr_b, te_b, home_adv, k_scale, rho))
                print(f"{home_adv:8d} {k_scale:8.1f} {rho:7.2f} {tr_b:12.4f} {te_b:11.4f}")
    results.sort()
    tr_b, te_b, ha, ks, rho = results[0]
    print(f"\nBest by train Brier: home_adv={ha}, k_scale={ks}, rho={rho} "
          f"(train {tr_b:.4f}, test {te_b:.4f})")
    print("Apply only if the test column agrees; a train win with a test loss "
          "is noise, keep the defaults.")


def probs_from_lambdas(lam_h, lam_a, rho=None):
    rho = DC_RHO if rho is None else rho
    def pois(k, lam):
        return math.exp(-lam) * lam ** k / math.factorial(k)
    def tau(h, a):
        if h == 0 and a == 0:
            return 1 - lam_h * lam_a * rho
        if h == 1 and a == 0:
            return 1 + lam_a * rho
        if h == 0 and a == 1:
            return 1 + lam_h * rho
        if h == 1 and a == 1:
            return 1 - rho
        return 1.0
    w = d = l = 0.0
    for h in range(MAX_GOALS + 1):
        ph = pois(h, lam_h)
        for a in range(MAX_GOALS + 1):
            p = ph * pois(a, lam_a) * tau(h, a)
            if h > a:
                w += p
            elif h == a:
                d += p
            else:
                l += p
    s = w + d + l
    return w / s, d / s, l / s


def predict_probs(dr, slope, base):
    lam_h = max(0.15, base / 2 + dr * slope / 2)
    lam_a = max(0.15, base / 2 - dr * slope / 2)
    return probs_from_lambdas(lam_h, lam_a)


def score(probs, outcome):
    """(hit, brier) for a (w,d,l) probability triple vs outcome 'w'/'d'/'l'."""
    w, d, l = probs
    obs = {"w": 0.0, "d": 0.0, "l": 0.0}
    obs[outcome] = 1.0
    brier = (w - obs["w"]) ** 2 + (d - obs["d"]) ** 2 + (l - obs["l"]) ** 2
    pick = max((("w", w), ("d", d), ("l", l)), key=lambda x: x[1])[0]
    return pick == outcome, brier


def collect_records(matches, years, slope, base, decay=maher.DECAY_PER_YEAR,
                    prior_n=maher.PRIOR_N, friendly_weight=maher.FRIENDLY_WEIGHT):
    """Walk-forward, no peeking: for every WC match in `years`, record both
    models' expected goals as of that day plus the real outcome.

    Elo updates match by match; the Maher model is refit once at each
    tournament's first match (a fit is expensive, and squad strength barely
    moves within a four-week tournament).

    Returns [(year, lam_eh, lam_ea, lam_mh, lam_ma, outcome), ...] so any
    ensemble blend can be evaluated afterwards without refitting anything.
    """
    ratings = {}
    records = []
    fits = {}
    for m in matches:
        year = int(m["date"][:4])
        for t in (m["home"], m["away"]):
            ratings.setdefault(t, START_ELO)
        dr = ratings[m["home"]] - ratings[m["away"]]
        if not m["neutral"]:
            dr += HOME_ADV
        gd = m["hs"] - m["as"]
        if m["tournament"] == "FIFA World Cup" and year in years:
            if year not in fits:
                fits[year] = maher.fit(matches, as_of=m["date"], decay=decay,
                                       prior_n=prior_n,
                                       friendly_weight=friendly_weight)
            lam_eh = max(0.15, base / 2 + dr * slope / 2)
            lam_ea = max(0.15, base / 2 - dr * slope / 2)
            f = fits[year]
            if m["home"] in f["teams"] and m["away"] in f["teams"]:
                lam_mh, lam_ma = maher.lambdas(f, m["home"], m["away"], m["neutral"])
                outcome = "w" if gd > 0 else ("d" if gd == 0 else "l")
                records.append((year, lam_eh, lam_ea,
                                max(0.15, lam_mh), max(0.15, lam_ma), outcome))
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        result = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = k_factor(m["tournament"]) * goal_multiplier(gd) * (result - we)
        ratings[m["home"]] += delta
        ratings[m["away"]] -= delta
    return records


def eval_blend(records, w_elo, years=None):
    """Mean (accuracy, brier) of the geometric lambda blend over records."""
    hits = briers = n = 0
    for year, leh, lea, lmh, lma, outcome in records:
        if years and year not in years:
            continue
        lam_h = leh ** w_elo * lmh ** (1 - w_elo)
        lam_a = lea ** w_elo * lma ** (1 - w_elo)
        hit, brier = score(probs_from_lambdas(lam_h, lam_a), outcome)
        hits += hit
        briers += brier
        n += 1
    return (hits / n, briers / n, n) if n else (0.0, 0.0, 0)


def tune_maher(matches, slope, base):
    """Grid-search Maher fit settings on 2006-2018, validate on 2022.

    Scored as the pure Maher model (w_elo=0) so the knobs are chosen on this
    model's own merit, not on how well they hide behind the Elo half.
    """
    print(f"{'decay':>6s} {'prior':>6s} {'frnd_w':>7s} {'train Brier':>12s} {'test Brier':>11s}")
    results = []
    for decay in (0.1, 0.2, 0.3, 0.4):
        for prior_n in (4.0, 8.0, 16.0):
            for fw in (0.5, 1.0):
                rec = collect_records(matches, TRAIN_YEARS | TEST_YEARS, slope,
                                      base, decay, prior_n, fw)
                _, tr_b, _ = eval_blend(rec, 0.0, TRAIN_YEARS)
                _, te_b, _ = eval_blend(rec, 0.0, TEST_YEARS)
                results.append((tr_b, te_b, decay, prior_n, fw))
                print(f"{decay:6.1f} {prior_n:6.0f} {fw:7.1f} {tr_b:12.4f} {te_b:11.4f}")
    results.sort()
    tr_b, te_b, decay, prior_n, fw = results[0]
    print(f"\nBest by train Brier: decay={decay}, prior_n={prior_n}, "
          f"friendly_weight={fw} (train {tr_b:.4f}, test {te_b:.4f})")
    print("Apply only if the test column agrees; update the defaults in maher.py.")


def tune_blend(matches, slope, base, data_dir, write):
    """Choose the Elo/Maher blend weight on 2006-2018, validate on 2022.

    With --write, stores the train-chosen weight (and its honest test score)
    in data/maher_ratings.json for predict.py and simulate.py to use.
    """
    rec = collect_records(matches, TRAIN_YEARS | TEST_YEARS, slope, base)
    print(f"{'w_elo':>6s} {'train acc':>10s} {'train Brier':>12s} "
          f"{'test acc':>9s} {'test Brier':>11s}")
    results = []
    for w in [i / 10 for i in range(11)]:
        tr_a, tr_b, _ = eval_blend(rec, w, TRAIN_YEARS)
        te_a, te_b, _ = eval_blend(rec, w, TEST_YEARS)
        results.append((tr_b, w, te_b))
        print(f"{w:6.1f} {tr_a:10.1%} {tr_b:12.4f} {te_a:9.1%} {te_b:11.4f}")
    results.sort()
    tr_b, w, te_b = results[0]
    print(f"\nBest by train Brier: w_elo={w} (train {tr_b:.4f}, test {te_b:.4f})")
    if write:
        path = Path(data_dir) / "maher_ratings.json"
        model = json.loads(path.read_text(encoding="utf-8"))
        model["ensemble"] = {"w_elo": w, "chosen_on": sorted(TRAIN_YEARS),
                             "validated_on": sorted(TEST_YEARS),
                             "train_brier": round(tr_b, 4),
                             "test_brier": round(te_b, 4)}
        path.write_text(json.dumps(model, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        print(f"Wrote ensemble weight to {path}")


def compare_models(matches, years, slope, base, data_dir):
    """Side-by-side walk-forward scores: Elo vs Maher vs the stored blend."""
    path = Path(data_dir) / "maher_ratings.json"
    w_blend = 0.5
    if path.exists():
        w_blend = json.loads(path.read_text(encoding="utf-8")) \
            .get("ensemble", {}).get("w_elo", 0.5)
    rec = collect_records(matches, years, slope, base)
    print(f"Blend uses stored w_elo={w_blend}\n")
    print(f"{'WC':6s} {'n':>4s} | {'Elo acc':>8s} {'Brier':>7s} | "
          f"{'Maher acc':>9s} {'Brier':>7s} | {'Blend acc':>9s} {'Brier':>7s}")
    for y in sorted(years) + [None]:
        sel = {y} if y else years
        ea, eb, n = eval_blend(rec, 1.0, sel)
        ma, mb, _ = eval_blend(rec, 0.0, sel)
        ba, bb, _ = eval_blend(rec, w_blend, sel)
        label = str(y) if y else "TOTAL"
        if n:
            print(f"{label:6s} {n:4d} | {ea:8.1%} {eb:7.4f} | "
                  f"{ma:9.1%} {mb:7.4f} | {ba:9.1%} {bb:7.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--years", default="2014,2018,2022",
                        help="World Cup years to evaluate")
    parser.add_argument("--slope", type=float, default=0.00545)
    parser.add_argument("--base-goals", type=float, default=2.47)
    parser.add_argument("--tune", action="store_true",
                        help="grid-search home_adv/k_scale/rho (train 2006-2018, test 2022)")
    parser.add_argument("--tune-maher", action="store_true",
                        help="grid-search Maher decay/prior/friendly-weight "
                             "(train 2006-2018, test 2022)")
    parser.add_argument("--tune-blend", action="store_true",
                        help="choose the Elo/Maher ensemble weight "
                             "(train 2006-2018, test 2022)")
    parser.add_argument("--write", action="store_true",
                        help="with --tune-blend: store the weight in maher_ratings.json")
    parser.add_argument("--models", action="store_true",
                        help="compare Elo vs Maher vs blend on --years")
    args = parser.parse_args()
    years = {int(y) for y in args.years.split(",")}

    matches = load_matches(Path(args.data_dir))
    if not matches:
        sys.exit("No data. Run fetch_data.py first.")

    if args.tune:
        tune(matches, args.slope, args.base_goals)
        return 0
    if args.tune_maher:
        tune_maher(matches, args.slope, args.base_goals)
        return 0
    if args.tune_blend:
        tune_blend(matches, args.slope, args.base_goals, args.data_dir, args.write)
        return 0
    if args.models:
        compare_models(matches, years, args.slope, args.base_goals, args.data_dir)
        return 0

    evals = replay(matches, years, args.slope, args.base_goals)
    print(f"{'WC':6s} {'n':>4s} {'accuracy':>9s} {'Brier':>7s}")
    all_rows = []
    for y in sorted(years):
        rows = evals[y]
        if not rows:
            print(f"{y:<6d} no matches found")
            continue
        all_rows += rows
        acc = sum(r[0] for r in rows) / len(rows)
        br = sum(r[1] for r in rows) / len(rows)
        print(f"{y:<6d} {len(rows):4d} {acc:9.1%} {br:7.4f}")
    if all_rows:
        acc = sum(r[0] for r in all_rows) / len(all_rows)
        br = sum(r[1] for r in all_rows) / len(all_rows)
        print(f"{'TOTAL':6s} {len(all_rows):4d} {acc:9.1%} {br:7.4f}   "
              f"(uniform-guess Brier: 0.6667)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

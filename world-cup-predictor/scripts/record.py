#!/usr/bin/env python3
"""Resolve logged predictions against real results and report accuracy.

For every prediction in predictions/log.json that has no actual result yet,
look up the match in the results data (results.csv + manual_results.csv,
matching teams within +/-2 days of the predicted date). When found, store the
actual score/outcome and the Brier score, then print an accuracy report.

Brier score here is the 3-outcome version: sum of squared differences between
the predicted probability vector (win/draw/loss) and the observed outcome.
0 = perfect, 2 = maximally wrong; always predicting 1/3-1/3-1/3 scores 0.667.

Run this at the start of every session — it is the feedback loop. Resolved
results also flow back into the Elo ratings the next time elo.py runs.
"""
import csv
import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path


def load_results(data_dir: Path):
    results = {}
    for name in ("results.csv", "manual_results.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["home_score"] in ("", "NA"):
                    continue
                key = (row["date"], row["home_team"], row["away_team"])
                results[key] = (int(float(row["home_score"])), int(float(row["away_score"])))
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log", default="predictions/log.json")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print("No predictions logged yet.")
        return 0
    log = json.loads(log_path.read_text(encoding="utf-8"))
    results = load_results(Path(args.data_dir))

    newly = 0
    for pred in log:
        if pred.get("actual"):
            continue
        base = datetime.strptime(pred["date"], "%Y-%m-%d")
        for delta in range(-2, 3):
            d = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
            for home, away, flip in ((pred["home"], pred["away"], False),
                                     (pred["away"], pred["home"], True)):
                score = results.get((d, home, away))
                if score is None:
                    continue
                hs, as_ = (score[1], score[0]) if flip else score
                outcome = "home_win" if hs > as_ else ("draw" if hs == as_ else "away_win")
                p = pred["probabilities"]
                obs = {"home_win": 0, "draw": 0, "away_win": 0}
                obs[outcome] = 1
                brier = sum((p[k] - obs[k]) ** 2 for k in obs)
                pred["actual"] = {"score": f"{hs}-{as_}", "outcome": outcome,
                                  "outcome_correct": outcome == pred["predicted_outcome"],
                                  "brier": round(brier, 4)}
                # score the benchmarks recorded at prediction time too
                if pred.get("market"):
                    for src in ("probabilities", "blend"):
                        q = pred["market"][src]
                        b = sum((q[k] - obs[k]) ** 2 for k in obs)
                        key = "market_brier" if src == "probabilities" else "blend_brier"
                        pred["actual"][key] = round(b, 4)
                newly += 1
                break
            if pred.get("actual"):
                break

    log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")

    resolved = [p for p in log if p.get("actual")]
    pending = [p for p in log if not p.get("actual")]
    print(f"Predictions on record: {len(log)} "
          f"({len(resolved)} resolved, {len(pending)} pending, {newly} newly resolved)")
    if resolved:
        correct = sum(p["actual"]["outcome_correct"] for p in resolved)
        brier = sum(p["actual"]["brier"] for p in resolved) / len(resolved)
        print(f"Outcome accuracy: {correct}/{len(resolved)} = {correct/len(resolved):.1%}")
        print(f"Mean Brier score: {brier:.4f}  (0=perfect, 0.667=always 33/33/33, 2=worst)")
        # model vs market scoreboard, on the subset where odds were recorded
        with_mkt = [p for p in resolved if p["actual"].get("market_brier") is not None]
        if with_mkt:
            n = len(with_mkt)
            mb = sum(p["actual"]["brier"] for p in with_mkt) / n
            kb = sum(p["actual"]["market_brier"] for p in with_mkt) / n
            bb = sum(p["actual"]["blend_brier"] for p in with_mkt) / n
            verdict = "model ahead" if mb < kb else "market ahead"
            print(f"Vs bookmakers ({n} matches with recorded odds): "
                  f"model {mb:.4f} | market {kb:.4f} | 50/50 blend {bb:.4f} "
                  f"-> {verdict}")
        print("\nRecent resolved:")
        for p in resolved[-10:]:
            mark = "+" if p["actual"]["outcome_correct"] else "x"
            adj = " [adjusted]" if p.get("adjustment") else ""
            print(f" {mark} {p['date']} {p['home']} {p['actual']['score']} {p['away']} — "
                  f"predicted {p['predicted_outcome']} "
                  f"(P={p['probabilities'][p['predicted_outcome']]:.0%}), "
                  f"brier {p['actual']['brier']:.3f}{adj}")
    if pending:
        print("\nPending:")
        for p in pending:
            print(f" ? {p['date']} {p['home']} vs {p['away']} — "
                  f"predicted {p['predicted_outcome']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

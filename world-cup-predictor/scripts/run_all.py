#!/usr/bin/env python3
"""One-command pipeline — runs the whole system without any AI involved.

  python3 run_all.py [--days 2] [--simulate] [--data-dir data]

Steps: fetch latest data -> rebuild Elo -> resolve past predictions ->
predict & log every World Cup fixture in the next N days (skipping ones
already logged) -> optionally re-simulate the tournament -> regenerate the
dashboard. Suitable for cron, e.g. daily at 08:00:

  0 8 * * * cd /path/to/project && python3 /path/to/skill/scripts/run_all.py --simulate

What this standalone mode cannot do (the Claude session workflow adds it):
fill the 1-2 day dataset lag from news sources, research injuries/lineups
for qualitative adjustments, or compare against bookmaker odds. Predictions
are the pure statistical baseline.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent


def run(script, *extra):
    cmd = [sys.executable, str(HERE / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=2,
                        help="predict fixtures up to N days ahead (default 2)")
    parser.add_argument("--simulate", action="store_true",
                        help="also re-run the 50k tournament simulation")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if run("fetch_data.py", "--data-dir", args.data_dir):
        print("fetch failed — continuing with existing data", file=sys.stderr)
    for step in (("elo.py",), ("maher.py",), ("record.py",)):
        if run(*step, "--data-dir", args.data_dir):
            return 1

    log_path = Path("predictions/log.json")
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    logged = {(p["date"], p["home"], p["away"]) for p in log}

    today = date.today()
    horizon = (today + timedelta(days=args.days)).isoformat()
    fixtures = []
    with open(data_dir / "results.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["tournament"] == "FIFA World Cup"
                    and r["home_score"] in ("", "NA")
                    and today.isoformat() <= r["date"] <= horizon
                    and (r["date"], r["home_team"], r["away_team"]) not in logged):
                fixtures.append(r)

    print(f"\n{len(fixtures)} unlogged fixture(s) through {horizon}")
    for r in fixtures:
        extra = ["--stage", "World Cup", "--date", r["date"], "--log",
                 "--city", r["city"], "--data-dir", args.data_dir]
        if r["neutral"].strip().upper() == "TRUE":
            extra.append("--neutral")
        # group stage ends 2026-06-27; later matches are knockouts
        if r["date"] > "2026-06-27":
            extra.append("--knockout")
        run("predict.py", r["home_team"], r["away_team"], *extra)

    if args.simulate:
        run("simulate.py", "-n", "50000", "--data-dir", args.data_dir)
    run("dashboard.py", "--data-dir", args.data_dir)
    print("\nDone. Open predictions/dashboard.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

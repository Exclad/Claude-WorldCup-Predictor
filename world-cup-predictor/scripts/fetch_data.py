#!/usr/bin/env python3
"""Download the international results dataset (martj42/international_results).

Writes data/results.csv. Reports the latest completed match date so the caller
knows whether the dataset is fresh enough or whether recent results need to be
added manually to data/manual_results.csv.
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="directory for data files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE}/{name}"
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, data_dir / name)
        except Exception as e:
            if name == "results.csv":
                raise
            print(f"  warning: {name} failed ({e}) — player/shootout features "
                  "will use the previous copy if one exists")
    dest = data_dir / "results.csv"

    rows = 0
    latest_played = ""
    upcoming = 0
    with open(dest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            if row["home_score"] in ("", "NA"):
                upcoming += 1
            elif row["date"] > latest_played:
                latest_played = row["date"]

    print(f"Saved {dest}: {rows} matches")
    print(f"Latest completed match: {latest_played}")
    print(f"Upcoming fixtures (no score yet): {upcoming}")

    manual = data_dir / "manual_results.csv"
    if not manual.exists():
        with open(manual, "w", newline="", encoding="utf-8") as f:
            f.write("date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n")
        print(f"Created empty {manual} — append very recent results here (same columns) "
              "when the dataset lags behind real life.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

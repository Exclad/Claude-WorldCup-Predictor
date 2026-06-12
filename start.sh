#!/usr/bin/env bash
# World Cup Predictor - run the pipeline once, then serve the dashboard
# at http://localhost:8000/ with the one-click re-run button.
set -e
cd "$(dirname "$0")"
python3 world-cup-predictor/scripts/run_all.py --days 2 --simulate
python3 world-cup-predictor/scripts/serve.py

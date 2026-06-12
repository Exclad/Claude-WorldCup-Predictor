@echo off
rem World Cup Predictor - run the pipeline once, then serve the dashboard
rem at http://localhost:8000/ with the one-click re-run button.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install it from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)
python world-cup-predictor\scripts\run_all.py --days 2 --simulate
python world-cup-predictor\scripts\serve.py
pause

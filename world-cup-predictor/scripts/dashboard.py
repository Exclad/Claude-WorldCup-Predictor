#!/usr/bin/env python3
"""Generate predictions/dashboard.html — a self-contained visual summary.

Sections: model record (accuracy, Brier, calibration), upcoming predictions,
resolved predictions vs reality, championship odds, Elo top 20 with movement
since the previous snapshot. Pure static HTML/CSS, no external dependencies;
just open the file in a browser. Regenerate at the end of every session.
"""
import html
import json
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

CSS = """
:root { --bg:#0f1419; --card:#1a2129; --ink:#e6e1d7; --dim:#8a94a0;
        --green:#4ade80; --red:#f87171; --gold:#fbbf24; --blue:#60a5fa; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--ink); padding:24px;
       font:15px/1.5 "Segoe UI", system-ui, sans-serif; max-width:1080px; margin:auto; }
h1 { font-size:26px; margin-bottom:4px; }
h2 { font-size:18px; margin:28px 0 12px; color:var(--gold); }
.sub { color:var(--dim); font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }
.card { background:var(--card); border-radius:10px; padding:14px 16px; }
.match { font-weight:600; font-size:16px; }
.stage { color:var(--dim); font-size:12px; }
.bar { display:flex; height:22px; border-radius:5px; overflow:hidden;
       margin:10px 0 4px; font-size:11px; font-weight:600; color:#0f1419; }
.bar div { display:flex; align-items:center; justify-content:center; min-width:0; }
.bh { background:var(--green); } .bd { background:var(--dim); } .ba { background:var(--blue); }
.legend { display:flex; justify-content:space-between; font-size:12px; color:var(--dim); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid #2a3340; }
th { color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase; }
.ok { color:var(--green); font-weight:700; } .bad { color:var(--red); font-weight:700; }
.num { text-align:right; } td.num { font-variant-numeric:tabular-nums; }
.statrow { display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; }
.stat { background:var(--card); border-radius:10px; padding:12px 18px; }
.stat b { font-size:22px; display:block; }
.hbar { background:var(--gold); height:14px; border-radius:3px; display:inline-block;
        vertical-align:middle; }
.up { color:var(--green); } .down { color:var(--red); }
.note { font-size:12px; color:var(--dim); margin-top:6px; }
.controls { background:var(--card); border-radius:10px; padding:16px; margin-top:16px;
            display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; }
button { background:var(--gold); color:#0f1419; border:0; border-radius:8px;
         padding:10px 18px; font:600 14px/1 inherit; cursor:pointer; }
button:disabled { opacity:.5; cursor:wait; }
.prompt { background:#0b0f14; border:1px solid #2a3340; border-radius:8px;
          padding:10px 12px; font:13px/1.5 ui-monospace, monospace; flex:1;
          min-width:280px; }
#status { font-size:13px; color:var(--dim); margin-top:8px; white-space:pre-wrap; }
"""

CLAUDE_PROMPT = (
    "Use the world-cup-predictor skill (world-cup-predictor/SKILL.md in this "
    "project) and follow its Session workflow steps 1-7 in order, exactly as "
    "written there. Non-negotiable rules: "
    "(1) never invent a score — only add results to data/manual_results.csv "
    "that a web search explicitly confirms, using the dataset's exact team "
    "names; (2) rebuild BOTH ratings (elo.py then maher.py) before any "
    "prediction; (3) for every upcoming match, web-search current decimal "
    "bookmaker odds and pass them as --market H,D,A on the logged run; "
    "(4) research injuries/suspensions/probable lineups; quantify confirmed "
    "scorer absences with --missing-home/--missing-away, use --adjust-* only "
    "with a documented evidence-based --adjust-reason, and apply no "
    "adjustment when there is no solid news; (5) log exactly one final "
    "prediction per match (only the last run gets --log); (6) finish with "
    "simulate.py -n 50000 and dashboard.py, then report each prediction with "
    "its probabilities, the market comparison, and the model's running "
    "accuracy record.")

CONTROLS = f"""
<h2>Update</h2>
<div class="controls">
  <div>
    <button id="rerun" onclick="rerun()">&#9654; Re-run predictions</button>
    <div id="status">Statistical pipeline only: fresh data, Elo, scoring, fixtures,
dashboard. Needs the local server (start.bat / start.sh).</div>
  </div>
  <div class="prompt">
    <b style="color:var(--gold)">Full update with news &amp; injuries — paste this to Claude:</b><br>
    <span id="ptext">{CLAUDE_PROMPT}</span><br>
    <button style="margin-top:8px" onclick="copyPrompt(this)">Copy prompt</button>
  </div>
</div>
<script>
async function rerun() {{
  const b = document.getElementById('rerun'), s = document.getElementById('status');
  b.disabled = true; s.textContent = 'Running pipeline... (30-60s)';
  try {{
    const r = await fetch('/rerun', {{method: 'POST'}});
    const t = await r.text();
    if (r.ok) {{ s.textContent = 'Done - reloading...'; location.reload(); }}
    else {{ s.textContent = 'Failed:\\n' + t; b.disabled = false; }}
  }} catch (e) {{
    s.textContent = 'No local server. Start it first: start.bat (Windows) or ' +
                    './start.sh (Mac/Linux), then use this page at http://localhost:8000/';
    b.disabled = false;
  }}
}}
function copyPrompt(btn) {{
  navigator.clipboard.writeText(document.getElementById('ptext').textContent)
    .then(() => {{ btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy prompt', 1500); }});
}}
</script>"""


def esc(s):
    return html.escape(str(s))


def prob_bar(p):
    h, d, a = p["home_win"], p["draw"], p["away_win"]
    def seg(cls, v):
        label = f"{v:.0%}" if v >= 0.08 else ""
        return f'<div class="{cls}" style="width:{v*100:.1f}%">{label}</div>'
    return f'<div class="bar">{seg("bh",h)}{seg("bd",d)}{seg("ba",a)}</div>'


def upcoming_card(p):
    out = p["predicted_outcome"].replace("home_win", p["home"]) \
                                .replace("away_win", p["away"]).replace("draw", "Draw")
    adj = ""
    if p.get("adjustment"):
        adj = f'<div class="note">Adjusted: {esc(p["adjustment"]["reason"])}</div>'
    comp = p.get("components")
    if comp:
        e, m = comp["elo"], comp["maher"]
        warn = (' · <span style="color:var(--red)">models disagree '
                f'{comp["max_disagreement"]:.0%}</span>') \
            if comp.get("models_disagree") else ""
        adj += (f'<div class="note">Elo {e["home_win"]:.0%}/{e["draw"]:.0%}/'
                f'{e["away_win"]:.0%} · Maher {m["home_win"]:.0%}/'
                f'{m["draw"]:.0%}/{m["away_win"]:.0%}{warn}</div>')
    if p.get("market"):
        mk = p["market"]["probabilities"]
        adj += (f'<div class="note">Market {mk["home_win"]:.0%}/'
                f'{mk["draw"]:.0%}/{mk["away_win"]:.0%} (de-vigged)</div>')
    score = p["top_scorelines"][0]["score"]
    h2h = p.get("head_to_head") or {}
    h2h_line = f'<div class="note">H2H: {esc(h2h["record"])} ({h2h["total"]} meetings)</div>' \
        if h2h.get("total") else ""
    return f"""<div class="card">
      <div class="match">{esc(p["home"])} vs {esc(p["away"])}</div>
      <div class="stage">{esc(p["stage"] or "")} · {esc(p["date"])}
        · Elo {p["elo"]["home"]:.0f} v {p["elo"]["away"]:.0f}</div>
      {prob_bar(p["probabilities"])}
      <div class="legend"><span>{esc(p["home"])} {p["probabilities"]["home_win"]:.0%}</span>
        <span>draw {p["probabilities"]["draw"]:.0%}</span>
        <span>{esc(p["away"])} {p["probabilities"]["away_win"]:.0%}</span></div>
      <div style="margin-top:8px">Pick: <b>{esc(out)}</b> · likely {esc(score)}</div>
      {h2h_line}{adj}</div>"""


def calibration_rows(resolved):
    buckets = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75),
               (0.75, 0.85), (0.85, 1.01)]
    rows = []
    for lo, hi in buckets:
        sel = [p for p in resolved
               if lo <= p["probabilities"][p["predicted_outcome"]] < hi]
        if not sel:
            continue
        hits = sum(p["actual"]["outcome_correct"] for p in sel)
        rate = hits / len(sel)
        rows.append(f"<tr><td>{lo:.0%}–{min(hi,1):.0%}</td>"
                    f'<td class="num">{len(sel)}</td><td class="num">{rate:.0%}</td>'
                    f'<td><span class="hbar" style="width:{rate*160:.0f}px"></span></td></tr>')
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="predictions/dashboard.html")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    elo = json.loads((data_dir / "elo_ratings.json").read_text(encoding="utf-8"))
    log_path = Path("predictions/log.json")
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    odds_path = Path("predictions/tournament_odds.json")
    odds = json.loads(odds_path.read_text(encoding="utf-8")) if odds_path.exists() else None
    hist_path = data_dir / "elo_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []

    resolved = [p for p in log if p.get("actual")]
    pending = sorted([p for p in log if not p.get("actual")], key=lambda p: p["date"])

    parts = [f"<!doctype html><meta charset='utf-8'><title>World Cup Predictor</title>"
             f"<style>{CSS}</style><body>",
             "<h1>World Cup Predictor</h1>",
             f"<div class='sub'>Generated {date.today().isoformat()} · ratings through "
             f"{esc(elo['latest_match'])} · {elo['total_matches']:,} matches rated</div>",
             CONTROLS]

    # model record
    parts.append("<h2>Model record</h2><div class='statrow'>")
    if resolved:
        correct = sum(p["actual"]["outcome_correct"] for p in resolved)
        brier = sum(p["actual"]["brier"] for p in resolved) / len(resolved)
        parts.append(f"<div class='stat'><b>{correct}/{len(resolved)}</b>outcomes correct "
                     f"({correct/len(resolved):.0%})</div>")
        parts.append(f"<div class='stat'><b>{brier:.3f}</b>mean Brier "
                     f"(0.667 = guessing)</div>")
        with_mkt = [p for p in resolved
                    if p["actual"].get("market_brier") is not None]
        if with_mkt:
            n = len(with_mkt)
            mb = sum(p["actual"]["brier"] for p in with_mkt) / n
            kb = sum(p["actual"]["market_brier"] for p in with_mkt) / n
            cls = "ok" if mb < kb else "bad"
            parts.append(f"<div class='stat'><b class='{cls}'>"
                         f"{mb:.3f} vs {kb:.3f}</b>model vs market Brier "
                         f"({n} matches with odds)</div>")
    parts.append(f"<div class='stat'><b>{len(pending)}</b>predictions pending</div></div>")

    if len(resolved) >= 5:
        parts.append("<h2>Calibration — when we say X%, does it happen X% of the time?</h2>"
                     "<table><tr><th>Confidence in pick</th><th class='num'>n</th>"
                     "<th class='num'>Hit rate</th><th></th></tr>"
                     + "".join(calibration_rows(resolved)) + "</table>")

    if pending:
        parts.append("<h2>Upcoming predictions</h2><div class='cards'>"
                     + "".join(upcoming_card(p) for p in pending) + "</div>")

    if resolved:
        rows = []
        exact_hits = 0
        for p in reversed(sorted(resolved, key=lambda p: p["date"])[-25:]):
            mark = '<span class="ok">✓</span>' if p["actual"]["outcome_correct"] \
                else '<span class="bad">✗</span>'
            pick = p["predicted_outcome"].replace("home_win", p["home"]) \
                .replace("away_win", p["away"]).replace("draw", "Draw")
            conf = p["probabilities"][p["predicted_outcome"]]
            adj = " *" if p.get("adjustment") else ""
            pred_score = p["top_scorelines"][0]["score"]
            exact = pred_score == p["actual"]["score"]
            exact_hits += exact
            pred_score_html = f'<span class="ok">{esc(pred_score)} &#127919;</span>' \
                if exact else esc(pred_score)
            rows.append(f"<tr><td>{esc(p['date'])}</td>"
                        f"<td>{esc(p['home'])} vs {esc(p['away'])}</td>"
                        f"<td>{esc(pick)} ({conf:.0%}){adj}</td>"
                        f"<td class='num'>{pred_score_html}</td>"
                        f"<td class='num'><b>{esc(p['actual']['score'])}</b></td>"
                        f"<td>{mark}</td><td class='num'>{p['actual']['brier']:.3f}</td></tr>")
        parts.append("<h2>Previous matches — predicted vs actual</h2>"
                     "<table><tr><th>Date</th><th>Match</th><th>Our pick</th>"
                     "<th class='num'>Predicted score</th><th class='num'>Actual</th>"
                     "<th></th><th class='num'>Brier</th></tr>"
                     + "".join(rows) + "</table>"
                     f"<div class='note'>&#127919; = exact scoreline hit "
                     f"({exact_hits}/{min(len(resolved), 25)} shown) · "
                     "* prediction included a qualitative adjustment</div>")

    if odds:
        rows = []
        for t, d in list(odds["teams"].items())[:12]:
            rows.append(f"<tr><td>{esc(t)}</td>"
                        f"<td><span class='hbar' style='width:{d['Champion']*600:.0f}px'>"
                        f"</span> {d['Champion']:.1%}</td>"
                        f"<td class='num'>{d['Final']:.1%}</td>"
                        f"<td class='num'>{d['SF']:.1%}</td></tr>")
        runs = odds["runs"]
        top_p = next(iter(odds["teams"].values()))["Champion"]
        se = (top_p * (1 - top_p) / runs) ** 0.5
        parts.append(f"<h2>Championship odds ({runs:,} simulations)</h2>"
                     "<table><tr><th>Team</th><th>Champion</th><th class='num'>Final</th>"
                     "<th class='num'>Semis</th></tr>" + "".join(rows) + "</table>"
                     f"<div class='note'>Monte Carlo noise: ±{se:.1%} on the "
                     "leader's number — differences smaller than that are not real.</div>")

    # Elo top 20 with movement vs previous snapshot
    prev = history[-2]["ratings"] if len(history) >= 2 else {}
    rows = []
    for i, (t, r) in enumerate(list(elo["ratings"].items())[:20], 1):
        delta = r["elo"] - prev.get(t, r["elo"])
        move = "" if abs(delta) < 0.5 else \
            f' <span class="{"up" if delta > 0 else "down"}">{delta:+.0f}</span>'
        rows.append(f"<tr><td class='num'>{i}</td><td>{esc(t)}</td>"
                    f"<td class='num'>{r['elo']:.0f}{move}</td>"
                    f"<td class='num'>{esc(r['last_match'])}</td></tr>")
    parts.append("<h2>Elo top 20</h2><table><tr><th class='num'>#</th><th>Team</th>"
                 "<th class='num'>Rating</th><th class='num'>Last match</th></tr>"
                 + "".join(rows) + "</table>")
    parts.append("</body>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"Dashboard written to {out} — open it in a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

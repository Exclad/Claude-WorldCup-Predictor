#!/usr/bin/env python3
"""Local dashboard server with a one-click re-run button.

Serves predictions/dashboard.html at http://localhost:8000/ and exposes
POST /rerun, which executes the full statistical pipeline (run_all.py) and
regenerates the dashboard. Started by start.bat / start.sh in the project
root. Stdlib only.
"""
import argparse
import subprocess
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).parent
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/dashboard.html"):
            page = Path("predictions/dashboard.html")
            if page.exists():
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, "No dashboard yet — click Re-run or run run_all.py once.")
        else:
            self._send(404, "Not found")

    def do_POST(self):
        if self.path != "/rerun":
            self._send(404, "Not found")
            return
        if not LOCK.acquire(blocking=False):
            self._send(409, "A run is already in progress — wait for it to finish.")
            return
        try:
            proc = subprocess.run(
                [sys.executable, str(HERE / "run_all.py"), "--days", "2", "--simulate"],
                capture_output=True, text=True, timeout=900)
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            self._send(200 if proc.returncode == 0 else 500, tail)
        except subprocess.TimeoutExpired:
            self._send(500, "Pipeline timed out after 15 minutes.")
        finally:
            LOCK.release()

    def log_message(self, fmt, *a):  # quieter console
        if "rerun" in (a[0] if a else ""):
            super().log_message(fmt, *a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Windows often blocks ports in Hyper-V's reserved ranges with WinError
    # 10013, and 8000 lands there on many machines. Walk a candidate list;
    # port 0 (OS-assigned) is the always-works last resort.
    candidates = [args.port] + [p for p in (8000, 8080, 8765, 8888, 9090, 0)
                                if p != args.port]
    server = None
    for port in candidates:
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as e:
            print(f"Port {port} unavailable ({e.strerror or e}); trying next...")
    if server is None:
        print("Could not bind any port — is a firewall blocking local servers?")
        return 1
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"World Cup Predictor dashboard: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

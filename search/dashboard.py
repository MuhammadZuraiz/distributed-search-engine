"""
Live crawl dashboard — run alongside master.py
Reads live_stats.json written by master every second.

Run in a separate terminal WHILE crawling:
    python search/dashboard.py
Then open: http://localhost:5001
"""

import json
import os
import sys
from flask import Flask, render_template, Response
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATS_FILE = "data/live_stats.json"
app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes stats every second."""
    def event_gen():
        while True:
            try:
                if os.path.exists(STATS_FILE):
                    with open(STATS_FILE, encoding="utf-8") as f:
                        data = json.load(f)
                    yield f"data: {json.dumps(data)}\n\n"
            except Exception:
                pass
            time.sleep(1)
    return Response(event_gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5001")
    app.run(debug=False, port=5001, threaded=True)
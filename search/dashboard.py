"""
Live crawl dashboard — upgraded with domain metrics
"""

import json
import os
import sys
from flask import Flask, render_template, Response, jsonify
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATS_FILE   = "data/live_stats.json"
METRICS_FILE = "logs/metrics.jsonl"

app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stream")
def stream():
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


@app.route("/api/domain-metrics")
def domain_metrics():
    """
    Read metrics.jsonl and compute per-domain stats:
    fetch count, p50/p95 latency, error rate.
    """
    if not os.path.exists(METRICS_FILE):
        return jsonify({})

    from collections import defaultdict
    domain_times   = defaultdict(list)
    domain_errors  = defaultdict(int)
    domain_fetches = defaultdict(int)
    domain_blocked = defaultdict(int)

    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line.strip())
                    if ev.get("type") != "fetch":
                        continue
                    domain = ev.get("domain", "unknown")
                    status = ev.get("status", "ok")
                    ms     = ev.get("duration_ms", 0)

                    domain_fetches[domain] += 1
                    if status == "ok":
                        domain_times[domain].append(ms)
                    elif status == "failed":
                        domain_errors[domain] += 1
                    elif status == "blocked":
                        domain_blocked[domain] += 1
                except Exception:
                    continue
    except Exception:
        return jsonify({})

    def pct(lst, p):
        if not lst:
            return 0
        s   = sorted(lst)
        idx = max(0, int(len(s) * p / 100) - 1)
        return round(s[idx], 1)

    result = {}
    for domain in domain_fetches:
        fetches = domain_fetches[domain]
        errors  = domain_errors[domain]
        times   = domain_times[domain]
        result[domain] = {
            "fetches":    fetches,
            "errors":     errors,
            "blocked":    domain_blocked[domain],
            "error_rate": round(errors / fetches * 100, 1) if fetches else 0,
            "p50_ms":     pct(times, 50),
            "p95_ms":     pct(times, 95),
            "p99_ms":     pct(times, 99),
            "avg_ms":     round(sum(times)/len(times), 1) if times else 0,
        }

    return jsonify(result)


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5001")
    app.run(debug=False, port=5001, threaded=True)
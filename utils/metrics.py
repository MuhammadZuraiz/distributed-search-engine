"""
Metrics collector — per-domain fetch latency + error rates
============================================================
Tracks p50/p95/p99 fetch times, error rates, and queue depth.
Written to logs/metrics.jsonl (one JSON object per line).
Thread-safe — safe to call from multiple worker threads.
"""

import json
import os
import time
import threading
from collections import defaultdict

METRICS_FILE = "logs/metrics.jsonl"
_lock        = threading.Lock()

# in-memory accumulators — flushed to disk periodically
_domain_times   = defaultdict(list)   # domain -> [duration_ms, ...]
_domain_errors  = defaultdict(int)    # domain -> error count
_domain_fetches = defaultdict(int)    # domain -> total fetch count
_domain_blocked = defaultdict(int)    # domain -> robots.txt block count


def record_fetch(domain, duration_ms, status="ok"):
    """Record one fetch event."""
    with _lock:
        _domain_fetches[domain] += 1
        if status == "ok":
            _domain_times[domain].append(duration_ms)
        elif status == "failed":
            _domain_errors[domain] += 1
        elif status == "blocked":
            _domain_blocked[domain] += 1

    _write_event({
        "type":        "fetch",
        "domain":      domain,
        "duration_ms": round(duration_ms, 1),
        "status":      status,
        "ts":          time.time()
    })


def get_domain_stats():
    """Return per-domain stats dict."""
    with _lock:
        stats = {}
        for domain in set(list(_domain_fetches.keys()) +
                          list(_domain_errors.keys())):
            times   = sorted(_domain_times.get(domain, []))
            fetches = _domain_fetches.get(domain, 0)
            errors  = _domain_errors.get(domain, 0)
            blocked = _domain_blocked.get(domain, 0)

            def percentile(lst, p):
                if not lst:
                    return 0
                idx = max(0, int(len(lst) * p / 100) - 1)
                return round(lst[idx], 1)

            stats[domain] = {
                "fetches":    fetches,
                "errors":     errors,
                "blocked":    blocked,
                "error_rate": round(errors / fetches * 100, 1) if fetches else 0,
                "p50_ms":     percentile(times, 50),
                "p95_ms":     percentile(times, 95),
                "p99_ms":     percentile(times, 99),
            }
        return stats


def record_queue_depth(depth, active_workers):
    """Record master queue depth snapshot."""
    _write_event({
        "type":           "queue",
        "depth":          depth,
        "active_workers": active_workers,
        "ts":             time.time()
    })


def _write_event(event):
    """Append one JSON event to the metrics log."""
    os.makedirs("logs", exist_ok=True)
    with _lock:
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
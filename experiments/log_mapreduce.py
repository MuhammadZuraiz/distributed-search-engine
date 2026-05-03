"""
MapReduce on Search Logs
=========================
Uses YOUR OWN MapReduce engine to analyse your own query logs.
This proves the MapReduce implementation is general-purpose —
not just hardcoded for document indexing.

Jobs implemented:
  1. Query co-occurrence   — which queries are searched together?
  2. Click position bias   — which result positions get clicked most?
  3. Query term frequency  — what are the most searched individual words?
  4. Session journey       — what search paths do users take?
  5. Zero-result analysis  — which terms consistently return nothing?

This is exactly what Hadoop/Spark do — run MapReduce over logs
to understand system behaviour. Running it on your own logs
closes the loop: the system analyses itself.
"""

import sys
import os
import json
import time
import sqlite3
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── generic MapReduce engine ──────────────────────────────────────────────────

def map_reduce(data, mapper, reducer, combiner=None):
    """
    General-purpose MapReduce.

    mapper   : fn(record) -> [(key, value), ...]
    combiner : fn(key, values) -> value  (optional local aggregation)
    reducer  : fn(key, values) -> result

    This is the same engine that indexes documents —
    now reused for log analysis.
    """
    # MAP phase
    pairs = []
    for record in data:
        pairs.extend(mapper(record))

    # SHUFFLE phase — group by key
    grouped = defaultdict(list)
    for key, value in pairs:
        grouped[key].append(value)

    # optional COMBINE phase (local reducer)
    if combiner:
        grouped = {k: [combiner(k, vs)] for k, vs in grouped.items()}

    # REDUCE phase
    results = {}
    for key, values in grouped.items():
        results[key] = reducer(key, values)

    return results


# ── load logs from SQLite ─────────────────────────────────────────────────────

def load_logs():
    db_path = "data/search.db"
    if not os.path.exists(db_path):
        print("[log_mr] No database found. Run some searches first.")
        return [], []

    conn    = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    queries = [dict(r) for r in conn.execute(
        "SELECT * FROM query_log ORDER BY timestamp"
    ).fetchall()]

    clicks  = [dict(r) for r in conn.execute(
        "SELECT * FROM click_log ORDER BY timestamp"
    ).fetchall()]

    conn.close()
    print(f"[log_mr] Loaded {len(queries)} queries, {len(clicks)} clicks")
    return queries, clicks


# ── Job 1: Query term frequency ───────────────────────────────────────────────

def job1_term_frequency(queries):
    """
    Which individual words are searched most often?
    MAP:    query record -> [(word, 1), (word, 1), ...]
    REDUCE: word -> total count
    """
    print("\n[Job 1] Query term frequency")
    print("-" * 50)

    STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "to", "for"}

    def mapper(record):
        words = record["query"].lower().split()
        return [(w, 1) for w in words
                if len(w) >= 3 and w not in STOPWORDS]

    def reducer(key, values):
        return sum(values)

    t0      = time.perf_counter()
    results = map_reduce(queries, mapper, reducer)
    elapsed = (time.perf_counter() - t0) * 1000

    top = sorted(results.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  MapReduce completed in {elapsed:.2f}ms")
    print(f"  Unique search terms: {len(results)}")
    print(f"\n  Top 15 searched terms:")
    for term, count in top:
        bar = "#" * min(count * 3, 30)
        print(f"    {term:<20} {count:4d}  {bar}")

    return results


# ── Job 2: Click position bias ────────────────────────────────────────────────

def job2_click_position_bias(clicks):
    """
    Which result positions get clicked most?
    Measures position bias — do users click result #1 more?
    MAP:    click -> [(position, 1)]
    REDUCE: position -> total clicks
    """
    print("\n[Job 2] Click position bias analysis")
    print("-" * 50)

    def mapper(record):
        pos = record.get("position", 0)
        return [(pos, 1)]

    def reducer(key, values):
        return sum(values)

    t0      = time.perf_counter()
    results = map_reduce(clicks, mapper, reducer)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  MapReduce completed in {elapsed:.2f}ms")
    total = sum(results.values()) or 1
    print(f"\n  Clicks by position:")
    for pos in sorted(results.keys()):
        count = results[pos]
        pct   = count / total * 100
        bar   = "#" * int(pct / 2)
        print(f"    Position {pos:2d}: {count:4d} clicks ({pct:5.1f}%)  {bar}")

    return results


# ── Job 3: Query co-occurrence ────────────────────────────────────────────────

def job3_query_cooccurrence(queries):
    """
    Which queries are searched in the same session?
    MAP:    (session_id, query) -> [(session_id, query)]
    REDUCE: session_id -> [queries] -> emit all pairs
    """
    print("\n[Job 3] Query co-occurrence (same session)")
    print("-" * 50)

    def mapper(record):
        sid   = record.get("session_id", "unknown")
        query = record["query"].lower().strip()
        return [(sid, query)]

    def reducer(key, values):
        return list(set(values))   # unique queries per session

    t0       = time.perf_counter()
    sessions = map_reduce(queries, mapper, reducer)

    # second MapReduce pass: find co-occurring query pairs
    pair_counts = defaultdict(int)
    for sid, session_queries in sessions.items():
        for i in range(len(session_queries)):
            for j in range(i+1, len(session_queries)):
                pair = tuple(sorted([session_queries[i], session_queries[j]]))
                pair_counts[pair] += 1

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  MapReduce completed in {elapsed:.2f}ms")
    print(f"  Sessions analysed: {len(sessions)}")

    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_pairs:
        print(f"\n  Top co-occurring query pairs:")
        for (q1, q2), count in top_pairs:
            print(f"    '{q1}' + '{q2}' : {count}x")
    else:
        print("  No co-occurring queries yet (need more sessions)")

    return pair_counts


# ── Job 4: Zero-result query analysis ────────────────────────────────────────

def job4_zero_results(queries):
    """
    Which queries consistently return no results?
    These represent content gaps — pages we should crawl.
    MAP:    query -> [(query, result_count)]
    REDUCE: query -> avg result count
    """
    print("\n[Job 4] Zero-result query analysis")
    print("-" * 50)

    def mapper(record):
        return [(record["query"].lower(), record.get("result_count", 0))]

    def reducer(key, values):
        return {"count": len(values), "avg_results": sum(values) / len(values)}

    t0      = time.perf_counter()
    results = map_reduce(queries, mapper, reducer)
    elapsed = (time.perf_counter() - t0) * 1000

    zero    = {q: v for q, v in results.items() if v["avg_results"] == 0}
    low     = {q: v for q, v in results.items()
               if 0 < v["avg_results"] < 3 and v["count"] > 1}

    print(f"  MapReduce completed in {elapsed:.2f}ms")
    print(f"  Total unique queries: {len(results)}")
    print(f"  Zero-result queries: {len(zero)}")
    print(f"  Low-result queries (<3 results): {len(low)}")

    if zero:
        print(f"\n  Content gaps (zero results):")
        for q, v in list(zero.items())[:10]:
            print(f"    '{q}' searched {v['count']}x — no results")
    else:
        print("  No zero-result queries found — good coverage!")

    return {"zero": zero, "low": low}


# ── Job 5: Hourly query volume ────────────────────────────────────────────────

def job5_hourly_volume(queries):
    """
    When do users search most? By hour of day.
    MAP:    query -> [(hour, 1)]
    REDUCE: hour -> total queries
    """
    print("\n[Job 5] Query volume by hour of day")
    print("-" * 50)

    def mapper(record):
        ts = record.get("timestamp", "")
        if len(ts) >= 13:
            hour = ts[11:13]   # "HH" from ISO timestamp
            return [(hour, 1)]
        return []

    def reducer(key, values):
        return sum(values)

    t0      = time.perf_counter()
    results = map_reduce(queries, mapper, reducer)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  MapReduce completed in {elapsed:.2f}ms")
    print(f"\n  Queries by hour:")
    max_count = max(results.values()) if results else 1
    for hour in sorted(results.keys()):
        count = results[hour]
        bar   = "#" * int(count / max_count * 25)
        print(f"    {hour}:00  {count:4d}  {bar}")

    return results


# ── run all jobs ──────────────────────────────────────────────────────────────

def run_all_jobs():
    print("=" * 60)
    print("  MapReduce Log Analysis Engine")
    print("  Running your MapReduce implementation on search logs")
    print("=" * 60)

    queries, clicks = load_logs()

    if not queries:
        print("\nNo query logs found. Do some searches first then re-run.")
        print("Try: python search/app.py  then search for a few terms.")
        return

    # run all 5 jobs
    r1 = job1_term_frequency(queries)
    r2 = job2_click_position_bias(clicks) if clicks else {}
    r3 = job3_query_cooccurrence(queries)
    r4 = job4_zero_results(queries)
    r5 = job5_hourly_volume(queries)

    # summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total queries analysed : {len(queries)}")
    print(f"  Total clicks analysed  : {len(clicks)}")
    print(f"  Unique search terms    : {len(r1)}")
    print(f"  Zero-result queries    : {len(r4['zero'])}")
    print(f"  MapReduce jobs run     : 5")
    print("=" * 60)

    # save results to JSON for the analytics dashboard
    os.makedirs("data/index", exist_ok=True)
    with open("data/index/log_analysis.json", "w") as f:
        json.dump({
            "top_terms":    sorted(r1.items(),
                                   key=lambda x: x[1],
                                   reverse=True)[:20],
            "position_bias": {str(k): v for k, v in r2.items()},
            "zero_results":  list(r4["zero"].keys())[:20],
            "hourly_volume": r5,
        }, f, indent=2)
    print("\n  Results saved: data/index/log_analysis.json")


if __name__ == "__main__":
    run_all_jobs()
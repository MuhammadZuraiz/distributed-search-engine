"""
Search Engine — BM25 + PageRank + Autocomplete + REST API + Snippets
"""

import sys
import os
import json
import re
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, render_template, jsonify, Response
from search.bm25 import BM25
from trie import build_trie

from embeddings import BERTSearcher
from semantic import SemanticSearcher

from db.database import init_db, get_url_map, log_query, log_click, get_click_boost

from utils.metrics import get_domain_stats

from spell import SpellCorrector

from expander import QueryExpander

from collaborative import SHARED_MODEL

import uuid

INDEX_PATH   = "data/index/inverted_index.json"
URL_MAP_PATH = "data/index/url_map.json"
PR_PATH      = "data/index/pagerank.json"

app = Flask(__name__)

# ── load everything at startup ────────────────────────────────────────────────
init_db()

print("Loading index...")
with open(INDEX_PATH, encoding="utf-8") as f:
    INDEX = json.load(f)

print("Loading URL map from database...")
URL_MAP = get_url_map()

PAGERANK = {}
if os.path.exists(PR_PATH):
    with open(PR_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    PAGERANK = {int(k): v for k, v in raw.items()}

RANKER     = BM25(INDEX, URL_MAP, PAGERANK)
TRIE       = build_trie(INDEX)

SPELL = SpellCorrector(INDEX)
print("Spell corrector ready")

EXPANDER = QueryExpander(INDEX)
print("Query expander ready")

print("Loading BERT embeddings...")
try:
    BERT = BERTSearcher()
    USE_BERT = True
    print("BERT search ready")
except Exception as e:
    print(f"BERT unavailable ({e}), falling back to TF-IDF")
    BERT     = None
    USE_BERT = False

print("Loading TF-IDF fallback vectors...")
SEMANTIC = SemanticSearcher()

TOTAL_DOCS = len(URL_MAP)

print(f"Ready: {len(INDEX):,} terms, {TOTAL_DOCS} docs, trie built")


# ── helpers ───────────────────────────────────────────────────────────────────

def highlight(text, query):
    if not text or not query:
        return text
    terms   = [re.escape(t) for t in query.split() if len(t) >= 3]
    if not terms:
        return text
    pattern = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)

def blended_search(query, expanded_query=None, top_n=20):
    terms  = query.strip().split()
    bm25_w = 0.75 if len(terms) == 1 else 0.60
    bert_w = 0.25 if USE_BERT else 0
    tfidf_w = 0 if USE_BERT else (1 - bm25_w)

    search_query = expanded_query or query

    # BM25
    bm25_results = RANKER.score(search_query, top_n=top_n * 2)
    bm25_map     = {r["doc_id"]: r for r in bm25_results}
    if bm25_results:
        max_bm25 = max(r["score"] for r in bm25_results) or 1
        for r in bm25_results:
            r["bm25_norm"] = r["score"] / max_bm25

    # BERT semantic
    bert_map = {}
    if USE_BERT and BERT:
        bert_results = BERT.search(query, top_n=top_n * 2)
        bert_map     = {doc_id: score for doc_id, score in bert_results
                        if score > 0.2}   # higher threshold for BERT
        max_bert     = max(bert_map.values()) if bert_map else 1
        bert_map     = {d: s/max_bert for d, s in bert_map.items()}

    # TF-IDF fallback
    tfidf_map = {}
    if not USE_BERT:
        tfidf_results = SEMANTIC.search(query, top_n=top_n * 2)
        tfidf_map     = {doc_id: score for doc_id, score in tfidf_results
                         if score > 0.05}
        max_tfidf     = max(tfidf_map.values()) if tfidf_map else 1
        tfidf_map     = {d: s/max_tfidf for d, s in tfidf_map.items()}

    all_ids = (set(bm25_map.keys()) |
               set(bert_map.keys()) |
               set(tfidf_map.keys()))

    blended = []
    for doc_id in all_ids:
        bm25_norm  = bm25_map[doc_id]["bm25_norm"] if doc_id in bm25_map else 0
        bert_norm  = bert_map.get(doc_id, 0)
        tfidf_norm = tfidf_map.get(doc_id, 0)
        sem_norm   = bert_norm if USE_BERT else tfidf_norm

        final = bm25_w * bm25_norm + (bert_w + tfidf_w) * sem_norm

        meta = URL_MAP.get(str(doc_id), {})
        blended.append({
            "doc_id":   doc_id,
            "url":      meta.get("url", ""),
            "title":    meta.get("title", "No title"),
            "snippet":  meta.get("snippet", ""),
            "bm25":     round(bm25_map[doc_id]["score"]
                              if doc_id in bm25_map else 0, 3),
            "semantic": round(sem_norm, 3),
            "pagerank": round(PAGERANK.get(doc_id, 0), 4),
            "score":    round(final, 3),
        })

    blended.sort(key=lambda x: x["score"], reverse=True)
    return blended[:top_n]

def get_session_id():
    """Get or create a persistent session ID from cookie."""
    sid = request.cookies.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
    return sid

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    query      = request.args.get("q", "").strip()
    session_id = get_session_id()

    # spell correction
    corrected_query   = None
    spell_corrections = {}
    if query:
        fixed, corrections = SPELL.correct_query(query)
        if corrections:
            corrected_query   = fixed
            spell_corrections = corrections

    active_query = corrected_query or query

    # query expansion
    expanded_query = None
    expansions_map = {}
    if active_query:
        expanded, expansions = EXPANDER.expand(active_query)
        if expansions:
            expanded_query = expanded
            expansions_map = expansions

    results = blended_search(active_query,
                             expanded_query=expanded_query) if active_query else []

    if query and not results and corrected_query:
        results = blended_search(corrected_query,
                                 expanded_query=expanded_query)

    if active_query and results:
        for r in results:
            boost      = get_click_boost(r["doc_id"], active_query)
            r["score"] = round(r["score"] * boost, 3)
        results.sort(key=lambda x: x["score"], reverse=True)

    if active_query:
        log_query(active_query, len(results), session_id)

    for r in results:
        r["snippet"] = highlight(r.get("snippet", ""), active_query)

    return render_template(
        "index.html",
        query=query,
        results=results,
        total_docs=TOTAL_DOCS,
        total_terms=len(INDEX),
        corrected_query=corrected_query,
        spell_corrections=spell_corrections,
        expansions_map=expansions_map,
    )

@app.route("/api/suggest")
def suggest():
    prefix = request.args.get("prefix", "").strip()
    if len(prefix) < 2:
        return jsonify({"suggestions": []})
    suggestions = TRIE.search_prefix(prefix, max_results=8)
    return jsonify({"suggestions": suggestions})


@app.route("/api/search")
def api_search():
    query   = request.args.get("q", "").strip()
    limit   = int(request.args.get("limit", 10))
    results = RANKER.score(query, top_n=limit) if query else []
    for r in results:
        meta         = URL_MAP.get(str(r["doc_id"]), {})
        r["snippet"] = meta.get("snippet", "")
    return jsonify({"query": query, "count": len(results), "results": results})


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total_documents": TOTAL_DOCS,
        "unique_terms":    len(INDEX),
        "top_pagerank": sorted(
            [(int(k), v) for k, v in PAGERANK.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
    })


@app.route("/api/document/<int:doc_id>")
def api_document(doc_id):
    meta = URL_MAP.get(str(doc_id))
    if not meta:
        return jsonify({"error": "Document not found"}), 404
    return jsonify({
        "doc_id":   doc_id,
        "url":      meta.get("url", ""),
        "title":    meta.get("title", ""),
        "snippet":  meta.get("snippet", ""),
        "pagerank": PAGERANK.get(doc_id, 0)
    })

@app.route("/api/click", methods=["POST"])
def track_click():
    data       = request.get_json()
    query      = data.get("query", "")
    doc_id     = data.get("doc_id")
    position   = data.get("position", 0)
    session_id = get_session_id()
    if query and doc_id is not None:
        log_click(query, doc_id, position, session_id)
    return jsonify({"ok": True})

@app.route("/api/metrics")
def api_metrics():
    from db.database import get_crawl_stats
    return jsonify({
        "crawl_stats":   get_crawl_stats(),
        "domain_stats":  get_domain_stats(),
    })

@app.route("/api/ratelimit")
def api_ratelimit():
    """Show current token bucket status per domain."""
    from utils.metrics import get_domain_stats
    return jsonify({
        "domain_fetch_stats": get_domain_stats(),
        "note": "Token buckets live in master process during crawl"
    })

@app.route("/api/spell")
def api_spell():
    word = request.args.get("w", "").strip()
    if not word:
        return jsonify({"word": word, "correction": None})
    correction = SPELL.correct(word)
    return jsonify({"word": word, "correction": correction})

@app.route("/api/expand")
def api_expand():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"query": query, "expansions": {}})
    _, expansions = EXPANDER.expand(query)
    return jsonify({"query": query, "expansions": expansions})

@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Trigger incremental re-index of new documents."""
    import threading
    def run():
        from indexer.incremental import run_incremental_index
        count = run_incremental_index()
        print(f"[reindex] Indexed {count} new documents")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started",
                    "message": "Incremental index running in background"})

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics")
def api_analytics():
    from db.database import (get_top_queries, get_zero_result_queries,
                              get_top_clicked_docs, get_conn)
    conn = get_conn()

    total_queries = conn.execute(
        "SELECT COUNT(*) FROM query_log"
    ).fetchone()[0]

    unique_queries = conn.execute(
        "SELECT COUNT(DISTINCT query) FROM query_log"
    ).fetchone()[0]

    total_clicks = conn.execute(
        "SELECT COUNT(*) FROM click_log"
    ).fetchone()[0]

    # hourly query volume (last 24 hours)
    hourly = conn.execute("""
        SELECT strftime('%H:00', timestamp) as hour,
               COUNT(*) as count
        FROM query_log
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY hour
        ORDER BY hour
    """).fetchall()

    return jsonify({
        "total_queries":       total_queries,
        "unique_queries":      unique_queries,
        "total_clicks":        total_clicks,
        "top_queries":         get_top_queries(20),
        "zero_result_queries": get_zero_result_queries(10),
        "top_clicked":         get_top_clicked_docs(10),
        "hourly_volume":       [dict(r) for r in hourly],
    })

@app.route("/api/federated")
def api_federated():
    """
    Federated search — searches index shards independently
    and merges results. Demonstrates distributed query execution.
    For demo purposes runs shards sequentially in-process.
    Full MPI version: mpiexec -n 7 python search/federated.py
    """
    from search.federated import (shard_index, ShardSearcher,
                                  FederatedSearchCoordinator)
    import json

    query = request.args.get("q", "").strip()
    n_shards = int(request.args.get("shards", 3))

    if not query:
        return jsonify({"error": "q parameter required"})

    terms   = [t.lower() for t in query.split() if len(t) >= 3]
    t_start = time.time()

    shards    = shard_index(INDEX, n_shards)
    searchers = [ShardSearcher(s, i, PAGERANK)
                 for i, s in enumerate(shards)]

    shard_results  = []
    shard_latencies = []
    for i, searcher in enumerate(searchers):
        t0      = time.time()
        results = searcher.search(terms, top_k=10)
        shard_latencies.append(round((time.time() - t0) * 1000, 2))
        shard_results.append(results)

    coordinator = FederatedSearchCoordinator(URL_MAP)
    merged      = coordinator.merge_shard_results(shard_results)
    total_ms    = round((time.time() - t_start) * 1000, 2)

    return jsonify({
        "query":           query,
        "n_shards":        n_shards,
        "total_ms":        total_ms,
        "shard_latencies": shard_latencies,
        "shard_sizes":     [len(s) for s in shards],
        "results":         merged[:10],
    })

@app.route("/collaborative")
def collaborative_search():
    return render_template("collaborative.html")


@app.route("/collab/signal", methods=["POST"])
def collab_signal():
    """Receive search signals from any user and update shared model."""
    data       = request.get_json()
    session_id = get_session_id()
    signal_type = data.get("type")

    if signal_type == "search":
        SHARED_MODEL.record_query(data.get("query", ""), session_id)

    elif signal_type == "click":
        SHARED_MODEL.record_click(
            doc_id    = data.get("doc_id"),
            query     = data.get("query", ""),
            position  = data.get("position", 10),
            session_id = session_id
        )

    return jsonify({"ok": True,
                    "active_users": SHARED_MODEL.get_active_user_count()})


@app.route("/collab/stream")
def collab_stream():
    """SSE stream pushing live collaborative activity to all connected users."""
    def event_generator():
        last_sent = 0
        while True:
            now = time.time()
            if now - last_sent >= 1.5:
                stats    = SHARED_MODEL.get_stats()
                trending = SHARED_MODEL.get_trending_docs(URL_MAP, limit=5)
                feed     = SHARED_MODEL.get_live_feed(limit=10)
                payload  = json.dumps({
                    "stats":    stats,
                    "trending": trending,
                    "feed":     feed,
                })
                yield f"data: {payload}\n\n"
                last_sent = now
            time.sleep(0.5)

    return Response(event_generator(), mimetype="text/event-stream")

@app.route("/collab/search")
def collab_search_api():
    query      = request.args.get("q", "").strip()
    session_id = get_session_id()

    if not query:
        return jsonify({"results": [], "query": ""})

    results = blended_search(query)

    for r in results:
        crowd_boost = SHARED_MODEL.get_crowd_boost(r["doc_id"])
        r["crowd"]  = round(crowd_boost, 3)
        r["score"]  = round(r["score"] + crowd_boost * 0.1, 3)

    results.sort(key=lambda x: x["score"], reverse=True)
    for r in results:
        r["snippet"] = highlight(r.get("snippet", ""), query)

    resp = jsonify({
        "query":        query,
        "results":      results[:20],
        "active_users": SHARED_MODEL.get_active_user_count(),
        "trending":     [q for q, _ in SHARED_MODEL.get_trending_queries(5)],
    })
    resp.set_cookie("session_id", session_id,
                    max_age=3600, samesite="Lax")
    return resp

@app.route("/api/log-analysis")
def api_log_analysis():
    """Returns results of MapReduce log analysis jobs."""
    path = "data/index/log_analysis.json"
    if not os.path.exists(path):
        return jsonify({"error": "Run experiments/log_mapreduce.py first"})
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    print("Search engine running at http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
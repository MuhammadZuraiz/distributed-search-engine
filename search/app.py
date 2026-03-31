"""
Search Engine — BM25 + PageRank + Autocomplete + REST API + Snippets
"""

import sys
import os
import json
import re
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, render_template, jsonify
from search.bm25 import BM25
from trie import build_trie

from embeddings import BERTSearcher
from semantic import SemanticSearcher

from db.database import init_db, get_url_map, log_query, log_click, get_click_boost

from utils.metrics import get_domain_stats

from spell import SpellCorrector

from expander import QueryExpander

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

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    query      = request.args.get("q", "").strip()
    session_id = request.cookies.get("session_id", "")

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
    session_id = request.cookies.get("session_id", "")
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

if __name__ == "__main__":
    print("Search engine running at http://localhost:5000")
    app.run(debug=False, port=5000)
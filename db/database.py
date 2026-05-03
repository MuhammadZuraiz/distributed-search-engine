"""
Database layer — SQLite backend
=================================
Replaces flat JSON files with a proper relational store.
All reads/writes go through this module.

Tables:
  documents  — crawled pages (url, title, text, snippet, links, pagerank)
  crawl_log  — per-URL crawl events (worker, duration, status, timestamp)
  query_log  — search queries + clicks for analytics + feedback
  index_meta — tracks which docs are indexed (for incremental indexing)
"""

import sqlite3
import json
import os
import threading
from contextlib import contextmanager

DB_PATH = "data/search.db"

# thread-local storage so each thread gets its own connection
_local = threading.local()


def get_conn():
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")   # write-ahead log for concurrency
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA cache_size=10000")
    return _local.conn


@contextmanager
def transaction():
    """Context manager for a single atomic transaction."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Create all tables if they don't exist."""
    with transaction() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      INTEGER PRIMARY KEY,
            url         TEXT    NOT NULL UNIQUE,
            title       TEXT,
            text        TEXT,
            snippet     TEXT,
            links       TEXT,        -- JSON array of outgoing URLs
            pagerank    REAL    DEFAULT 0.0,
            domain      TEXT,
            crawled_at  TEXT,        -- ISO timestamp
            worker_rank INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_documents_url    ON documents(url);
        CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);

        CREATE TABLE IF NOT EXISTS crawl_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT,
            doc_id      INTEGER,
            worker_rank INTEGER,
            status      TEXT,        -- 'ok', 'failed', 'blocked', 'duplicate'
            duration_ms REAL,
            depth       INTEGER,
            timestamp   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_crawl_log_status ON crawl_log(status);
        CREATE INDEX IF NOT EXISTS idx_crawl_log_worker ON crawl_log(worker_rank);

        CREATE TABLE IF NOT EXISTS query_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query        TEXT,
            result_count INTEGER,
            timestamp    TEXT,
            session_id   TEXT
        );

        CREATE TABLE IF NOT EXISTS click_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            query      TEXT,
            doc_id     INTEGER,
            position   INTEGER,
            timestamp  TEXT,
            session_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_click_log_query ON click_log(query);

        CREATE TABLE IF NOT EXISTS index_meta (
            doc_id     INTEGER PRIMARY KEY,
            indexed_at TEXT,
            term_count INTEGER
        );
        """)
    print(f"[db] Initialised: {DB_PATH}")


# ── document operations ───────────────────────────────────────────────────────

def insert_document(doc_id, url, title, text, snippet, links,
                    worker_rank=None, crawled_at=None):
    """Insert or replace a crawled document."""
    from urllib.parse import urlparse
    from datetime import datetime
    domain     = urlparse(url).netloc
    crawled_at = crawled_at or datetime.utcnow().isoformat()
    with transaction() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO documents
                (doc_id, url, title, text, snippet, links,
                 domain, crawled_at, worker_rank)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (doc_id, url, title, text, snippet,
              json.dumps(links), domain, crawled_at, worker_rank))


def get_document(doc_id):
    """Fetch one document by ID. Returns dict or None."""
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["links"] = json.loads(d["links"] or "[]")
    return d


def get_all_documents():
    """Return all documents as a list of dicts."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["links"] = json.loads(d["links"] or "[]")
        result.append(d)
    return result


def get_document_count():
    conn = get_conn()
    return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def update_pagerank(scores: dict):
    """Bulk update PageRank scores. scores = {doc_id: score}"""
    with transaction() as conn:
        conn.executemany(
            "UPDATE documents SET pagerank = ? WHERE doc_id = ?",
            [(score, doc_id) for doc_id, score in scores.items()]
        )


def get_url_map():
    """Return {str(doc_id): {url, title, snippet}} for search engine."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT doc_id, url, title, snippet FROM documents"
    ).fetchall()
    return {str(row["doc_id"]): {
        "url":     row["url"],
        "title":   row["title"],
        "snippet": row["snippet"] or ""
    } for row in rows}


def get_crawled_urls():
    """Return set of all crawled URLs (for deduplication)."""
    conn = get_conn()
    rows = conn.execute("SELECT url FROM documents").fetchall()
    return {row["url"] for row in rows}


# ── crawl log operations ──────────────────────────────────────────────────────

def log_crawl(url, doc_id, worker_rank, status,
              duration_ms=0, depth=0):
    from datetime import datetime
    with transaction() as conn:
        conn.execute("""
            INSERT INTO crawl_log
                (url, doc_id, worker_rank, status, duration_ms, depth, timestamp)
            VALUES (?,?,?,?,?,?,?)
        """, (url, doc_id, worker_rank, status,
              duration_ms, depth,
              datetime.utcnow().isoformat()))


def get_crawl_stats():
    """Return crawl statistics dict."""
    conn = get_conn()
    total  = conn.execute("SELECT COUNT(*) FROM crawl_log").fetchone()[0]
    ok     = conn.execute("SELECT COUNT(*) FROM crawl_log WHERE status='ok'").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM crawl_log WHERE status='failed'").fetchone()[0]
    blocked= conn.execute("SELECT COUNT(*) FROM crawl_log WHERE status='blocked'").fetchone()[0]
    avg_ms = conn.execute(
        "SELECT AVG(duration_ms) FROM crawl_log WHERE status='ok'"
    ).fetchone()[0] or 0
    return {
        "total": total, "ok": ok,
        "failed": failed, "blocked": blocked,
        "avg_duration_ms": round(avg_ms, 1)
    }


# ── query + click log operations ─────────────────────────────────────────────

def log_query(query, result_count, session_id=""):
    from datetime import datetime
    with transaction() as conn:
        conn.execute("""
            INSERT INTO query_log (query, result_count, timestamp, session_id)
            VALUES (?,?,?,?)
        """, (query, result_count,
              datetime.utcnow().isoformat(), session_id))


def log_click(query, doc_id, position, session_id=""):
    from datetime import datetime
    with transaction() as conn:
        conn.execute("""
            INSERT INTO click_log (query, doc_id, position, timestamp, session_id)
            VALUES (?,?,?,?,?)
        """, (query, doc_id, position,
              datetime.utcnow().isoformat(), session_id))


def get_click_boost(doc_id, query):
    """Return click boost multiplier for a doc+query pair."""
    conn  = get_conn()
    clicks = conn.execute("""
        SELECT COUNT(*) FROM click_log
        WHERE doc_id = ? AND query LIKE ?
    """, (doc_id, f"%{query.split()[0] if query else ''}%")).fetchone()[0]
    # log scale: 0 clicks=1.0x, 1=1.1x, 5=1.2x, 20=1.3x
    import math
    return 1.0 + 0.1 * math.log1p(clicks)


# ── index meta operations ─────────────────────────────────────────────────────

def mark_indexed(doc_id, term_count):
    from datetime import datetime
    with transaction() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO index_meta (doc_id, indexed_at, term_count)
            VALUES (?,?,?)
        """, (doc_id, datetime.utcnow().isoformat(), term_count))


def get_unindexed_doc_ids():
    """Return doc_ids not yet in index_meta — for incremental indexing."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.doc_id FROM documents d
        LEFT JOIN index_meta i ON d.doc_id = i.doc_id
        WHERE i.doc_id IS NULL
    """).fetchall()
    return [row["doc_id"] for row in rows]


# ── analytics queries ─────────────────────────────────────────────────────────

def get_top_queries(limit=20):
    conn = get_conn()
    rows = conn.execute("""
        SELECT query, COUNT(*) as count,
               AVG(result_count) as avg_results
        FROM query_log
        GROUP BY query
        ORDER BY count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_zero_result_queries(limit=20):
    conn = get_conn()
    rows = conn.execute("""
        SELECT query, COUNT(*) as count
        FROM query_log
        WHERE result_count = 0
        GROUP BY query
        ORDER BY count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_top_clicked_docs(limit=10):
    conn = get_conn()
    rows = conn.execute("""
        SELECT c.doc_id, d.title, d.url, COUNT(*) as clicks
        FROM click_log c
        JOIN documents d ON c.doc_id = d.doc_id
        GROUP BY c.doc_id
        ORDER BY clicks DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── migration: import existing JSON files ─────────────────────────────────────

def migrate_from_json(crawled_dir="data/crawled_distributed", delete_after=False):
    """
    One-time migration: import all existing JSON files into SQLite.
    Optionally delete JSON files after successful import.
    """
    import glob
    from datetime import datetime

    files   = sorted(glob.glob(os.path.join(crawled_dir, "*.json")))
    total   = len(files)
    success = 0

    print(f"[db] Migrating {total} JSON files to SQLite...")

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)

            insert_document(
                doc_id      = doc["doc_id"],
                url         = doc.get("url", ""),
                title       = doc.get("title", ""),
                text        = doc.get("text", ""),
                snippet     = doc.get("snippet", ""),
                links       = doc.get("links", []),
                crawled_at  = datetime.utcnow().isoformat()
            )
            success += 1

        except Exception as e:
            print(f"  [warn] Failed to migrate {fp}: {e}")

    print(f"[db] Migration complete: {success}/{total} documents imported")

    if delete_after and success == total:
        for fp in files:
            os.remove(fp)
        print(f"[db] Deleted {total} JSON files")
    elif delete_after:
        print(f"[db] Skipped deletion — {total - success} files failed")

    return success
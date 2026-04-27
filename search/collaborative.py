"""
Collaborative Real-Time Search
================================
Multiple simultaneous users influence each other's results.

When User A searches "distributed systems" and clicks doc #42,
this immediately boosts doc #42 for ALL users currently searching
related terms — in real time via Server-Sent Events.

This demonstrates:
  - Distributed shared mutable state across HTTP sessions
  - Real-time push via SSE (no polling)
  - Collaborative information retrieval (active research area)
  - Eventually consistent score updates

Architecture:
  SharedRelevanceModel : thread-safe score store (shared across all sessions)
  CollaborativeRanker  : blends personal BM25 with crowd wisdom
  /collaborative       : search page with live "trending" feed
  /collab/stream       : SSE endpoint pushing live activity
  /collab/signal       : POST endpoint receiving clicks + queries
"""

import time
import json
import threading
import math
from collections import defaultdict, deque


class SharedRelevanceModel:
    """
    Thread-safe shared score store.
    All Flask worker threads read/write this simultaneously.

    Scores decay over time — recent signals matter more than old ones.
    This prevents one viral query from permanently dominating results.
    """

    DECAY_HALF_LIFE = 300   # scores halve every 5 minutes
    MAX_HISTORY     = 100   # keep last N events for the live feed

    def __init__(self):
        self._lock         = threading.RLock()
        self._doc_scores   = defaultdict(float)   # doc_id -> crowd score
        self._query_counts = defaultdict(int)      # query -> search count
        self._active_users = {}                    # session_id -> last_seen
        self._event_feed   = deque(maxlen=self.MAX_HISTORY)
        self._last_decay   = time.time()

    def record_query(self, query, session_id):
        """Record that a user searched for this query."""
        with self._lock:
            self._query_counts[query.lower()] += 1
            self._active_users[session_id]     = time.time()
            self._event_feed.appendleft({
                "type":      "search",
                "query":     query,
                "ts":        time.time(),
                "active":    self._count_active_users(),
            })

    def record_click(self, doc_id, query, position, session_id):
        """
        Record a click — boost this document's crowd score.
        Higher boost for clicks at lower positions (position 1 > position 10).
        """
        position_weight = 1.0 / math.log2(position + 2)  # DCG-style weight
        boost           = position_weight * 2.0

        with self._lock:
            self._apply_decay()
            self._doc_scores[doc_id]       += boost
            self._active_users[session_id]  = time.time()
            self._event_feed.appendleft({
                "type":     "click",
                "doc_id":   doc_id,
                "query":    query,
                "boost":    round(boost, 3),
                "ts":       time.time(),
                "active":   self._count_active_users(),
            })

    def get_crowd_boost(self, doc_id):
        """Return the current crowd relevance boost for a document."""
        with self._lock:
            self._apply_decay()
            return self._doc_scores.get(doc_id, 0.0)

    def get_trending_queries(self, limit=5):
        """Return the most searched queries right now."""
        with self._lock:
            return sorted(
                self._query_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

    def get_trending_docs(self, url_map, limit=5):
        """Return the most clicked documents right now."""
        with self._lock:
            self._apply_decay()
            top = sorted(
                self._doc_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            result = []
            for doc_id, score in top:
                meta = url_map.get(str(doc_id), {})
                result.append({
                    "doc_id": doc_id,
                    "title":  meta.get("title", "Unknown"),
                    "url":    meta.get("url", ""),
                    "score":  round(score, 3),
                })
            return result

    def get_live_feed(self, limit=20):
        """Return recent events for the live activity feed."""
        with self._lock:
            return list(self._event_feed)[:limit]

    def get_active_user_count(self):
        with self._lock:
            return self._count_active_users()

    def get_stats(self):
        with self._lock:
            self._apply_decay()
            return {
                "active_users":    self._count_active_users(),
                "total_queries":   sum(self._query_counts.values()),
                "unique_queries":  len(self._query_counts),
                "scored_docs":     len(self._doc_scores),
                "trending_queries": [q for q, _ in
                                     self.get_trending_queries(5)],
            }

    def _count_active_users(self):
        """Users active in the last 5 minutes."""
        cutoff = time.time() - 300
        return sum(1 for t in self._active_users.values() if t > cutoff)

    def _apply_decay(self):
        """Exponential decay — called before every read/write."""
        now     = time.time()
        elapsed = now - self._last_decay
        if elapsed < 10:   # only decay every 10 seconds
            return
        decay_factor     = math.exp(-elapsed * math.log(2) / self.DECAY_HALF_LIFE)
        for doc_id in list(self._doc_scores.keys()):
            self._doc_scores[doc_id] *= decay_factor
            if self._doc_scores[doc_id] < 0.001:
                del self._doc_scores[doc_id]
        self._last_decay = now


# ── global shared model — one instance for the entire Flask app ───────────────
SHARED_MODEL = SharedRelevanceModel()
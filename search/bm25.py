"""
BM25 ranking — industry-standard probabilistic retrieval
=========================================================
Parameters:
    k1 = 1.5  (term frequency saturation)
    b  = 0.75 (document length normalisation)
"""

import math
from collections import defaultdict


K1 = 1.5
B  = 0.75


class BM25:
    def __init__(self, index, url_map, pagerank=None):
        """
        index    : inverted index {word: {doc_ids, tf}}
        url_map  : {str(doc_id): {url, title}}
        pagerank : {doc_id: score} or None
        """
        self.index     = index
        self.url_map   = url_map
        self.pagerank  = pagerank or {}
        self.N         = len(url_map)   # total documents

        # compute average document length from TF data
        doc_lengths = defaultdict(int)
        for word, entry in index.items():
            for doc_id_str, tf in entry["tf"].items():
                doc_lengths[int(doc_id_str)] += tf

        self.doc_lengths = dict(doc_lengths)
        self.avgdl = (sum(doc_lengths.values()) / self.N) if self.N > 0 else 1

    def score(self, query, top_n=20):
        """
        Score all documents for a multi-keyword query using BM25.
        Optionally blends in PageRank.
        Returns list of result dicts sorted by score.
        """
        terms = [t.lower().strip() for t in query.split() if len(t) >= 3]
        if not terms:
            return []

        scores = defaultdict(float)

        for term in terms:
            if term not in self.index:
                continue

            entry  = self.index[term]
            df     = len(entry["doc_ids"])             # document frequency
            idf    = math.log((self.N - df + 0.5) /
                               (df + 0.5) + 1)        # BM25 IDF

            for doc_id in entry["doc_ids"]:
                tf     = entry["tf"].get(str(doc_id), 0)
                dl     = self.doc_lengths.get(doc_id, self.avgdl)
                # BM25 term score
                tf_norm = (tf * (K1 + 1)) / \
                          (tf + K1 * (1 - B + B * dl / self.avgdl))
                scores[doc_id] += idf * tf_norm

        # blend with PageRank (20% weight)
        if self.pagerank:
            for doc_id in list(scores.keys()):
                pr = self.pagerank.get(doc_id, 0.0)
                scores[doc_id] = 0.80 * scores[doc_id] + 0.20 * pr * 10

        # rank and format
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_n]:
            meta = self.url_map.get(str(doc_id), {})
            pr   = self.pagerank.get(doc_id, 0.0)
            results.append({
                "doc_id":    doc_id,
                "url":       meta.get("url",   ""),
                "title":     meta.get("title", "No title"),
                "bm25":      round(score, 3),
                "pagerank":  round(pr, 4),
                "score":     round(score, 2),
            })
        return results
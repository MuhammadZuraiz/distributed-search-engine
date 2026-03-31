"""
Query expansion using co-occurrence + curated synonyms
========================================================
Expands query terms with related terms found in the index.

Two strategies:
  1. Curated synonym map — hand-picked domain synonyms
  2. Co-occurrence expansion — terms that appear together
     frequently in the same documents

Example:
  "car"     -> also search "automobile vehicle"
  "fast"    -> also search "quick rapid speed"
  "crawler" -> also search "spider scraper bot"
"""

from collections import defaultdict


# ── curated synonym map ───────────────────────────────────────────────────────
# Domain-specific synonyms relevant to our crawled content

SYNONYMS = {
    # computing
    "distributed":  ["parallel", "concurrent", "decentralised"],
    "parallel":     ["concurrent", "distributed", "simultaneous"],
    "fault":        ["failure", "error", "crash", "tolerance"],
    "crawler":      ["spider", "scraper", "bot", "indexer"],
    "search":       ["query", "lookup", "retrieval", "find"],
    "database":     ["storage", "repository", "datastore"],
    "network":      ["internet", "web", "connectivity"],
    "algorithm":    ["method", "procedure", "technique"],
    "performance":  ["speed", "efficiency", "throughput", "latency"],
    "scalability":  ["scaling", "growth", "capacity"],
    "cluster":      ["nodes", "servers", "machines"],
    "replication":  ["copy", "duplicate", "backup", "redundancy"],

    # general
    "fast":         ["quick", "rapid", "speed", "swift"],
    "car":          ["automobile", "vehicle", "motor"],
    "book":         ["novel", "text", "publication", "literature"],
    "learn":        ["study", "education", "training"],
    "build":        ["construct", "create", "develop", "implement"],
    "small":        ["tiny", "little", "minimal", "compact"],
    "large":        ["big", "huge", "massive", "extensive"],
    "language":     ["programming", "syntax", "code"],
    "python":       ["programming", "scripting", "language"],
    "data":         ["information", "dataset", "records"],
    "history":      ["historical", "past", "timeline", "chronicle"],
    "science":      ["research", "study", "knowledge", "academic"],
}


class QueryExpander:
    def __init__(self, index, max_expansion_terms=3):
        """
        index              : inverted index {word: {doc_ids, tf}}
        max_expansion_terms: max synonyms to add per term
        """
        self.index    = index
        self.max_exp  = max_expansion_terms
        self._co_occur = None   # built lazily

    def _build_cooccurrence(self, sample_size=2000):
        """
        Build a co-occurrence map from the index.
        For each pair of terms that appear in the same document,
        increment their co-occurrence count.
        Only samples top terms by frequency for performance.
        """
        print("[expander] Building co-occurrence map...")

        # get top terms by document frequency
        top_terms = sorted(
            self.index.items(),
            key=lambda x: len(x[1]["doc_ids"]),
            reverse=True
        )[:sample_size]

        # invert: doc_id -> [terms]
        doc_terms = defaultdict(list)
        for term, entry in top_terms:
            for doc_id in entry["doc_ids"]:
                doc_terms[doc_id].append(term)

        # count co-occurrences
        co_occur = defaultdict(lambda: defaultdict(int))
        for doc_id, terms in doc_terms.items():
            for i, t1 in enumerate(terms):
                for t2 in terms[i+1:i+6]:   # window of 5
                    if t1 != t2:
                        co_occur[t1][t2] += 1
                        co_occur[t2][t1] += 1

        self._co_occur = co_occur
        print(f"[expander] Co-occurrence map built: "
              f"{len(co_occur)} terms")

    def get_synonyms(self, term):
        """Return list of expansion terms for a single term."""
        term   = term.lower().strip()
        result = []

        # 1. curated synonyms first
        for syn in SYNONYMS.get(term, []):
            if syn in self.index and syn not in result:
                result.append(syn)
            if len(result) >= self.max_exp:
                return result

        # 2. co-occurrence expansion
        if self._co_occur is None:
            self._build_cooccurrence()

        co_terms = sorted(
            self._co_occur.get(term, {}).items(),
            key=lambda x: x[1],
            reverse=True
        )
        for co_term, count in co_terms:
            if co_term not in result and co_term != term and count >= 3:
                result.append(co_term)
            if len(result) >= self.max_exp:
                break

        return result[:self.max_exp]

    def expand(self, query):
        """
        Expand a query with related terms.
        Returns (expanded_query, expansions_map).

        expanded_query : original + expansion terms joined
        expansions_map : {original_term: [expansion_terms]}
        """
        terms      = [t.lower() for t in query.split() if len(t) >= 3]
        expansions = {}
        extra      = []

        for term in terms:
            syns = self.get_synonyms(term)
            if syns:
                expansions[term] = syns
                extra.extend(syns)

        expanded = query + (" " + " ".join(extra) if extra else "")
        return expanded, expansions
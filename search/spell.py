"""
Spell correction using Levenshtein edit distance
==================================================
If a query term isn't in the index, find the closest
matching term using edit distance <= 2.

Example:
  "distribted" -> "distributed"
  "compuing"   -> "computing"
  "phyton"     -> "python"
"""


def edit_distance(s1, s2):
    """
    Standard dynamic programming Levenshtein distance.
    Returns minimum edits (insert, delete, substitute) to transform s1 -> s2.
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # substitute
                )

    return dp[m][n]


class SpellCorrector:
    def __init__(self, index):
        """
        index : the inverted index {word: {doc_ids, tf}}
        Builds a fast lookup structure from all indexed terms.
        """
        # group terms by first two characters for fast candidate filtering
        self.terms      = set(index.keys())
        self.term_freq  = {w: len(v["doc_ids"]) for w, v in index.items()}
        self._prefix_map = {}   # first 2 chars -> [terms]

        for term in self.terms:
            key = term[:2] if len(term) >= 2 else term[:1]
            if key not in self._prefix_map:
                self._prefix_map[key] = []
            self._prefix_map[key].append(term)

    def correct(self, word, max_distance=2):
        """
        Return the best correction for word, or None if word is already valid
        or no close match found.

        Strategy:
          1. If word is in index, no correction needed.
          2. Find candidates sharing first 1-2 chars (fast filter).
          3. Compute edit distance only for candidates.
          4. Return highest-frequency term within max_distance.
        """
        word = word.lower().strip()

        # already correct
        if word in self.terms:
            return None

        # too short to correct meaningfully
        if len(word) < 3:
            return None

        # gather candidates — terms starting with same first char
        # (edit distance >= 2 for terms with completely different start)
        candidates = []
        for prefix_len in [2, 1]:
            key  = word[:prefix_len]
            cands = self._prefix_map.get(key, [])
            # only consider terms of similar length (±3 chars)
            for c in cands:
                if abs(len(c) - len(word)) <= 3:
                    candidates.append(c)
            if candidates:
                break

        if not candidates:
            return None

        # find closest by edit distance, break ties by term frequency
        best_word = None
        best_dist = max_distance + 1
        best_freq = 0

        for candidate in candidates:
            dist = edit_distance(word, candidate)
            freq = self.term_freq.get(candidate, 0)
            if dist < best_dist or (dist == best_dist and freq > best_freq):
                best_dist = dist
                best_word = candidate
                best_freq = freq

        if best_dist <= max_distance:
            return best_word
        return None

    def correct_query(self, query):
        """
        Correct all terms in a multi-word query.
        Returns (corrected_query, corrections_made) where
        corrections_made is a dict of {original: corrected}.
        """
        terms       = query.lower().split()
        corrected   = []
        corrections = {}

        for term in terms:
            fix = self.correct(term)
            if fix:
                corrected.append(fix)
                corrections[term] = fix
            else:
                corrected.append(term)

        return " ".join(corrected), corrections
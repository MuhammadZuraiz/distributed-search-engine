"""
Prefix Trie for search autocomplete
=====================================
Built from all indexed terms at startup.
Returns top-N completions for any prefix in <5ms.
"""

from collections import defaultdict


class TrieNode:
    __slots__ = ["children", "is_end", "count"]

    def __init__(self):
        self.children = {}
        self.is_end   = False
        self.count    = 0      # how many docs contain this term (for ranking)


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, count=1):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count  = count

    def search_prefix(self, prefix, max_results=8):
        """
        Return up to max_results words that start with prefix,
        sorted by document frequency descending.
        """
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS from this node to collect all completions
        results = []
        self._dfs(node, prefix.lower(), results)
        results.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in results[:max_results]]

    def _dfs(self, node, current, results):
        if node.is_end:
            results.append((current, node.count))
        for char, child in node.children.items():
            self._dfs(child, current + char, results)


def build_trie(index):
    """Build a trie from the inverted index. index = {word: {doc_ids, tf}}"""
    trie = Trie()
    for word, entry in index.items():
        doc_count = len(entry.get("doc_ids", []))
        trie.insert(word, doc_count)
    return trie
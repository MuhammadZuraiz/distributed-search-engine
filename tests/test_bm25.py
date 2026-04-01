"""Tests for BM25 ranking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_index(docs):
    """Build a minimal inverted index from a list of (doc_id, text) tuples."""
    from collections import defaultdict
    index = defaultdict(lambda: {"doc_ids": [], "tf": {}})
    for doc_id, text in docs:
        for word in text.lower().split():
            if doc_id not in index[word]["doc_ids"]:
                index[word]["doc_ids"].append(doc_id)
            index[word]["tf"][str(doc_id)] = \
                index[word]["tf"].get(str(doc_id), 0) + 1
    return dict(index)


def make_url_map(docs):
    return {str(doc_id): {"url": f"http://test.com/{doc_id}",
                          "title": text[:30], "snippet": ""}
            for doc_id, text in docs}


def test_bm25_higher_tf_scores_higher():
    from search.bm25 import BM25
    docs    = [(0, "python python python programming"),
               (1, "python language")]
    index   = make_index(docs)
    url_map = make_url_map(docs)
    ranker  = BM25(index, url_map)
    results = ranker.score("python")
    assert results[0]["doc_id"] == 0, "Doc with higher TF should rank first"


def test_bm25_returns_empty_for_unknown_term():
    from search.bm25 import BM25
    docs    = [(0, "distributed computing systems")]
    index   = make_index(docs)
    url_map = make_url_map(docs)
    ranker  = BM25(index, url_map)
    results = ranker.score("unknownxyz123")
    assert results == []


def test_bm25_multi_term_query():
    from search.bm25 import BM25
    docs = [
        (0, "distributed computing parallel systems"),
        (1, "cooking recipes food kitchen"),
        (2, "distributed parallel processing"),
    ]
    index   = make_index(docs)
    url_map = make_url_map(docs)
    ranker  = BM25(index, url_map)
    results = ranker.score("distributed parallel")
    ids     = [r["doc_id"] for r in results]
    assert 0 in ids and 2 in ids
    assert 1 not in ids


def test_bm25_respects_top_n():
    from search.bm25 import BM25
    docs    = [(i, f"python programming language {i}") for i in range(20)]
    index   = make_index(docs)
    url_map = make_url_map(docs)
    ranker  = BM25(index, url_map)
    assert len(ranker.score("python", top_n=5))  == 5
    assert len(ranker.score("python", top_n=10)) == 10


def test_bm25_scores_are_positive():
    from search.bm25 import BM25
    docs    = [(0, "web crawler spider indexer"),
               (1, "search engine query retrieval")]
    index   = make_index(docs)
    url_map = make_url_map(docs)
    ranker  = BM25(index, url_map)
    results = ranker.score("web search")
    for r in results:
        assert r["score"] >= 0
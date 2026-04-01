"""Tests for prefix trie autocomplete."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.trie import Trie, build_trie


def test_trie_basic_prefix():
    t = Trie()
    t.insert("distributed", 10)
    t.insert("distribution", 8)
    t.insert("district", 5)
    t.insert("python", 20)
    results = t.search_prefix("dist")
    assert "distributed"  in results
    assert "distribution" in results
    assert "district"     in results
    assert "python"       not in results


def test_trie_empty_prefix():
    t = Trie()
    t.insert("hello", 1)
    assert t.search_prefix("") == []
    assert t.search_prefix("z") == []


def test_trie_exact_match():
    t = Trie()
    t.insert("computing", 5)
    results = t.search_prefix("computing")
    assert "computing" in results


def test_trie_sorted_by_frequency():
    t = Trie()
    t.insert("parallel",    count=100)
    t.insert("parrot",      count=2)
    t.insert("parameter",   count=50)
    results = t.search_prefix("par", max_results=3)
    assert results[0] == "parallel"    # highest count first
    assert results[1] == "parameter"


def test_trie_max_results_respected():
    t = Trie()
    for i in range(20):
        t.insert(f"term{i:02d}", count=i)
    results = t.search_prefix("term", max_results=5)
    assert len(results) == 5


def test_build_trie_from_index():
    index = {
        "distributed": {"doc_ids": [0,1,2], "tf": {}},
        "computing":   {"doc_ids": [0,1],   "tf": {}},
        "python":      {"doc_ids": [0],      "tf": {}},
    }
    trie    = build_trie(index)
    results = trie.search_prefix("dis")
    assert "distributed" in results
"""Tests for consistent hash ring."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hash_ring import HashRing


def test_hash_ring_returns_valid_node():
    ring = HashRing([1, 2, 3, 4, 5, 6])
    node = ring.get_node("https://en.wikipedia.org/wiki/Python")
    assert node in [1, 2, 3, 4, 5, 6]


def test_same_domain_always_same_node():
    ring = HashRing([1, 2, 3, 4, 5, 6])
    urls = [
        "https://en.wikipedia.org/wiki/Python",
        "https://en.wikipedia.org/wiki/Java",
        "https://en.wikipedia.org/wiki/Distributed_computing",
        "https://en.wikipedia.org/wiki/Web_crawler",
    ]
    nodes = [ring.get_node_for_url(u) for u in urls]
    assert len(set(nodes)) == 1, "All Wikipedia URLs must map to same node"


def test_different_domains_distributed():
    ring    = HashRing([1, 2, 3, 4, 5, 6])
    domains = [
        "https://en.wikipedia.org/wiki/A",
        "https://books.toscrape.com/",
        "https://quotes.toscrape.com/",
        "https://crawler-test.com/",
    ]
    nodes = [ring.get_node_for_url(d) for d in domains]
    # not all should map to the same node
    assert len(set(nodes)) > 1, "Different domains should spread across nodes"


def test_virtual_nodes_count():
    ring = HashRing([1, 2, 3], virtual_nodes=100)
    assert len(ring.ring) == 300   # 3 nodes x 100 virtual nodes


def test_distribution_evenness():
    """Each node should get roughly 1/N of assignments."""
    ring    = HashRing([1, 2, 3, 4, 5, 6], virtual_nodes=150)
    counts  = {i: 0 for i in range(1, 7)}
    samples = [f"https://site{i}.com/page{j}"
               for i in range(50) for j in range(10)]
    for url in samples:
        node = ring.get_node_for_url(url)
        counts[node] += 1

    total  = sum(counts.values())
    avg    = total / 6
    for node, count in counts.items():
        ratio = count / avg
        assert 0.3 < ratio < 3.0, \
            f"Node {node} got {count} ({ratio:.2f}x avg) — too imbalanced"
"""
Gossip Protocol for Distributed URL Deduplication
===================================================
Implements epidemic/gossip-style information dissemination
across MPI worker nodes.

How it works:
  1. Each worker maintains its own local Bloom filter of
     URLs it has personally crawled or seen in links
  2. Every GOSSIP_INTERVAL pages, a worker asks master for
     a random peer's filter bytes
  3. Worker merges the peer filter into its own via bitwise OR
  4. Over time, all workers converge on global knowledge
     WITHOUT a central coordinator

This demonstrates:
  - Eventual consistency (not immediate, but guaranteed convergence)
  - Epidemic algorithms (information spreads like a virus)
  - CAP theorem trade-off: we choose Availability + Partition
    tolerance over strict Consistency

Used in production by:
  - Amazon DynamoDB  (membership + failure detection)
  - Apache Cassandra (cluster state propagation)
  - Bitcoin          (transaction propagation)

Convergence guarantee:
  After O(log N) gossip rounds, all N nodes have seen
  all information with high probability.
"""

import time
import math
from utils.bloom_filter import BloomFilter


GOSSIP_INTERVAL = 5      # gossip every N URLs crawled by this worker
GOSSIP_ROUNDS   = 3       # number of peers to contact per gossip round


class WorkerGossipState:
    """
    Per-worker gossip state — lives in each worker process.
    Tracks local Bloom filter + gossip statistics.
    """

    def __init__(self, rank, capacity=500_000, fp_rate=0.001):
        self.rank         = rank
        self.local_bloom  = BloomFilter(capacity, fp_rate)
        self.urls_crawled = 0
        self.gossip_count = 0
        self.merges_done  = 0
        self.urls_blocked_by_gossip = 0   # URLs this worker avoided due to peer info

    def add_url(self, url):
        """Record that this worker has seen this URL."""
        self.local_bloom.add(url)
        self.urls_crawled += 1

    def check_url(self, url):
        """
        Check if this URL is known (locally or via gossip).
        Returns True if URL was probably already seen by ANY worker.
        """
        return url in self.local_bloom

    def should_gossip(self):
        """Return True if it's time to gossip."""
        return (self.urls_crawled > 0 and
                self.urls_crawled % GOSSIP_INTERVAL == 0)

    def merge_peer_filter(self, peer_bytes, peer_rank):
        """
        Merge a peer's Bloom filter into our local one.
        Bitwise OR — we learn everything the peer knows.
        """
        from bitarray import bitarray
        peer_bits = bitarray()
        peer_bits.frombytes(peer_bytes)

        if len(peer_bits) == len(self.local_bloom.bits):
            before_count = self.local_bloom.bits.count()
            self.local_bloom.bits |= peer_bits
            after_count  = self.local_bloom.bits.count()
            new_bits     = after_count - before_count
            self.merges_done += 1
            return new_bits   # how many new URLs we learned
        return 0

    def get_filter_bytes(self):
        """Serialise local filter for transmission."""
        return self.local_bloom.bits.tobytes()

    def stats(self):
        return {
            "rank":                    self.rank,
            "urls_crawled":            self.urls_crawled,
            "gossip_rounds":           self.gossip_count,
            "merges_done":             self.merges_done,
            "urls_blocked_by_gossip":  self.urls_blocked_by_gossip,
            "filter_load_factor":      round(self.local_bloom.load_factor(), 4),
            "filter_fp_rate":          round(self.local_bloom.current_fp_rate(), 6),
        }


class MasterGossipCoordinator:
    """
    Runs in the master process.
    Stores each worker's latest filter bytes.
    When a worker requests a peer, master picks a random
    live worker and returns their filter.

    The master is NOT a coordinator in the traditional sense —
    it's just a rendezvous point. Workers could gossip
    peer-to-peer directly in a full implementation.
    """

    def __init__(self, worker_ranks):
        self.worker_ranks   = list(worker_ranks)
        self.filter_store   = {}   # rank -> bytes
        self.gossip_count   = {r: 0 for r in worker_ranks}
        self.last_update    = {r: 0 for r in worker_ranks}

    def store_filter(self, rank, filter_bytes):
        """Store a worker's filter update."""
        self.filter_store[rank]  = filter_bytes
        self.last_update[rank]   = time.time()
        self.gossip_count[rank] += 1

    def get_random_peer_filter(self, requesting_rank):
        """
        Return a random peer's filter bytes.
        Excludes the requesting worker itself.
        Prefers recently-updated filters.
        """
        import random
        candidates = [
            r for r in self.worker_ranks
            if r != requesting_rank and r in self.filter_store
        ]
        if not candidates:
            return None, None

        # prefer most recently updated peer
        peer = max(candidates, key=lambda r: self.last_update[r])
        return peer, self.filter_store[peer]

    def get_convergence_stats(self):
        """
        Estimate convergence: how similar are the filters?
        Two identical filters = fully converged.
        """
        if len(self.filter_store) < 2:
            return {"converged": False, "similarity": 0}

        from bitarray import bitarray
        filters = list(self.filter_store.values())
        bits_list = []
        for fb in filters:
            b = bitarray()
            b.frombytes(fb)
            bits_list.append(b)

        # pairwise similarity = bits in common / total bits set
        similarities = []
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                intersection = (bits_list[i] & bits_list[j]).count()
                union        = (bits_list[i] | bits_list[j]).count()
                sim          = intersection / union if union > 0 else 1.0
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        return {
            "converged":        avg_sim > 0.95,
            "similarity":       round(avg_sim, 4),
            "filters_stored":   len(self.filter_store),
            "gossip_counts":    dict(self.gossip_count),
        }
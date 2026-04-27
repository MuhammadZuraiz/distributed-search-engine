"""
Bloom Filter — probabilistic URL deduplication
================================================
A Bloom filter answers "have I seen this URL before?"
with zero false negatives and a tunable false positive rate.

How it works:
  - A bit array of size m, initially all zeros
  - k independent hash functions
  - INSERT: hash URL k times, set those k bits to 1
  - QUERY:  hash URL k times, if ALL k bits are 1 → "probably seen"
            if ANY bit is 0 → "definitely not seen"

Memory comparison:
  Python set()   : ~200 bytes per URL string → 200MB for 1M URLs
  Bloom filter   : ~2MB for 1M URLs at 1% false positive rate

False positive rate formula:
  p ≈ (1 - e^(-kn/m))^k
  where n=items inserted, m=bit array size, k=hash functions

This is used in:
  Google Bigtable  — avoid disk lookups for missing keys
  Chrome           — safe browsing URL checking
  Akamai CDN       — one-hit-wonder detection
  Bitcoin          — transaction deduplication
  Apache Cassandra — SSTable membership testing
"""

import math
import hashlib
from bitarray import bitarray


class BloomFilter:
    def __init__(self, capacity=1_000_000, false_positive_rate=0.001):
        """
        capacity          : expected number of URLs to insert
        false_positive_rate: desired FP rate (0.001 = 0.1%)

        Optimal parameters calculated automatically:
          m = -n * ln(p) / (ln(2)^2)   bit array size
          k = m/n * ln(2)               number of hash functions
        """
        self.capacity           = capacity
        self.false_positive_rate = false_positive_rate

        # optimal bit array size
        self.m = self._optimal_m(capacity, false_positive_rate)

        # optimal number of hash functions
        self.k = self._optimal_k(self.m, capacity)

        # initialise bit array
        self.bits    = bitarray(self.m)
        self.bits.setall(0)
        self.n_items = 0   # items inserted so far

        print(f"[bloom] Initialised: capacity={capacity:,}, "
              f"FP rate={false_positive_rate:.3%}, "
              f"bits={self.m:,} ({self.m/8/1024/1024:.2f}MB), "
              f"hash_functions={self.k}")

    @staticmethod
    def _optimal_m(n, p):
        """Optimal bit array size for n items at false positive rate p."""
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m, n):
        """Optimal number of hash functions."""
        return max(1, int(m / n * math.log(2)))

    def _hash_positions(self, item):
        """
        Generate k bit positions for an item using double hashing.
        Uses SHA-256 and MD5 as two independent hash bases,
        then combines them: h_i(x) = h1(x) + i * h2(x)
        This avoids computing k separate hash functions.
        """
        item_bytes = item.encode("utf-8")

        h1 = int(hashlib.sha256(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.md5(item_bytes).hexdigest(), 16)

        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item):
        """Insert an item into the filter."""
        for pos in self._hash_positions(item):
            self.bits[pos] = 1
        self.n_items += 1

    def __contains__(self, item):
        """
        Return True if item was probably inserted.
        Return False if item was definitely NOT inserted.
        """
        return all(self.bits[pos] for pos in self._hash_positions(item))

    def current_fp_rate(self):
        """
        Estimate the current false positive rate based on
        how many items have been inserted so far.
        """
        if self.n_items == 0:
            return 0.0
        exponent = -self.k * self.n_items / self.m
        return (1 - math.exp(exponent)) ** self.k

    def memory_bytes(self):
        """Return memory used by the bit array in bytes."""
        return self.m // 8

    def load_factor(self):
        """Fraction of bits set to 1."""
        return self.bits.count() / self.m

    def stats(self):
        """Return a stats dict for logging/display."""
        return {
            "capacity":        self.capacity,
            "inserted":        self.n_items,
            "bit_array_size":  self.m,
            "hash_functions":  self.k,
            "memory_mb":       round(self.memory_bytes() / 1024 / 1024, 3),
            "load_factor":     round(self.load_factor(), 4),
            "fp_rate_current": round(self.current_fp_rate(), 6),
            "fp_rate_target":  self.false_positive_rate,
        }


class DistributedBloomFilter:
    """
    Bloom filter designed for MPI distribution.
    The master holds the authoritative filter.
    Workers send URL batches, master checks + updates.

    Also supports merging two filters (for gossip protocol later):
      merged = filter_a | filter_b
    Merge = bitwise OR of the two bit arrays.
    This is the key property that makes Bloom filters
    suitable for distributed gossip — you can combine
    any two filters without losing information.
    """

    def __init__(self, capacity=1_000_000, false_positive_rate=0.001):
        self.filter = BloomFilter(capacity, false_positive_rate)

    def check_and_add(self, url):
        """
        Atomic check-then-add.
        Returns True if URL is NEW (not seen before).
        Returns False if URL was already seen (or false positive).
        """
        if url in self.filter:
            return False   # already seen
        self.filter.add(url)
        return True        # new URL

    def check_batch(self, urls):
        """
        Check a batch of URLs at once.
        Returns list of (url, is_new) tuples.
        """
        results = []
        for url in urls:
            is_new = self.check_and_add(url)
            results.append((url, is_new))
        return results

    def merge(self, other_filter):
        """
        Merge another BloomFilter into this one via bitwise OR.
        Used for gossip protocol — workers share their filters
        and merge to converge on global knowledge.
        """
        if self.filter.m != other_filter.filter.m:
            raise ValueError("Cannot merge filters of different sizes")
        self.filter.bits |= other_filter.filter.bits
        self.filter.n_items += other_filter.filter.n_items

    def to_bytes(self):
        """Serialise bit array to bytes for MPI transmission."""
        return self.filter.bits.tobytes()

    def from_bytes(self, data):
        """Deserialise bit array from bytes received via MPI."""
        self.filter.bits = bitarray()
        self.filter.bits.frombytes(data)

    def stats(self):
        return self.filter.stats()
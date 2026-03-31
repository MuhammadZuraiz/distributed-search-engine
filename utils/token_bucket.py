"""
Token bucket rate limiter
==========================
Each domain gets its own bucket with a fixed capacity and refill rate.
Workers request tokens from master before fetching.
Master grants or denies based on bucket state.

PDC concept demonstrated:
  - Shared mutable state (buckets live only in master)
  - Synchronisation (all workers coordinate through master)
  - Message passing (TOKEN_REQUEST / TOKEN_GRANT tags)
  - This is how AWS API Gateway, Nginx, and Cloudflare enforce rate limits
"""

import time


class TokenBucket:
    """
    Single token bucket for one domain.
    capacity  : max tokens (burst size)
    rate      : tokens added per second
    """
    def __init__(self, capacity=5, rate=2.0):
        self.capacity  = capacity
        self.rate      = rate          # tokens per second
        self.tokens    = capacity      # start full
        self.last_refill = time.time()

    def refill(self):
        """Add tokens based on elapsed time since last refill."""
        now     = time.time()
        elapsed = now - self.last_refill
        added   = elapsed * self.rate
        self.tokens      = min(self.capacity, self.tokens + added)
        self.last_refill = now

    def consume(self):
        """
        Try to consume one token.
        Returns (granted, wait_seconds).
        If granted=True, fetch is allowed immediately.
        If granted=False, wait_seconds is how long until a token is available.
        """
        self.refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        else:
            # how long until next token
            wait = (1.0 - self.tokens) / self.rate
            return False, round(wait, 3)

    def status(self):
        self.refill()
        return {
            "tokens":   round(self.tokens, 2),
            "capacity": self.capacity,
            "rate":     self.rate,
        }


class RateLimiterRegistry:
    """
    Holds one TokenBucket per domain.
    Lives entirely in the master process.
    """

    # per-domain rate configs (tokens/sec, burst)
    DOMAIN_RATES = {
        "en.wikipedia.org":    (1.0, 3),   # 1 req/sec, burst 3
        "books.toscrape.com":  (3.0, 6),   # 3 req/sec, burst 6
        "quotes.toscrape.com": (3.0, 6),
        "crawler-test.com":    (5.0, 10),  # fast test site
    }
    DEFAULT_RATE = (1.0, 4)

    def __init__(self):
        self._buckets = {}

    def _get_bucket(self, domain):
        if domain not in self._buckets:
            rate, cap = self.DOMAIN_RATES.get(domain, self.DEFAULT_RATE)
            self._buckets[domain] = TokenBucket(capacity=cap, rate=rate)
        return self._buckets[domain]

    def request_token(self, domain):
        """
        Called by master when a worker requests a token for a domain.
        Returns (granted, wait_seconds).
        """
        bucket = self._get_bucket(domain)
        return bucket.consume()

    def get_all_status(self):
        """Return status of all buckets — for dashboard/metrics."""
        return {
            domain: bucket.status()
            for domain, bucket in self._buckets.items()
        }
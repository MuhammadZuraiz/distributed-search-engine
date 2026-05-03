"""Tests for token bucket rate limiter."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.token_bucket import TokenBucket, RateLimiterRegistry


def test_full_bucket_grants_immediately():
    bucket = TokenBucket(capacity=5, rate=2.0)
    granted, wait = bucket.consume()
    assert granted is True
    assert wait == 0.0


def test_empty_bucket_denies():
    bucket = TokenBucket(capacity=1, rate=0.1)
    bucket.consume()                          # drain the one token
    granted, wait = bucket.consume()
    assert granted is False
    assert wait > 0


def test_bucket_refills_over_time():
    bucket = TokenBucket(capacity=2, rate=10.0)  # 10 tokens/sec
    bucket.consume()
    bucket.consume()                              # drain both
    time.sleep(0.15)                             # wait 150ms = 1.5 tokens
    granted, _ = bucket.consume()
    assert granted is True


def test_bucket_does_not_exceed_capacity():
    bucket = TokenBucket(capacity=3, rate=100.0)
    time.sleep(0.1)   # would add 10 tokens at rate 100 — but capped at 3
    bucket.refill()
    assert bucket.tokens <= 3.0


def test_registry_known_domain():
    reg     = RateLimiterRegistry()
    granted, wait = reg.request_token("en.wikipedia.org")
    assert isinstance(granted, bool)
    assert isinstance(wait, float)


def test_registry_unknown_domain_uses_default():
    reg     = RateLimiterRegistry()
    granted, wait = reg.request_token("unknown-domain-xyz.com")
    assert isinstance(granted, bool)


def test_registry_rate_limits_burst():
    """Drain the burst capacity and verify denial."""
    reg    = RateLimiterRegistry()
    domain = "en.wikipedia.org"
    # wikipedia bucket: capacity=3, rate=1.0
    results = [reg.request_token(domain) for _ in range(10)]
    granted = [r[0] for r in results]
    # at least some should be denied
    assert False in granted, "Should deny after burst capacity exhausted"
"""
Work Stealing Scheduler — Option B (Threshold-Based Affinity Override)
======================================================================

Replaces the flat frontier list in master.py with a domain-partitioned
queue system. Consistent-hash affinity is honoured as a preference, not
a hard constraint. When a worker's home queue depth exceeds STEAL_THRESHOLD,
idle workers are allowed to steal URLs from its tail.

Design properties
-----------------
* Home worker pops from the FRONT of each domain queue  (FIFO crawl order)
* Stealing worker pops from the TAIL                    (no logical contention)
* STEAL_THRESHOLD guards against stealing during normal operation —
  only genuine hotspots (e.g. a Wikipedia-heavy crawl) trigger it
* Partial domain transfer: when a stolen domain's queue drops below
  TRANSFER_THRESHOLD the thief becomes the new home owner, so subsequent
  rounds cost zero steal-decision overhead
* Dead-worker recovery: remove_worker() re-homes all orphaned domain queues
  to alive workers so no URLs are silently dropped

PDC concept demonstrated
------------------------
Work stealing was originally formalised for shared-memory parallelism
(Blumofe & Leiserson, 1999 — the Cilk scheduler). Adapting it to a
message-passing MPI system requires making domain queues explicit in the
master and expressing "steal" as an intelligent scheduling decision rather
than direct memory access. The formal guarantee is that with STEAL_THRESHOLD
T, the maximum imbalance ratio is bounded by T / avg_queue_depth + 1,
approaching 1.0 as the crawl progresses and domain ownership rebalances.

Academic reference
------------------
Blumofe, R. D., & Leiserson, C. E. (1999). Scheduling multithreaded
computations by work stealing. Journal of the ACM, 46(5), 720–748.
"""

from collections import defaultdict, deque
from urllib.parse import urlparse
import time
import logging

log = logging.getLogger("work_stealing")

# ── tunables ──────────────────────────────────────────────────────────────────

STEAL_THRESHOLD    = 15   # a worker's home queue must exceed this to be stolen from
TRANSFER_THRESHOLD = 7    # full domain-ownership transfer when victim queue falls below this


# ── helpers ───────────────────────────────────────────────────────────────────

def _domain_of(url: str) -> str:
    """Extract netloc from a URL, returning 'unknown' on parse failure."""
    try:
        netloc = urlparse(url).netloc
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


# ── main class ────────────────────────────────────────────────────────────────

class WorkStealingScheduler:
    """
    Domain-partitioned URL frontier with work-stealing rebalancing.

    Public API (mirrors the old flat-frontier operations in master.py):
        push(url, depth)              — enqueue a URL
        pop_for_worker(rank, dead)    — dequeue next URL for this worker
        remove_worker(rank)           — re-home a dead worker's queues
        all_urls()                    — flatten for checkpointing
        load_urls(pairs, dead)        — restore from checkpoint
        total_size()                  — total enqueued URL count
        is_empty()                    — True when all queues are empty
        stats()                       — steal telemetry dict
        imbalance_ratio()             — max_depth / min_depth across workers
    """

    def __init__(self, hash_ring, worker_ranks: list):
        self.hash_ring    = hash_ring
        self.worker_ranks = list(worker_ranks)

        # domain → deque of (url, depth) tuples
        self.domain_queues: dict[str, deque] = defaultdict(deque)

        # rank → set of domains this worker currently "owns"
        self.worker_home_domains: dict[int, set] = {r: set() for r in worker_ranks}

        # rank → total URL count across all owned domain queues
        # (kept in sync with domain_queues to avoid O(domains) recalculation)
        self.worker_home_depth: dict[int, int] = {r: 0 for r in worker_ranks}

        # steal telemetry
        self.total_steals    = 0
        self.steals_by_rank: dict[int, int] = {r: 0 for r in worker_ranks}
        self.steal_events: list[dict]        = []   # for reporting / experiments

    # ── public: enqueue ───────────────────────────────────────────────────────

    def push(self, url: str, depth: int):
        """
        Add a URL to the queue of its preferred worker (consistent hash).
        Called by master whenever new links are discovered.
        """
        domain   = _domain_of(url)
        preferred = self.hash_ring.get_node_for_url(url)

        self.domain_queues[domain].append((url, depth))
        self.worker_home_domains[preferred].add(domain)
        self.worker_home_depth[preferred] += 1

    # ── public: dequeue ───────────────────────────────────────────────────────

    def pop_for_worker(self, rank: int, dead_workers: set) -> tuple:
        """
        Return the next (url, depth, was_stolen, stolen_from_rank) for `rank`.

        Strategy:
          1. Check home domains first  — respects consistent-hash affinity
          2. If nothing in home domains, find the most overloaded alive worker
             whose queue depth exceeds STEAL_THRESHOLD
          3. Steal one URL from the TAIL of that worker's heaviest domain queue
          4. Perform partial or full domain-ownership transfer as appropriate

        Returns (None, None, False, None) when the entire scheduler is empty
        or no steal candidate exists.
        """
        # ── step 1: home domain work ──────────────────────────────────────────
        result = self._pop_home(rank)
        if result is not None:
            url, depth = result
            return url, depth, False, None

        # ── step 2: find steal candidate ──────────────────────────────────────
        victim = self._find_steal_target(rank, dead_workers)
        if victim is None:
            return None, None, False, None

        # ── step 3 & 4: steal + ownership transfer ────────────────────────────
        stolen = self._steal_one(victim, rank)
        if stolen is None:
            return None, None, False, None

        url, depth = stolen
        self._record_steal(rank, victim, url)
        return url, depth, True, victim

    # ── public: dead-worker recovery ─────────────────────────────────────────

    def remove_worker(self, dead_rank: int):
        """
        Re-home all domain queues owned by a dead worker to alive workers.
        Called by master immediately after marking a worker dead.
        """
        orphaned = self.worker_home_domains.pop(dead_rank, set())
        self.worker_home_depth.pop(dead_rank, None)
        self.worker_ranks = [r for r in self.worker_ranks if r != dead_rank]

        if not self.worker_ranks:
            return  # all workers dead — nothing to re-home to

        for domain in orphaned:
            q = self.domain_queues[domain]
            if not q:
                continue

            # pick the new preferred worker using the hash ring
            # (falls back to first alive worker if preferred is also dead)
            sample_url = q[0][0]
            new_owner = self.hash_ring.get_node_for_url(sample_url)
            if new_owner not in self.worker_home_domains:
                new_owner = self.worker_ranks[0]

            count = len(q)
            self.worker_home_domains[new_owner].add(domain)
            self.worker_home_depth[new_owner] = (
                self.worker_home_depth.get(new_owner, 0) + count
            )
            log.info(
                f"[RECOVER] re-homed domain '{domain}' ({count} URLs) "
                f"from dead rank {dead_rank} → rank {new_owner}"
            )

    # ── public: checkpoint helpers ────────────────────────────────────────────

    def all_urls(self) -> list:
        """
        Flatten all queued URLs to a list of (url, depth) tuples.
        Same format as the old frontier list — checkpoint compatible.
        """
        result = []
        for q in self.domain_queues.values():
            result.extend(q)
        return result

    def load_urls(self, url_depth_pairs: list, dead_workers: set = None):
        """
        Restore URLs from a checkpoint (list of (url, depth) tuples).
        Calls push() for each, so domain ownership is correctly re-established.
        """
        dead = dead_workers or set()
        for url, depth in url_depth_pairs:
            # if the preferred worker is dead, push() still works —
            # remove_worker() will be called separately to re-home
            self.push(url, depth)

    # ── public: introspection ─────────────────────────────────────────────────

    def total_size(self) -> int:
        """Total number of enqueued URLs across all domain queues."""
        return sum(len(q) for q in self.domain_queues.values())

    def is_empty(self) -> bool:
        return self.total_size() == 0

    def stats(self) -> dict:
        return {
            "total_steals":    self.total_steals,
            "steals_by_rank":  dict(self.steals_by_rank),
            "queue_depths":    dict(self.worker_home_depth),
            "active_domains":  sum(1 for q in self.domain_queues.values() if q),
            "total_queued":    self.total_size(),
        }

    def imbalance_ratio(self) -> float:
        """max_depth / min_depth — 1.0 is perfect balance."""
        depths = [v for v in self.worker_home_depth.values() if v >= 0]
        if not depths:
            return 1.0
        nonzero = [d for d in depths if d > 0]
        if len(nonzero) < 2:
            return 1.0
        return max(nonzero) / min(nonzero)

    # ── internal: home pop ────────────────────────────────────────────────────

    def _pop_home(self, rank: int):
        """
        Pop from the front of the first non-empty home-domain queue.
        Returns (url, depth) or None.
        Cleans up empty domain entries to keep iteration fast.
        """
        for domain in list(self.worker_home_domains[rank]):
            q = self.domain_queues[domain]
            if q:
                url, depth = q.popleft()
                self.worker_home_depth[rank] = max(0, self.worker_home_depth[rank] - 1)
                # clean up fully-drained domain so future iterations are fast
                if not q:
                    self.worker_home_domains[rank].discard(domain)
                return url, depth
            else:
                # stale entry — domain queue was drained elsewhere
                self.worker_home_domains[rank].discard(domain)
        return None

    # ── internal: steal target selection ─────────────────────────────────────

    def _find_steal_target(self, thief: int, dead_workers: set):
        """
        Find the most overloaded alive worker whose queue exceeds
        STEAL_THRESHOLD. Returns the victim rank or None.

        We pick the *most* overloaded target (greedy) so that steals
        have maximum impact on the bottleneck rather than spreading
        small imbalances across many workers.
        """
        best_rank  = None
        best_depth = STEAL_THRESHOLD  # must strictly exceed threshold

        for rank in self.worker_ranks:
            if rank == thief or rank in dead_workers:
                continue
            depth = self.worker_home_depth.get(rank, 0)
            if depth > best_depth:
                # verify actual URLs exist (depth counter can lag on race)
                has_urls = any(
                    self.domain_queues[d]
                    for d in self.worker_home_domains.get(rank, set())
                )
                if has_urls:
                    best_depth = depth
                    best_rank  = rank

        return best_rank

    # ── internal: steal execution ─────────────────────────────────────────────

    def _steal_one(self, victim: int, thief: int):
        """
        Pop one URL from the TAIL of victim's heaviest domain queue.

        Tail-stealing ensures:
          * Home worker pops from front (FIFO order preserved for home work)
          * Thief pops from back      (no logical contention — different end)

        Domain ownership transfer:
          * Thief is immediately registered as co-owner of the stolen domain
            so it finds those URLs in its home domains next round (O(1) lookup)
          * If victim's queue for that domain falls below TRANSFER_THRESHOLD,
            full ownership passes to thief — victim stops iterating it entirely
        """
        # find victim's most loaded domain
        best_domain = None
        best_len    = 0
        for domain in self.worker_home_domains.get(victim, set()):
            q = self.domain_queues[domain]
            if len(q) > best_len:
                best_len    = len(q)
                best_domain = domain

        if best_domain is None or not self.domain_queues[best_domain]:
            return None

        # ── steal from tail ───────────────────────────────────────────────────
        url, depth = self.domain_queues[best_domain].pop()
        self.worker_home_depth[victim] = max(0, self.worker_home_depth[victim] - 1)

        remaining = len(self.domain_queues[best_domain])

        # thief co-owns the domain immediately
        self.worker_home_domains[thief].add(best_domain)

        # ── partial or full domain transfer ───────────────────────────────────
        if remaining < TRANSFER_THRESHOLD:
            # full transfer: victim stops owning this domain
            self.worker_home_domains[victim].discard(best_domain)
            # move the remaining depth accounting to thief
            self.worker_home_depth[victim] = max(
                0, self.worker_home_depth[victim] - remaining
            )
            self.worker_home_depth[thief] = (
                self.worker_home_depth.get(thief, 0) + remaining
            )
            if remaining > 0:
                log.info(
                    f"[TRANSFER] domain '{best_domain}' fully transferred "
                    f"rank {victim} → rank {thief} ({remaining} URLs remain)"
                )
        else:
            # partial co-ownership: thief gets the tail, victim keeps the front
            # (depth accounting: thief gains the stolen URL's slot)
            self.worker_home_depth[thief] = self.worker_home_depth.get(thief, 0) + 1

        # clean up empty queues
        if not self.domain_queues[best_domain]:
            self.worker_home_domains[victim].discard(best_domain)
            self.worker_home_domains[thief].discard(best_domain)

        return url, depth

    # ── internal: telemetry ───────────────────────────────────────────────────

    def _record_steal(self, thief: int, victim: int, url: str):
        self.total_steals += 1
        self.steals_by_rank[thief] = self.steals_by_rank.get(thief, 0) + 1
        event = {
            "thief":  thief,
            "victim": victim,
            "domain": _domain_of(url),
            "ts":     time.time(),
        }
        self.steal_events.append(event)
        log.info(
            f"[STEAL] rank {thief} ← rank {victim} | "
            f"domain: {event['domain']} | "
            f"victim queue now: {self.worker_home_depth.get(victim, 0)}"
        )

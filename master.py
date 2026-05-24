"""
Master Node (MPI Rank 0)
=========================
Manages the URL frontier, assigns tasks to workers,
collects results, and shuts workers down when done.

Phase 1 — Work Stealing (Option B, Threshold-Based Affinity Override)
-----------------------------------------------------------------------
The flat frontier list is replaced with WorkStealingScheduler, which
partitions URLs by domain and allows idle workers to steal from the
tail of overloaded workers' queues when queue depth exceeds
STEAL_THRESHOLD. Directly addresses the measured 6.77x imbalance
in Wikipedia-heavy crawls.

Phase 2 — Speculative Execution
---------------------------------
SpeculationManager monitors in-progress task durations against a
per-domain p95 baseline built from master-measured round-trip times.
When a task exceeds SPECULATION_FACTOR x p95, the next idle worker
receives a speculative copy of the same URL. First result wins;
duplicate is silently discarded.

Priority order in TAG_READY handler
--------------------------------------
  1. Speculative re-issue   (straggler detected, p95 baseline ready)
  2. Home domain / steal    (Phase 1 scheduler handles both)
  3. WAIT                   (all workers busy, frontier non-empty)
  4. DONE                   (frontier empty, no active workers)

Run:
    mpiexec -n 7 python master.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os
import pickle
import time
from urllib.parse import urlparse
from mpi4py import MPI

from utils.parser import is_allowed
from utils.logger import get_logger
from db.database import init_db, insert_document, log_crawl
from utils.metrics import record_queue_depth
from utils.token_bucket import RateLimiterRegistry
from utils.hash_ring import HashRing
from utils.bloom_filter import DistributedBloomFilter
from utils.gossip import MasterGossipCoordinator
from utils.work_stealing import WorkStealingScheduler       # Phase 1
from utils.speculation import SpeculationManager            # Phase 2

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Distributed_computing",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/History",
    "https://en.wikipedia.org/wiki/Philosophy",
    "https://en.wikipedia.org/wiki/Economics",
    "https://en.wikipedia.org/wiki/Biology",
    "https://books.toscrape.com/",
    "https://quotes.toscrape.com/",
    "https://crawler-test.com/",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Data_science",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Astronomy",
    "https://en.wikipedia.org/wiki/Medicine",
    "https://en.wikipedia.org/wiki/Chemistry",
    "https://en.wikipedia.org/wiki/Geography",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/Art",
    "https://en.wikipedia.org/wiki/Music",
    "https://en.wikipedia.org/wiki/Sports",
]

MAX_PAGES           = 25000
MAX_DEPTH           = 4
CHECKPOINT_INTERVAL = 200

# ── MPI message tags ──────────────────────────────────────────────────────────
TAG_READY          = 1
TAG_TASK           = 2
TAG_RESULT         = 3
TAG_DONE           = 4
TAG_HEARTBEAT      = 5
TAG_TOKEN_REQUEST  = 6
TAG_TOKEN_GRANT    = 7
TAG_GOSSIP_REQUEST = 8
TAG_GOSSIP_REPLY   = 9

HEARTBEAT_TIMEOUT = 15
POLL_INTERVAL     = 0.05

STATS_FILE      = "data/live_stats.json"
CHECKPOINT_FILE = "data/checkpoint.pkl"


# ── helpers ───────────────────────────────────────────────────────────────────

def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc if netloc else "unknown"
    except Exception:
        return "unknown"


def write_stats(pages_crawled, queue_size, elapsed, worker_counts,
                dead_workers, max_pages, last_event="",
                steal_stats=None, spec_stats=None):
    pps = pages_crawled / elapsed if elapsed > 0 else 0
    workers = {
        str(r): {"pages": c, "dead": r in dead_workers}
        for r, c in worker_counts.items()
    }
    stats = {
        "pages_crawled": pages_crawled,
        "queue_size":    queue_size,
        "elapsed":       round(elapsed, 1),
        "pages_per_sec": round(pps, 2),
        "max_pages":     max_pages,
        "workers":       workers,
        "last_event":    last_event,
        "steal_stats":   steal_stats or {},
        "spec_stats":    spec_stats  or {},
    }
    os.makedirs("data", exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def save_checkpoint(frontier_pairs, visited, doc_id):
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({
            "frontier": frontier_pairs,
            "visited":  visited,
            "doc_id":   doc_id,
        }, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            data = pickle.load(f)
        return data["frontier"], data["visited"], data["doc_id"]
    except Exception:
        return None


def normalise_url(url):
    from urllib.parse import urlunparse
    p = urlparse(url.strip())
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def save_page(doc_id, data, worker_rank=None, duration_ms=0):
    insert_document(
        doc_id      = doc_id,
        url         = data["url"],
        title       = data["title"],
        text        = data["text"],
        snippet     = data.get("snippet", ""),
        links       = data["links"],
        worker_rank = worker_rank,
    )
    log_crawl(
        url         = data["url"],
        doc_id      = doc_id,
        worker_rank = worker_rank or 0,
        status      = "ok",
        duration_ms = duration_ms,
        depth       = data.get("depth", 0),
    )


# ── master ────────────────────────────────────────────────────────────────────

log = get_logger("master")


def run_master(comm):
    init_db()
    num_workers = comm.Get_size() - 1
    log.info(f"Master started — {num_workers} worker(s)")

    # ── hash ring ─────────────────────────────────────────────────────────────
    worker_ranks = list(range(1, comm.Get_size()))
    hash_ring    = HashRing(worker_ranks)

    # ── Phase 1: work stealing scheduler ─────────────────────────────────────
    scheduler = WorkStealingScheduler(hash_ring, worker_ranks)

    # ── Phase 2: speculation manager ─────────────────────────────────────────
    speculation = SpeculationManager()

    log.info("Work-stealing scheduler ready  (STEAL_THRESHOLD=15, tail-stealing)")
    log.info("Speculation manager ready      (factor=2.0xp95, min_samples=5)")

    # ── checkpoint or fresh start ─────────────────────────────────────────────
    checkpoint = load_checkpoint()
    if checkpoint:
        frontier_pairs, visited, doc_id = checkpoint
        bloom = DistributedBloomFilter(capacity=10_000_000, false_positive_rate=0.001)
        for url in visited:
            bloom.filter.add(url)
        scheduler.load_urls(frontier_pairs)
        log.info(f"Resumed: {doc_id} done, {scheduler.total_size()} queued")
    else:
        bloom   = DistributedBloomFilter(capacity=10_000_000, false_positive_rate=0.001)
        visited = set()
        doc_id  = 0
        for url in SEED_URLS:
            norm = normalise_url(url)
            scheduler.push(norm, 0)
            visited.add(norm)
            bloom.filter.add(norm)
        log.info(f"Fresh crawl — {scheduler.total_size()} seeds enqueued")

    # ── runtime state ─────────────────────────────────────────────────────────
    active_workers = 0
    start_time     = time.time()

    # rank -> (url, depth, doc_id, start_time)
    # Both original and speculative workers share this dict (keyed by rank).
    # Two entries can carry the same doc_id when speculation is active —
    # that is correct: they are two different workers on the same task.
    in_progress:  dict[int, tuple] = {}

    last_seen     = {r: time.time() for r in worker_ranks}
    dead_workers  = set()
    worker_counts = {r: 0 for r in worker_ranks}   # accepted results per worker

    rate_limiter       = RateLimiterRegistry()
    gossip_coordinator = MasterGossipCoordinator(worker_ranks)

    # ── main event loop ───────────────────────────────────────────────────────
    while True:

        # ── fault detection ───────────────────────────────────────────────────
        now = time.time()
        for rank in list(in_progress.keys()):
            if rank in dead_workers:
                continue
            if now - last_seen[rank] > HEARTBEAT_TIMEOUT:
                url, depth, failed_doc_id, _ = in_progress.pop(rank)
                dead_workers.add(rank)
                active_workers -= 1
                num_workers    -= 1
                log.info(f"[FAULT] rank {rank} timed out — requeuing doc {failed_doc_id}")

                # Phase 1: re-queue in-flight, redistribute orphaned domains
                scheduler.push(url, depth)
                scheduler.remove_worker(rank)

                # Phase 2: if the dead task had a speculative copy still running,
                # that worker's result will arrive later and handle_result() will
                # accept it as the first arrival. No special handling needed.

                write_stats(
                    doc_id, scheduler.total_size(), now - start_time,
                    worker_counts, dead_workers, MAX_PAGES,
                    last_event=f"[FAULT] rank {rank} timed out",
                    steal_stats=scheduler.stats(),
                    spec_stats=speculation.stats(),
                )

        if num_workers == 0 and active_workers == 0:
            break

        # ── non-blocking probe ────────────────────────────────────────────────
        status      = MPI.Status()
        has_message = comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if not has_message:
            time.sleep(POLL_INTERVAL)
            if int(time.time()) % 10 == 0:
                record_queue_depth(scheduler.total_size(), active_workers)
            continue

        sender = status.Get_source()
        tag    = status.Get_tag()

        if sender in dead_workers:
            comm.recv(source=sender, tag=MPI.ANY_TAG)
            continue

        # ── heartbeat ─────────────────────────────────────────────────────────
        if tag == TAG_HEARTBEAT:
            comm.recv(source=sender, tag=TAG_HEARTBEAT)
            last_seen[sender] = time.time()

        # ── rate-limit token ──────────────────────────────────────────────────
        elif tag == TAG_TOKEN_REQUEST:
            domain        = comm.recv(source=sender, tag=TAG_TOKEN_REQUEST)
            granted, wait = rate_limiter.request_token(domain)
            comm.send((granted, wait), dest=sender, tag=TAG_TOKEN_GRANT)

        # ── gossip ────────────────────────────────────────────────────────────
        elif tag == TAG_GOSSIP_REQUEST:
            data = comm.recv(source=sender, tag=TAG_GOSSIP_REQUEST)
            if isinstance(data, bytes):
                gossip_coordinator.store_filter(sender, data)
                peer_rank, peer_bytes = gossip_coordinator.get_random_peer_filter(sender)
                if peer_bytes:
                    comm.send((peer_rank, peer_bytes), dest=sender, tag=TAG_GOSSIP_REPLY)
                    conv = gossip_coordinator.get_convergence_stats()
                    log.info(
                        f"[gossip] rank {sender} <-> rank {peer_rank} "
                        f"(similarity: {conv['similarity']:.2%})"
                    )
                else:
                    comm.send((None, None), dest=sender, tag=TAG_GOSSIP_REPLY)

        # ── worker ready ──────────────────────────────────────────────────────
        elif tag == TAG_READY:
            comm.recv(source=sender, tag=TAG_READY)
            last_seen[sender] = time.time()
            now = time.time()

            # ── Priority 1: Phase 2 — speculative re-issue ────────────────────
            #
            # Check for stragglers before touching the scheduler.
            # Speculation costs nothing from the URL budget (doc_id not
            # incremented) and uses an idle worker to hedge against tail latency.
            # The threshold is domain-aware: 2x p95 of that domain's fetch time.
            #
            straggler = speculation.find_straggler(in_progress, now)

            if straggler is not None:
                orig_rank, spec_url, spec_depth, spec_doc_id = straggler

                comm.send(
                    (spec_url, spec_depth, spec_doc_id, False),
                    dest=sender, tag=TAG_TASK,
                )

                # Register in in_progress for heartbeat monitoring.
                # in_progress is rank-keyed: two entries share spec_doc_id
                # as a value. That is intentional and safe.
                in_progress[sender] = (spec_url, spec_depth, spec_doc_id, now)
                active_workers += 1

                # Lock the speculation — handle_result() uses this record to
                # identify which worker won and to discard the loser.
                speculation.record_speculation(spec_doc_id, orig_rank, sender)

                # Do NOT increment doc_id    — reusing an existing slot.
                # Do NOT increment worker_counts[sender] — only count it if
                # this result is accepted (done in TAG_RESULT below).

                write_stats(
                    doc_id, scheduler.total_size(), now - start_time,
                    worker_counts, dead_workers, MAX_PAGES,
                    last_event=(
                        f"[SPEC] doc {spec_doc_id} re-issued "
                        f"rank {sender} ← rank {orig_rank}"
                    ),
                    steal_stats=scheduler.stats(),
                    spec_stats=speculation.stats(),
                )

            # ── Priority 2 & 3: Phase 1 — work stealing ───────────────────────
            #
            # pop_for_worker() tries home domains first (priority 2), then
            # steals from the most overloaded alive worker (priority 3).
            # Both cases return (url, depth, was_stolen, stolen_from).
            #
            elif doc_id < MAX_PAGES and not scheduler.is_empty():
                url, depth, was_stolen, stolen_from = scheduler.pop_for_worker(
                    sender, dead_workers
                )

                if url is not None:
                    comm.send((url, depth, doc_id, was_stolen), dest=sender, tag=TAG_TASK)
                    in_progress[sender] = (url, depth, doc_id, now)
                    worker_counts[sender] += 1
                    active_workers += 1

                    assigned_doc_id  = doc_id
                    doc_id          += 1

                    if was_stolen:
                        log.info(
                            f"-> [STEAL] [{assigned_doc_id}] "
                            f"rank {sender} <- rank {stolen_from}: {url}"
                        )
                    else:
                        log.info(f"-> [{assigned_doc_id}] rank {sender}: {url}")

                    if doc_id % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(scheduler.all_urls(), visited, doc_id)
                        log.info(
                            f"[ckpt] {doc_id} pages | "
                            f"imbalance {scheduler.imbalance_ratio():.2f}x | "
                            f"steals {scheduler.total_steals} | "
                            f"spec {speculation.total_speculated}"
                        )

                    write_stats(
                        doc_id, scheduler.total_size(), now - start_time,
                        worker_counts, dead_workers, MAX_PAGES,
                        last_event=(
                            f"[STEAL] [{assigned_doc_id}] rank {sender}"
                            if was_stolen else
                            f"-> [{assigned_doc_id}] rank {sender}"
                        ),
                        steal_stats=scheduler.stats(),
                        spec_stats=speculation.stats(),
                    )

                else:
                    # Transient: scheduler non-empty but pop returned None.
                    # All queues are below STEAL_THRESHOLD and this worker
                    # has no home URLs. Resolves in the next poll cycle.
                    comm.send(("__WAIT__", 0, -1, False), dest=sender, tag=TAG_TASK)

            # ── Priority 4: DONE ──────────────────────────────────────────────
            elif active_workers == 0:
                comm.send(None, dest=sender, tag=TAG_DONE)
                num_workers -= 1
                log.info(f"[done] rank {sender} shut down ({num_workers} left)")
                if num_workers == 0:
                    break

            # ── Priority 5: WAIT ──────────────────────────────────────────────
            else:
                comm.send(("__WAIT__", 0, -1, False), dest=sender, tag=TAG_TASK)

        # ── result received ───────────────────────────────────────────────────
        elif tag == TAG_RESULT:
            message = comm.recv(source=sender, tag=TAG_RESULT)
            last_seen[sender] = time.time()

            # Capture start_time BEFORE popping — feeds the speculation p95 baseline.
            task_info  = in_progress.pop(sender, None)
            task_start = task_info[3] if task_info else time.time()
            active_workers -= 1

            result_doc_id, url, title, text, snippet, links, depth = message

            # Master-measured round-trip: fetch + parse + MPI serialisation.
            # More complete than the worker-measured HTTP-only duration in metrics.py.
            duration_ms = (time.time() - task_start) * 1000
            domain      = _domain_of(url)

            # ── Phase 2: duplicate suppression ────────────────────────────────
            #
            # handle_result() returns True  for the first arrival of this doc_id
            #                  returns False for any subsequent (duplicate) result.
            #
            # Gating all save/link logic behind this check is the complete and
            # correct implementation of duplicate suppression. There is no
            # cancellation message — we let the losing worker finish its fetch
            # (it already has) and simply discard the result here.
            #
            is_first_result = speculation.handle_result(result_doc_id, sender)

            if not is_first_result:
                # The winning result is already in the DB.
                # worker_counts is NOT updated — speculative assignments skip
                # the worker_counts increment in TAG_READY, so the counts stay
                # consistent regardless of which worker wins.
                log.info(
                    f"<- [DUP/SKIP] doc {result_doc_id} "
                    f"rank {sender} discarded"
                )
                continue

            # ── accepted result: save to DB ───────────────────────────────────
            if title not in ("Failed", "Blocked"):
                save_page(
                    result_doc_id,
                    {
                        "url":     url,
                        "title":   title,
                        "text":    text,
                        "snippet": snippet,
                        "links":   links,
                        "depth":   depth,
                    },
                    worker_rank = sender,
                    duration_ms = duration_ms,
                )
                log.info(
                    f"<- [{result_doc_id}] rank {sender} "
                    f"({duration_ms:.0f}ms): {title[:50]}"
                )

                # Phase 2: update p95 baseline for this domain.
                # Only successful fetches — failures skew p95 upward and
                # would cause over-speculation on healthy connections.
                speculation.record_completion(result_doc_id, domain, duration_ms)

                # Credit the winning worker (original or speculative).
                if sender in worker_counts:
                    worker_counts[sender] += 1

            else:
                log.info(f"<- [{result_doc_id}] rank {sender} ({title}): {url[:60]}")

            # ── enqueue discovered links ───────────────────────────────────────
            if depth < MAX_DEPTH:
                new_links = 0
                for link in links:
                    norm = normalise_url(link)
                    if bloom.check_and_add(norm) and is_allowed(norm):
                        visited.add(norm)
                        if doc_id + scheduler.total_size() < MAX_PAGES * 3:
                            scheduler.push(norm, depth + 1)
                            new_links += 1
                if new_links:
                    log.info(
                        f"   +{new_links} links | "
                        f"scheduler: {scheduler.total_size()} total"
                    )

            write_stats(
                doc_id, scheduler.total_size(), time.time() - start_time,
                worker_counts, dead_workers, MAX_PAGES,
                last_event=f"<- [{result_doc_id}] rank {sender}: {title[:35]}",
                steal_stats=scheduler.stats(),
                spec_stats=speculation.stats(),
            )

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info(
        f"Crawl complete: {doc_id} pages in {elapsed:.1f}s "
        f"({doc_id/elapsed:.2f} pages/sec)"
    )

    log.info("Worker page distribution (accepted results):")
    for rank, count in sorted(worker_counts.items()):
        marker = " [DEAD]" if rank in dead_workers else ""
        log.info(f"  rank {rank}: {count:4d} pages{marker}")

    max_w = max(worker_counts.values()) if worker_counts else 1
    min_w = min(
        (v for r, v in worker_counts.items() if r not in dead_workers and v > 0),
        default=1,
    )
    log.info(
        f"Page imbalance ratio: {max_w / max(min_w, 1):.2f}x  "
        f"(baseline without optimisation: 6.77x)"
    )

    # Phase 1
    ws = scheduler.stats()
    log.info("-" * 60)
    log.info("Phase 1  Work Stealing")
    log.info(f"  Total steals          : {ws['total_steals']}")
    log.info(f"  Steals by rank        : {ws['steals_by_rank']}")
    log.info(f"  Final imbalance ratio : {scheduler.imbalance_ratio():.2f}x")

    # Phase 2
    ss = speculation.stats()
    log.info("-" * 60)
    log.info("Phase 2  Speculative Execution")
    log.info(f"  Speculated            : {ss['total_speculated']}")
    log.info(f"  Spec wins             : {ss['total_wins']}")
    log.info(f"  Duplicates discarded  : {ss['total_duplicates']}")
    log.info(f"  Win rate              : {ss['win_rate']:.1%}")
    log.info("  Domain p95 baselines:")
    for domain, p95 in sorted(ss["domain_p95_ms"].items(), key=lambda x: -x[1]):
        n = ss["samples_per_domain"].get(domain, 0)
        log.info(f"    {domain:<45} p95={p95:7.0f}ms  (n={n})")

    # Bloom filter
    bf = bloom.stats()
    log.info("-" * 60)
    log.info("Bloom Filter")
    log.info(f"  Inserted     : {bf['inserted']:,}")
    log.info(f"  Memory       : {bf['memory_mb']} MB")
    log.info(f"  Load factor  : {bf['load_factor']:.2%}")
    log.info(f"  FP rate est. : {bf['fp_rate_current']:.4%}")
    log.info("=" * 60)

    write_stats(
        doc_id, 0, elapsed, worker_counts, dead_workers, MAX_PAGES,
        last_event="Crawl complete",
        steal_stats=scheduler.stats(),
        spec_stats=speculation.stats(),
    )

    for rank in dead_workers:
        try:
            comm.send(None, dest=rank, tag=TAG_DONE)
        except Exception:
            pass

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Checkpoint cleared")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        run_master(comm)
    else:
        from worker import run_worker
        run_worker(comm)

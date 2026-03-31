"""
Phase 2 — Master Node (MPI Rank 0)
====================================
Manages the URL frontier, assigns tasks to workers,
collects results, and shuts workers down when done.

Run:
    mpiexec -n 4 python master.py
    (rank 0 = master, ranks 1-3 = workers)
"""
# add near the top of master.py
# ── configuration ─────────────────────────────────────────────────────────────

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os
import pickle
import time

from mpi4py import MPI

from utils.parser import is_allowed
from utils.logger import get_logger

from db.database import init_db, insert_document, log_crawl

from utils.metrics import record_queue_depth

from utils.token_bucket import RateLimiterRegistry

from utils.hash_ring import HashRing

SEED_URLS = [
    "https://books.toscrape.com/",
    "https://quotes.toscrape.com/",
    "https://crawler-test.com/",
    "https://en.wikipedia.org/wiki/Distributed_computing",
    "https://en.wikipedia.org/wiki/Web_crawler",
]

MAX_PAGES          = 1000
MAX_DEPTH          = 3
OUTPUT_DIR         = "data/crawled_distributed"

TAG_READY          = 1
TAG_TASK           = 2
TAG_RESULT         = 3
TAG_DONE           = 4
TAG_HEARTBEAT      = 5

TAG_TOKEN_REQUEST = 6    # worker -> master: "may I fetch this domain?"
TAG_TOKEN_GRANT   = 7    # master -> worker: "yes" or "wait Xs"

HEARTBEAT_TIMEOUT  = 15
POLL_INTERVAL      = 0.05

STATS_FILE         = "data/live_stats.json"

CHECKPOINT_FILE = "data/checkpoint.pkl"

def write_stats(pages_crawled, queue_size, elapsed, worker_counts,
                dead_workers, max_pages, last_event=""):
    pps = pages_crawled / elapsed if elapsed > 0 else 0
    workers = {
        str(r): {"pages": c, "dead": r in dead_workers}
        for r, c in worker_counts.items()
    }
    stats = {
        "pages_crawled":  pages_crawled,
        "queue_size":     queue_size,
        "elapsed":        round(elapsed, 1),
        "pages_per_sec":  round(pps, 2),
        "max_pages":      max_pages,
        "workers":        workers,
        "last_event":     last_event,
    }
    os.makedirs("data", exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f)

def save_checkpoint(frontier, visited, doc_id):
    """Save crawl state to disk every CHECKPOINT_INTERVAL pages."""
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({
            "frontier": list(frontier),
            "visited":  visited,
            "doc_id":   doc_id
        }, f)


def load_checkpoint():
    """Load saved crawl state. Returns (frontier, visited, doc_id) or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            data = pickle.load(f)
        return data["frontier"], data["visited"], data["doc_id"]
    except Exception:
        return None

# add this helper function above run_master()
def normalise_url(url):
    """Canonicalise a URL so duplicates with minor differences are caught."""
    from urllib.parse import urlparse, urlunparse
    p = urlparse(url.strip())
    # lowercase scheme and host, strip trailing slash from path
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
        worker_rank = worker_rank
    )
    log_crawl(
        url         = data["url"],
        doc_id      = doc_id,
        worker_rank = worker_rank or 0,
        status      = "ok",
        duration_ms = duration_ms,
        depth       = data.get("depth", 0)
    )

# ─────────────────────────────────────────────────────────────────────────────

log = get_logger("master")

def run_master(comm):
    init_db()
    num_workers   = comm.Get_size() - 1
    total_workers = num_workers
    log.info(f"Master started — {num_workers} worker(s), timeout={HEARTBEAT_TIMEOUT}s")

    CHECKPOINT_INTERVAL = 50

    # ── try to resume from checkpoint ────────────────────────────────────────
    checkpoint = load_checkpoint()
    # after checkpoint load, make sure OUTPUT_DIR doc_id starts correctly
    if checkpoint:
        frontier, visited, doc_id = checkpoint
        log.info(f"Resuming from checkpoint: {doc_id} pages already crawled, "
                 f"{len(frontier)} URLs in frontier")
    else:
        frontier = []
        visited  = set()
        doc_id   = 0
        for url in SEED_URLS:
            norm = normalise_url(url)
            frontier.append((norm, 0))
            visited.add(norm)
        log.info("Starting fresh crawl")

    # active_workers starts at 0 always — never restored from checkpoint
    active_workers = 0
    start_time     = time.time()

    in_progress   = {}
    last_seen     = {r: time.time() for r in range(1, comm.Get_size())}
    dead_workers  = set()
    worker_counts = {r: 0 for r in range(1, comm.Get_size())}

    # ── consistent hash ring ──────────────────────────────────────────────────
    worker_ranks = [r for r in range(1, comm.Get_size())]
    hash_ring    = HashRing(worker_ranks)
    log.info(f"Hash ring initialised with {len(worker_ranks)} nodes, "
             f"{hash_ring.virtual_nodes} virtual nodes each")

    # ── token bucket rate limiter ─────────────────────────────────────────────
    rate_limiter = RateLimiterRegistry()
    log.info("Token bucket rate limiter initialised")

    while True:

        # ── check for dead workers ────────────────────────────────────────
        now = time.time()
        for rank in list(in_progress.keys()):
            if rank in dead_workers:
                continue
            elapsed_since = now - last_seen[rank]
            if elapsed_since > HEARTBEAT_TIMEOUT:
                url, depth, failed_doc_id, _ = in_progress.pop(rank)
                dead_workers.add(rank)
                active_workers -= 1
                num_workers    -= 1
                log.info(f"  [FAULT] rank {rank} timed out after {elapsed_since:.1f}s")
                log.info(f"  [REQUEUE] re-queuing doc {failed_doc_id}: {url}")
                frontier.insert(0, (url, depth))

                # ── write_stats call 1: fault detected ────────────────────
                write_stats(doc_id, len(frontier), time.time() - start_time,
                            worker_counts, dead_workers, MAX_PAGES,
                            last_event=f"[FAULT] rank {rank} timed out")

                if num_workers == 0 and active_workers == 0:
                    break

        # ── non-blocking probe ────────────────────────────────────────────
        status      = MPI.Status()
        has_message = comm.iprobe(source=MPI.ANY_SOURCE,
                                  tag=MPI.ANY_TAG,
                                  status=status)
        if not has_message:
            time.sleep(POLL_INTERVAL)
            # record queue depth every 10 seconds
            if int(time.time()) % 10 == 0:
                record_queue_depth(len(frontier), active_workers)
            continue

        sender = status.Get_source()
        tag    = status.Get_tag()

        if sender in dead_workers:
            comm.recv(source=sender, tag=MPI.ANY_TAG)
            continue

        # ── heartbeat ─────────────────────────────────────────────────────
        if tag == TAG_HEARTBEAT:
            comm.recv(source=sender, tag=TAG_HEARTBEAT)
            last_seen[sender] = time.time()
            continue

        # ── token request ─────────────────────────────────────────────────
        elif tag == TAG_TOKEN_REQUEST:
            domain  = comm.recv(source=sender, tag=TAG_TOKEN_REQUEST)
            granted, wait = rate_limiter.request_token(domain)
            comm.send((granted, wait), dest=sender, tag=TAG_TOKEN_GRANT)
            if not granted:
                log.info(f"  [rate] rank {sender} must wait {wait}s "
                         f"for {domain}")

        # ── worker ready ──────────────────────────────────────────────────
        elif tag == TAG_READY:
            comm.recv(source=sender, tag=TAG_READY)
            last_seen[sender] = time.time()

            if frontier and doc_id < MAX_PAGES:
                # ── try to honour consistent hashing ─────────────────────
                # find a URL whose preferred worker is the sender
                assigned_url   = None
                assigned_depth = None
                preferred_idx  = None

                for i, (url, depth) in enumerate(frontier):
                    preferred = hash_ring.get_node_for_url(url)
                    if preferred == sender or preferred in dead_workers:
                        assigned_url   = url
                        assigned_depth = depth
                        preferred_idx  = i
                        break

                # fallback: just take the front of the queue
                if assigned_url is None:
                    assigned_url, assigned_depth = frontier[0]
                    preferred_idx = 0

                # remove from frontier
                del frontier[preferred_idx]

                comm.send((assigned_url, assigned_depth, doc_id),
                          dest=sender, tag=TAG_TASK)
                in_progress[sender] = (assigned_url, assigned_depth,
                                       doc_id, time.time())
                worker_counts[sender] += 1
                active_workers += 1
                log.info(f"  -> assigned [{doc_id}] to rank {sender} "
                         f"(hash:{hash_ring.get_node_for_url(assigned_url)}): "
                         f"{assigned_url}")
                doc_id += 1
                # ── checkpoint every N pages ──────────────────────────────────
                if doc_id % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(frontier, visited, doc_id)
                    log.info(f"  [checkpoint] saved at {doc_id} pages")

                write_stats(doc_id, len(frontier), time.time() - start_time,
                            worker_counts, dead_workers, MAX_PAGES,
                            last_event=f"-> [{doc_id}] assigned to rank {sender}")

            elif active_workers == 0:
                comm.send(None, dest=sender, tag=TAG_DONE)
                num_workers -= 1
                log.info(f"  [done] rank {sender} shut down ({num_workers} remaining)")
                if num_workers == 0:
                    break
            else:
                comm.send(("__WAIT__", 0, -1), dest=sender, tag=TAG_TASK)

        # ── result received ───────────────────────────────────────────────
        elif tag == TAG_RESULT:
            message = comm.recv(source=sender, tag=TAG_RESULT)
            last_seen[sender] = time.time()
            in_progress.pop(sender, None)
            active_workers -= 1

            result_doc_id, url, title, text, snippet, links, depth = message

            if title not in ("Failed", "Blocked"):
                save_page(result_doc_id, {
                    "url":     url,
                    "title":   title,
                    "text":    text,
                    "snippet": snippet,
                    "links":   links,
                    "depth":   depth,
                }, worker_rank=sender)
                log.info(f"  <- received [{result_doc_id}] from rank {sender}: {title[:40]}")

            if depth < MAX_DEPTH:
                for link in links:
                    norm = normalise_url(link)
                    if norm not in visited and is_allowed(norm):
                        visited.add(norm)
                        if doc_id + len(frontier) < MAX_PAGES * 3:
                            frontier.append((norm, depth + 1))

            # ── write_stats call 3: result received ───────────────────────
            write_stats(doc_id, len(frontier), time.time() - start_time,
                        worker_counts, dead_workers, MAX_PAGES,
                        last_event=f"<- [{result_doc_id}] from rank {sender}: {title[:35]}")

    # ── summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info("-" * 60)
    log.info(f"Distributed crawl complete: {doc_id} pages in {elapsed:.1f}s")
    log.info(f"Pages/second : {doc_id / elapsed:.2f}")
    if dead_workers:
        log.info(f"Dead workers detected: {dead_workers}")
    else:
        log.info("Fault tolerance: no worker failures detected")
    log.info("Worker page distribution:")
    for rank, count in sorted(worker_counts.items()):
        if rank not in dead_workers:
            log.info(f"  rank {rank}: {count:3d} pages")
    max_w = max(worker_counts.values())
    min_w = min((v for r, v in worker_counts.items()
                 if r not in dead_workers), default=1)
    log.info(f"  imbalance ratio: {max_w / min_w:.2f}")
    log.info("-" * 60)

    # final stats write so dashboard shows 100%
    write_stats(doc_id, 0, elapsed, worker_counts, dead_workers, MAX_PAGES,
                last_event="Crawl complete")
    
    # send DONE to any dead workers still hanging
    for rank in dead_workers:
        try:
            comm.send(None, dest=rank, tag=TAG_DONE)
        except Exception:
            pass

    # remove checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Checkpoint cleared -- crawl complete")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        run_master(comm)
    else:
        # workers are in worker.py — but when running via master.py,
        # non-zero ranks need to run worker logic too
        from worker import run_worker
        run_worker(comm)
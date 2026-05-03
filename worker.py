"""
Phase 4 — Worker with Heartbeat
=================================
Sends a heartbeat to master every HEARTBEAT_INTERVAL seconds
while a crawl task is in progress.
"""

import threading
import time

import requests
from mpi4py import MPI

from utils.parser import parse_page
from utils.logger import get_logger

from utils.robots import is_allowed as robots_allowed, wait_if_needed

from utils.metrics import record_fetch

from urllib.parse import urlparse as _urlparse

from utils.gossip import WorkerGossipState, GOSSIP_INTERVAL

TAG_TOKEN_REQUEST = 6
TAG_TOKEN_GRANT   = 7

TAG_GOSSIP_REQUEST = 8
TAG_GOSSIP_REPLY   = 9

TAG_READY      = 1
TAG_TASK       = 2
TAG_RESULT     = 3
TAG_DONE       = 4
TAG_HEARTBEAT  = 5

TIMEOUT            = 10
DELAY              = 0.3
HEARTBEAT_INTERVAL = 4     # send heartbeat every N seconds while crawling
HEADERS = {
    "User-Agent": "PDC-Crawler/1.0 (university project; educational use)"
}

log = get_logger("worker")

def fetch(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None, None
        return resp.text, resp.url
    except requests.RequestException as e:
        log.warning(f"Rank {MPI.COMM_WORLD.Get_rank()} failed: {url} — {e}")
        return None, None


def heartbeat_loop(comm, stop_event):
    """Runs in a background thread, pinging master while crawling."""
    while not stop_event.is_set():
        comm.send("beat", dest=0, tag=TAG_HEARTBEAT)
        time.sleep(HEARTBEAT_INTERVAL)

def request_token(comm, url):
    """
    Ask master for a crawl token for this URL's domain.
    Blocks until master grants it (with backoff if needed).
    """
    domain = _urlparse(url).netloc
    while True:
        comm.send(domain, dest=0, tag=TAG_TOKEN_REQUEST)
        granted, wait = comm.recv(source=0, tag=TAG_TOKEN_GRANT)
        if granted:
            return
        # master said wait — sleep and retry
        time.sleep(wait + 0.05)

def run_worker(comm):
    rank         = comm.Get_rank()
    gossip_state = WorkerGossipState(rank)
    log.info(f"Worker rank {rank} ready (gossip interval: every {GOSSIP_INTERVAL} URLs)")

    while True:
        comm.send("READY", dest=0, tag=TAG_READY)

        status = MPI.Status()
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
        incoming_tag = status.Get_tag()

        if incoming_tag == TAG_DONE:
            comm.recv(source=0, tag=TAG_DONE)
            stats = gossip_state.stats()
            log.info(f"Worker rank {rank} shutting down -- "
                     f"gossip rounds: {stats['gossip_rounds']}, "
                     f"merges: {stats['merges_done']}, "
                     f"filter load: {stats['filter_load_factor']:.2%}")
            break

        url, depth, doc_id = comm.recv(source=0, tag=TAG_TASK)

        if url == "__WAIT__":
            time.sleep(0.5)
            continue

        log.info(f"  rank {rank} crawling [{doc_id}]: {url}")

        # ── robots.txt check ──────────────────────────────────────────────
        if not robots_allowed(url):
            log.info(f"  rank {rank} blocked by robots.txt: {url}")
            comm.send((doc_id, url, "Blocked", "", "", [], depth),
                      dest=0, tag=TAG_RESULT)
            continue

        # ── request rate limit token from master ──────────────────────────
        request_token(comm, url)

        # ── start heartbeat ───────────────────────────────────────────────
        stop_event = threading.Event()
        hb_thread  = threading.Thread(target=heartbeat_loop,
                                      args=(comm, stop_event), daemon=True)
        hb_thread.start()

        # ── enforce crawl delay ───────────────────────────────────────────
        wait_if_needed(url)

        # ── timed fetch ───────────────────────────────────────────────────
        from urllib.parse import urlparse
        domain     = urlparse(url).netloc
        t_start    = time.time()
        html, final_url = fetch(url)
        duration_ms = (time.time() - t_start) * 1000

        stop_event.set()
        hb_thread.join(timeout=2)

        if html is None:
            record_fetch(domain, duration_ms, status="failed")
            comm.send((doc_id, url, "Failed", "", "", [], depth),
                      dest=0, tag=TAG_RESULT)
            continue

        title, text, snippet, links = parse_page(html, final_url)

        # ── update local gossip filter ────────────────────────────────────
        gossip_state.add_url(url)
        for link in links[:50]:   # sample links to avoid overloading filter
            gossip_state.local_bloom.add(link)

        # ── gossip round ──────────────────────────────────────────────────
        if gossip_state.should_gossip():
            new_bits = do_gossip(comm, gossip_state)
            if new_bits > 0:
                log.info(f"  [gossip] rank {rank} learned {new_bits} "
                         f"new URL bits from peer")

        record_fetch(domain, duration_ms, status="ok")
        comm.send((doc_id, final_url, title, text, snippet, links, depth),
                  dest=0, tag=TAG_RESULT)

def do_gossip(comm, gossip_state):
    """Send our filter to master, receive a peer's, merge it."""
    try:
        my_bytes = gossip_state.get_filter_bytes()
        comm.send(my_bytes, dest=0, tag=TAG_GOSSIP_REQUEST)

        # use a short timeout probe before blocking recv
        status    = MPI.Status()
        deadline  = time.time() + 3.0   # wait max 3 seconds
        while time.time() < deadline:
            if comm.iprobe(source=0, tag=TAG_GOSSIP_REPLY, status=status):
                peer_rank, peer_bytes = comm.recv(source=0, tag=TAG_GOSSIP_REPLY)
                if peer_bytes is not None:
                    new_bits = gossip_state.merge_peer_filter(peer_bytes, peer_rank)
                    gossip_state.gossip_count += 1
                    return new_bits
                return 0
            time.sleep(0.05)
    except Exception as e:
        log.warning(f"Gossip failed for rank {gossip_state.rank}: {e}")
    return 0

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if MPI.COMM_WORLD.Get_rank() != 0:
        run_worker(comm)
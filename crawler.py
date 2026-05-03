"""
Phase 1 — Sequential Baseline Crawler
======================================
Crawls from a list of seed URLs up to MAX_PAGES pages,
staying within the same domain as each seed.

Run:
    python crawler.py

Output:
    data/crawled/<doc_id>.json   one file per crawled page
"""

import json
import os
import time
from collections import deque
from urllib.parse import urlparse

import requests

from utils.parser import parse_page, is_same_domain
from utils.logger import get_logger

# ── configuration ────────────────────────────────────────────────────────────

SEED_URLS = [
    "https://books.toscrape.com/",          # safe, legal practice crawl target
]

MAX_PAGES   = 50          # stop after this many pages (keep small for Phase 1)
MAX_DEPTH   = 2           # how many link-hops from seed
DELAY       = 0.5         # seconds between requests (be polite)
OUTPUT_DIR  = "data/crawled"
TIMEOUT     = 10          # seconds before a request times out

HEADERS = {
    "User-Agent": "PDC-Crawler/1.0 (university project; educational use)"
}

# ─────────────────────────────────────────────────────────────────────────────

log = get_logger("crawler")


def fetch(url):
    """Download a URL. Returns (html_string, final_url) or (None, None) on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        # only process HTML pages
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None, None
        return resp.text, resp.url          # resp.url handles redirects
    except requests.RequestException as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None, None


def save_page(doc_id, url, title, text, snippet, links):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = {
        "doc_id":  doc_id,
        "url":     url,
        "title":   title,
        "text":    text,
        "snippet": snippet,
        "links":   links,
    }
    path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

def crawl(seed_urls, max_pages=MAX_PAGES, max_depth=MAX_DEPTH):
    """
    BFS crawl starting from seed_urls.

    frontier  : deque of (url, depth) tuples
    visited   : set of URLs already crawled or queued
    """
    frontier = deque()
    visited  = set()

    for url in seed_urls:
        frontier.append((url, 0))
        visited.add(url)

    doc_id     = 0
    start_time = time.time()

    log.info(f"Starting crawl — seeds: {seed_urls}, max_pages: {max_pages}")

    while frontier and doc_id < max_pages:
        url, depth = frontier.popleft()

        log.info(f"[{doc_id+1}/{max_pages}] depth={depth}  {url}")

        html, final_url = fetch(url)
        if html is None:
            continue

        title, text, snippet, links = parse_page(html, final_url)
        save_page(doc_id, final_url, title, text, snippet, links)
        doc_id += 1

        # ── enqueue new links ──────────────────────────────────────────────
        if depth < max_depth:
            for link in links:
                if link not in visited and is_same_domain(link, url):
                    visited.add(link)
                    frontier.append((link, depth + 1))

        time.sleep(DELAY)                   # polite crawl delay

    elapsed = time.time() - start_time

    log.info("─" * 60)
    log.info(f"Crawl complete: {doc_id} pages in {elapsed:.1f}s")
    log.info(f"Pages/second : {doc_id / elapsed:.2f}")
    log.info(f"Output dir   : {OUTPUT_DIR}/")
    log.info("─" * 60)

    return doc_id, elapsed                  # return stats for benchmark later


if __name__ == "__main__":
    crawl(SEED_URLS)
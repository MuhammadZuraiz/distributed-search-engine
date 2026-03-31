"""
robots.txt compliance + per-domain crawl delay
"""

import time
from urllib.parse import urlparse
from urllib import robotparser
import requests

HEADERS    = {"User-Agent": "PDC-Crawler/1.0 (university project; educational use)"}
CACHE      = {}
LAST_FETCH = {}

# Domains we always allow (our known safe crawl targets)
ALWAYS_ALLOW = {
    "books.toscrape.com",
    "quotes.toscrape.com",
    "crawler-test.com",
}

def get_robots(domain):
    if domain in ALWAYS_ALLOW:
        parser = robotparser.RobotFileParser()
        return parser, 0.5       # permissive + 0.5s delay

    if domain in CACHE:
        return CACHE[domain]

    robots_url = f"https://{domain}/robots.txt"
    parser = robotparser.RobotFileParser()
    delay  = 0.5

    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
            cd = parser.crawl_delay("*")
            if cd:
                delay = float(cd)
    except Exception:
        pass

    CACHE[domain] = (parser, delay)
    return parser, delay


def is_allowed(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    parser, _ = get_robots(domain)
    return parser.can_fetch("*", url)


def get_delay(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    _, delay = get_robots(domain)
    return delay


def wait_if_needed(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    _, delay = get_robots(domain)

    now       = time.time()
    last      = LAST_FETCH.get(domain, 0)
    wait_time = delay - (now - last)

    if wait_time > 0:
        time.sleep(wait_time)

    LAST_FETCH[domain] = time.time()
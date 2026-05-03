from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def parse_page(html, base_url):
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())

    # ── extract snippet (first 600 meaningful chars) ──────────────────────
    paragraphs = soup.find_all("p")
    snippet    = ""
    for p in paragraphs:
        p_text = p.get_text(separator=" ").strip()
        if len(p_text) > 80:          # skip tiny paragraphs
            snippet = p_text[:600]
            break
    if not snippet:
        snippet = text[:600]          # fallback to raw text

    links = []
    for a_tag in soup.find_all("a", href=True):
        href     = a_tag["href"].strip()
        absolute = urljoin(base_url, href)
        parsed   = urlparse(absolute)
        if parsed.scheme in ("http", "https"):
            clean = absolute.split("#")[0]
            links.append(clean)

    return title, text, snippet, links   # snippet added to return tuple

ALLOWED_DOMAINS = {
    "books.toscrape.com",
    "quotes.toscrape.com",
    "crawler-test.com",
    "en.wikipedia.org",
}

def is_allowed(url):
    """Return True if url belongs to any of our allowed domains."""
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower()
    return any(netloc == d or netloc.endswith("." + d) for d in ALLOWED_DOMAINS)
"""
Step 8 — Interactive PageRank Link Graph
==========================================
Builds a networkx graph from crawled link data.
Exports as interactive HTML using pyvis.
Node size = PageRank score.
Node color = domain.
Click any node to open its URL.

Run:
    python experiments/graph_viz.py
Output:
    experiments/graphs/link_graph.html
"""

import json
import glob
import os
import networkx as nx
from pyvis.network import Network
from urllib.parse import urlparse

CRAWLED_DIR = "data/crawled_distributed"
PR_PATH     = "data/index/pagerank.json"
OUTPUT      = "experiments/graphs/link_graph.html"
MAX_NODES   = 300    # cap for performance — top 300 by PageRank
MAX_EDGES   = 800    # cap edges to keep graph readable

os.makedirs("experiments/graphs", exist_ok=True)

# ── load PageRank scores ──────────────────────────────────────────────────────

with open(PR_PATH, encoding="utf-8") as f:
    pr_raw = json.load(f)
pagerank = {int(k): v for k, v in pr_raw.items()}

# ── load crawled documents ────────────────────────────────────────────────────

docs = {}
for fp in glob.glob(os.path.join(CRAWLED_DIR, "*.json")):
    with open(fp, encoding="utf-8") as f:
        doc = json.load(f)
    docs[doc["doc_id"]] = doc

# ── build URL -> doc_id map ───────────────────────────────────────────────────

url_to_id = {doc["url"]: doc_id for doc_id, doc in docs.items()}

# ── select top N docs by PageRank ─────────────────────────────────────────────

top_ids = sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True)
top_ids = set(top_ids[:MAX_NODES])

# ── domain color map ──────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "en.wikipedia.org":    "#4A90D9",
    "books.toscrape.com":  "#E8A838",
    "quotes.toscrape.com": "#5BAD6F",
    "crawler-test.com":    "#D85A30",
}

def get_color(url):
    domain = urlparse(url).netloc
    return DOMAIN_COLORS.get(domain, "#888780")

def get_domain_label(url):
    domain = urlparse(url).netloc
    labels = {
        "en.wikipedia.org":    "Wikipedia",
        "books.toscrape.com":  "Books",
        "quotes.toscrape.com": "Quotes",
        "crawler-test.com":    "Crawler Test",
    }
    return labels.get(domain, domain)

# ── build networkx graph ──────────────────────────────────────────────────────

G = nx.DiGraph()

# add nodes
for doc_id in top_ids:
    doc   = docs[doc_id]
    pr    = pagerank.get(doc_id, 0)
    url   = doc.get("url", "")
    title = doc.get("title", "")[:50]
    G.add_node(
        doc_id,
        label=title,
        url=url,
        pr=pr,
        color=get_color(url),
        domain=get_domain_label(url)
    )

# add edges (only between nodes in our top set)
edge_count = 0
for doc_id in top_ids:
    if edge_count >= MAX_EDGES:
        break
    doc   = docs[doc_id]
    links = doc.get("links", [])
    for link in links:
        if edge_count >= MAX_EDGES:
            break
        target_id = url_to_id.get(link)
        if target_id and target_id in top_ids and target_id != doc_id:
            if not G.has_edge(doc_id, target_id):
                G.add_edge(doc_id, target_id)
                edge_count += 1

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── build pyvis network ───────────────────────────────────────────────────────

net = Network(
    height="900px",
    width="100%",
    bgcolor="#0f1923",
    font_color="#e8ecf0",
    directed=True
)

# scale node size by PageRank
max_pr = max(pagerank.values()) or 1

for node_id, data in G.nodes(data=True):
    pr          = data.get("pr", 0)
    size        = 8 + (pr / max_pr) * 50    # size 8 to 58
    title_html  = (
        f"<b>{data['label']}</b><br>"
        f"Domain: {data['domain']}<br>"
        f"PageRank: {pr:.4f}<br>"
        f"<a href='{data['url']}' target='_blank'>{data['url'][:60]}</a>"
    )
    net.add_node(
        node_id,
        label=data["label"][:25],
        title=title_html,
        size=size,
        color=data["color"],
        url=data["url"]
    )

for src, tgt in G.edges():
    net.add_edge(src, tgt, color="#2a3a50", arrows="to", width=0.5)

# ── physics settings for nice layout ─────────────────────────────────────────

net.set_options("""
{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 120,
      "springConstant": 0.04,
      "damping": 0.15
    },
    "stabilization": {
      "iterations": 200
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "smooth": {
      "type": "curvedCW",
      "roundness": 0.2
    }
  }
}
""")

# ── add legend ────────────────────────────────────────────────────────────────

legend_html = """
<div style="position:fixed;top:20px;right:20px;background:#1a2535;
     border:1px solid #2a3a50;border-radius:8px;padding:16px;
     font-family:Arial;font-size:13px;color:#e8ecf0;z-index:999">
  <b style="font-size:14px">PDC Link Graph</b><br><br>
  <span style="color:#4A90D9">&#9679;</span> Wikipedia<br>
  <span style="color:#E8A838">&#9679;</span> Books to Scrape<br>
  <span style="color:#5BAD6F">&#9679;</span> Quotes to Scrape<br>
  <span style="color:#D85A30">&#9679;</span> Crawler Test<br><br>
  <span style="color:#aaa;font-size:11px">Node size = PageRank score<br>
  Hover for details</span>
</div>
"""

net.html = net.html.replace("</body>", legend_html + "</body>")
net.save_graph(OUTPUT)

print(f"Graph saved: {OUTPUT}")
print(f"Open in browser: file:///{os.path.abspath(OUTPUT)}")

# ── print top 10 nodes by PageRank ────────────────────────────────────────────
print("\nTop 10 nodes by PageRank:")
print("-" * 60)
for doc_id in sorted(top_ids, key=lambda x: pagerank[x], reverse=True)[:10]:
    doc = docs[doc_id]
    pr  = pagerank[doc_id]
    print(f"  [{doc_id:4d}] PR={pr:.4f}  {doc.get('title','')[:50]}")
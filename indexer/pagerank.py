"""
PageRank — iterative link-graph scoring
========================================
Reads all crawled JSON files, builds a link graph,
runs PageRank for N iterations, returns {doc_id: score}.
"""

import json
import glob
import os
from collections import defaultdict


DAMPING    = 0.85
ITERATIONS = 20


def build_link_graph(crawled_dir):
    """
    Returns:
        url_to_id  : {url: doc_id}
        id_to_url  : {doc_id: url}
        links_out  : {doc_id: [doc_id, ...]}  outgoing links
        links_in   : {doc_id: [doc_id, ...]}  incoming links
    """
    url_to_id = {}
    id_to_url = {}
    raw_links = {}     # doc_id -> [url, url, ...]

    for fp in glob.glob(os.path.join(crawled_dir, "*.json")):
        with open(fp, encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc["doc_id"]
        url    = doc.get("url", "")
        url_to_id[url]    = doc_id
        id_to_url[doc_id] = url
        raw_links[doc_id] = doc.get("links", [])

    # resolve raw URLs to doc_ids (only keep internal links)
    links_out = defaultdict(list)
    links_in  = defaultdict(list)

    for src_id, targets in raw_links.items():
        seen = set()
        for target_url in targets:
            if target_url in url_to_id:
                tgt_id = url_to_id[target_url]
                if tgt_id != src_id and tgt_id not in seen:
                    links_out[src_id].append(tgt_id)
                    links_in[tgt_id].append(src_id)
                    seen.add(tgt_id)

    return url_to_id, id_to_url, dict(links_out), dict(links_in)


def compute_pagerank(crawled_dir):
    """
    Run PageRank over the crawled link graph.
    Returns {doc_id: pagerank_score} normalised to [0, 1].
    """
    url_to_id, id_to_url, links_out, links_in = build_link_graph(crawled_dir)
    n = len(id_to_url)
    if n == 0:
        return {}

    all_ids = list(id_to_url.keys())

    # initialise uniform scores
    scores = {doc_id: 1.0 / n for doc_id in all_ids}

    for iteration in range(ITERATIONS):
        new_scores = {}
        for doc_id in all_ids:
            # sum contributions from all pages linking to this one
            incoming_sum = 0.0
            for src_id in links_in.get(doc_id, []):
                out_count = len(links_out.get(src_id, []))
                if out_count > 0:
                    incoming_sum += scores[src_id] / out_count

            new_scores[doc_id] = (1 - DAMPING) / n + DAMPING * incoming_sum

        # check convergence
        delta = sum(abs(new_scores[d] - scores[d]) for d in all_ids)
        scores = new_scores
        if delta < 1e-6:
            print(f"[pagerank] converged at iteration {iteration + 1}")
            break

    # normalise to [0, 1]
    max_score = max(scores.values()) or 1.0
    normalised = {doc_id: round(score / max_score, 6)
                  for doc_id, score in scores.items()}

    return normalised
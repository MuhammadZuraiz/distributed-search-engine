"""
Federated Distributed Search
==============================
Splits the inverted index across N shards.
A query goes to ALL shards in parallel via MPI.
Each shard returns its local top-K results.
A coordinator merges and re-ranks them globally.

This closes the loop on the distributed architecture:
  Crawling    → distributed (MPI workers)
  Indexing    → distributed (MapReduce)
  Querying    → distributed (federated shards)  ← this file

Architecture:
  Rank 0 = query coordinator (receives query, merges results)
  Ranks 1..N = index shards (each holds 1/N of the index)

Real systems using this pattern:
  Google        — 1000s of index shards per datacenter
  Apache Solr   — SolrCloud distributed search
  Elasticsearch — distributed shards with replica failover
  Bing          — tiered shard architecture

Key insight: the merger uses a priority queue to efficiently
combine K sorted lists from N shards — O(K*N*log(N)) time.
"""

import json
import math
import heapq
import time
from collections import defaultdict

# add this near the top of run_federated_search_server(), before the db import:
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_url_map


INDEX_PATH = "data/index/inverted_index.json"
PR_PATH    = "data/index/pagerank.json"


def shard_index(index, n_shards):
    """
    Partition the inverted index into n_shards equal parts.
    Sharding strategy: by doc_id range.
      Shard 0: doc_ids 0..max/n
      Shard 1: doc_ids max/n..2*max/n
      etc.

    Returns list of n_shards dicts, each a partial index.
    """
    # find max doc_id
    all_doc_ids = set()
    for entry in index.values():
        all_doc_ids.update(entry["doc_ids"])

    if not all_doc_ids:
        return [{} for _ in range(n_shards)]

    max_id    = max(all_doc_ids)
    shard_size = math.ceil((max_id + 1) / n_shards)

    shards = [{} for _ in range(n_shards)]

    for term, entry in index.items():
        # split this term's postings across shards
        for shard_idx in range(n_shards):
            lo = shard_idx * shard_size
            hi = lo + shard_size
            shard_doc_ids = [d for d in entry["doc_ids"] if lo <= d < hi]
            if shard_doc_ids:
                shards[shard_idx][term] = {
                    "doc_ids": shard_doc_ids,
                    "tf": {str(d): entry["tf"].get(str(d), 1)
                           for d in shard_doc_ids}
                }

    return shards


class ShardSearcher:
    """
    Searches one shard of the index using BM25.
    Runs in each MPI worker rank.
    """

    def __init__(self, shard_index, shard_id, pagerank=None):
        self.index    = shard_index
        self.shard_id = shard_id
        self.pagerank = pagerank or {}
        self.N        = sum(len(e["doc_ids"]) for e in shard_index.values())
        self.avgdl    = self._compute_avgdl()

        # BM25 parameters
        self.k1 = 1.5
        self.b  = 0.75

    def _compute_avgdl(self):
        doc_lengths = defaultdict(int)
        for entry in self.index.values():
            for doc_id_str, tf in entry["tf"].items():
                doc_lengths[int(doc_id_str)] += tf
        if not doc_lengths:
            return 1
        return sum(doc_lengths.values()) / len(doc_lengths)

    def search(self, query_terms, top_k=10):
        """
        BM25 search over this shard.
        Returns list of (score, doc_id) sorted descending.
        """
        scores = defaultdict(float)
        N      = max(self.N, 1)

        doc_lengths = defaultdict(int)
        for entry in self.index.values():
            for doc_id_str, tf in entry["tf"].items():
                doc_lengths[int(doc_id_str)] += tf

        for term in query_terms:
            if term not in self.index:
                continue
            entry = self.index[term]
            df    = len(entry["doc_ids"])
            idf   = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id in entry["doc_ids"]:
                tf  = entry["tf"].get(str(doc_id), 0)
                dl  = doc_lengths.get(doc_id, self.avgdl)
                tf_norm = (tf * (self.k1 + 1)) / \
                          (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[doc_id] += idf * tf_norm

        # blend with PageRank
        for doc_id in scores:
            pr = self.pagerank.get(doc_id, 0)
            scores[doc_id] = 0.85 * scores[doc_id] + 0.15 * pr * 10

        # return top_k
        top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [(score, doc_id) for doc_id, score in top]


class FederatedSearchCoordinator:
    """
    Coordinator that merges results from all shards.
    Uses a priority queue for efficient K-way merge.
    """

    def __init__(self, url_map):
        self.url_map = url_map

    def merge_shard_results(self, shard_results, top_n=20):
        """
        Merge results from N shards using a max-heap.

        shard_results : list of lists, each [(score, doc_id), ...]
        Returns        : top_n results globally, sorted by score
        """
        # flatten all results into one list
        all_results = []
        for shard_id, results in enumerate(shard_results):
            for score, doc_id in results:
                all_results.append((score, doc_id, shard_id))

        # sort by score descending
        all_results.sort(key=lambda x: x[0], reverse=True)

        # deduplicate (same doc can't appear in multiple shards
        # but just in case) and format
        seen     = set()
        merged   = []
        for score, doc_id, shard_id in all_results:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            meta = self.url_map.get(str(doc_id), {})
            merged.append({
                "doc_id":   doc_id,
                "url":      meta.get("url", ""),
                "title":    meta.get("title", "No title"),
                "snippet":  meta.get("snippet", ""),
                "score":    round(score, 3),
                "shard_id": shard_id,
            })
            if len(merged) >= top_n:
                break

        return merged


def run_federated_search_server(n_shards=6):
    """
    MPI-based federated search server.

    Rank 0: coordinator — receives queries, distributes to shards,
            merges results, returns to caller.
    Ranks 1..N: shard workers — load their index partition,
                wait for queries, return local results.

    Run as:
        mpiexec -n 7 python search/federated.py
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    TAG_QUERY  = 10
    TAG_RESULT = 11
    TAG_QUIT   = 12

    if rank == 0:
        # ── coordinator ───────────────────────────────────────────────────
        print(f"[federated] Coordinator starting with {size-1} shards")

        # load and shard the index
        with open(INDEX_PATH, encoding="utf-8") as f:
            full_index = json.load(f)

        pr = {}
        if os.path.exists(PR_PATH):
            with open(PR_PATH, encoding="utf-8") as f:
                pr = {int(k): v for k, v in json.load(f).items()}

        shards = shard_index(full_index, size - 1)
        print(f"[federated] Index sharded: "
              f"{[len(s) for s in shards]} terms per shard")

        # distribute shards to workers
        for worker_rank in range(1, size):
            shard_data = {
                "shard":    shards[worker_rank - 1],
                "shard_id": worker_rank - 1,
                "pagerank": pr,
            }
            comm.send(shard_data, dest=worker_rank, tag=TAG_QUERY)

        print("[federated] Shards distributed. Ready for queries.")

        # load URL map for result formatting
        from db.database import get_url_map
        url_map     = get_url_map()
        coordinator = FederatedSearchCoordinator(url_map)

        # query loop
        while True:
            query = input("\nFederated search> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                for r in range(1, size):
                    comm.send(None, dest=r, tag=TAG_QUIT)
                break
            if not query:
                continue

            terms    = [t.lower() for t in query.split() if len(t) >= 3]
            t_start  = time.time()

            # broadcast query to all shards
            for worker_rank in range(1, size):
                comm.send(terms, dest=worker_rank, tag=TAG_QUERY)

            # collect shard results
            shard_results = []
            for worker_rank in range(1, size):
                results = comm.recv(source=worker_rank, tag=TAG_RESULT)
                shard_results.append(results)

            elapsed = (time.time() - t_start) * 1000
            merged  = coordinator.merge_shard_results(shard_results)

            print(f"\nQuery: '{query}' — {len(merged)} results "
                  f"from {size-1} shards in {elapsed:.1f}ms")
            print("-" * 70)
            for i, r in enumerate(merged[:10], 1):
                print(f"  {i:2d}. [{r['score']:.3f}] (shard {r['shard_id']}) "
                      f"{r['title'][:55]}")
                print(f"       {r['url'][:70]}")

    else:
        # ── shard worker ──────────────────────────────────────────────────
        # receive shard data
        init_data = comm.recv(source=0, tag=TAG_QUERY)
        searcher  = ShardSearcher(
            shard_index = init_data["shard"],
            shard_id    = init_data["shard_id"],
            pagerank    = init_data["pagerank"],
        )
        print(f"[federated] Shard {init_data['shard_id']} (rank {rank}): "
              f"{len(init_data['shard'])} terms loaded")

        # search loop
        while True:
            status = MPI.Status()
            comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
            incoming_tag = status.Get_tag()

            if incoming_tag == TAG_QUIT:
                comm.recv(source=0, tag=TAG_QUIT)
                print(f"[federated] Shard {init_data['shard_id']} shutting down")
                break

            terms   = comm.recv(source=0, tag=TAG_QUERY)
            results = searcher.search(terms, top_k=10)
            comm.send(results, dest=0, tag=TAG_RESULT)


if __name__ == "__main__":
    run_federated_search_server()
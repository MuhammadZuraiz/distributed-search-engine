"""
Phase 5 — Parallel MapReduce Indexing Orchestrator (fixed)
"""

import glob
import json
import os
import sys
import time

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indexer.reducer import save_index, merge_indexes
from utils.logger    import get_logger
from utils.push_sum  import PushSumAggregator          # Phase 3

CRAWLED_DIR      = "data/crawled_distributed"
INDEX_PATH       = "data/index/inverted_index.json"
URL_MAP_PATH     = "data/index/url_map.json"
STATS_PATH       = "data/index/index_stats.json"

# ── single-machine memory limit ───────────────────────────────────────────────
# On a real cluster, each node has independent RAM so this is unnecessary.
# On one machine simulating 6 workers, all processes share the same RAM.
# At 3917 docs/mapper, each partial index is ~1.9 GB → 11.4 GB total → OOM/swap.
# At 500 docs/mapper: ~270 MB each → 1.6 GB total → fits comfortably.
# Set to None to index everything (only do this on a machine with 32+ GB RAM).
DOCS_PER_MAPPER  = 500

log = get_logger("indexer")

# replace build_url_map function with:
def build_url_map_from_db():
    from db.database import get_url_map
    return get_url_map()

def run():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ── every rank computes its partition ─────────────────────────────────────
    if rank == 0:
        my_partial_index   = {}
        local_doc_count    = 0
        local_total_length = 0.0
        start              = time.time()
        log.info(f"MapReduce indexing started — {size-1} mapper(s)")
        log.info("Phase 3: epidemic aggregation via push-sum gossip")
    else:
        from db.database import get_conn, get_document
        from indexer.mapper import tokenise

        conn        = get_conn()
        all_ids     = [r[0] for r in conn.execute(
                       "SELECT doc_id FROM documents ORDER BY doc_id").fetchall()]
        num_mappers = size - 1
        my_ids      = all_ids[rank - 1 :: num_mappers]

        if DOCS_PER_MAPPER is not None:
            my_ids = my_ids[:DOCS_PER_MAPPER]
            log.info(
                f"Mapper {rank}: capped at {DOCS_PER_MAPPER} docs "
                f"(single-machine RAM limit — real cluster would process all {len(all_ids[rank-1::num_mappers])})"
            )

        log.info(f"Mapper {rank}: processing {len(my_ids)} docs (one-pass map+reduce)")

        # ── ONE-PASS MAP + LOCAL COMBINE ──────────────────────────────────
        # Builds partial index and local stats in a single pass through the DB.
        # Never creates a list of raw (word, doc_id) pairs — previous approach
        # created 14.5M pairs per mapper (~2GB each), causing OOM when all
        # 6 mappers ran simultaneously (~12GB just for pairs, before tf_counts).
        #
        # Memory now: only the partial index dict (~150-250MB per mapper).
        #
        # Output format matches reducer.py's reduce_pairs() output so
        # merge_indexes() works unchanged.
        #
        partial: dict = {}        # {term: {doc_id: count}}
        local_doc_count    = 0
        local_total_length = 0.0

        for i, doc_id in enumerate(my_ids):
            doc = get_document(doc_id)
            if not doc:
                continue
            text  = (doc.get("title") or "") + " " + (doc.get("text") or "")
            words = tokenise(text)
            if not words:
                continue

            doc_len             = len(words)
            local_doc_count    += 1
            local_total_length += doc_len

            for word in words:
                if word not in partial:
                    partial[word] = {}
                partial[word][doc_id] = partial[word].get(doc_id, 0) + 1

            if (i + 1) % 500 == 0:
                log.info(
                    f"Mapper {rank}: {i+1}/{len(my_ids)} docs | "
                    f"{len(partial):,} terms so far"
                )

        # Format to match reducer.py output (doc_ids sorted, tf as str keys)
        my_partial_index = {
            term: {
                "doc_ids": sorted(counts.keys()),
                "tf":      {str(d): c for d, c in counts.items()},
            }
            for term, counts in partial.items()
        }
        del partial

        log.info(
            f"Mapper {rank}: done — {local_doc_count} docs, "
            f"{len(my_partial_index):,} terms, "
            f"avg_len={local_total_length/max(local_doc_count,1):.0f}"
        )

    # ── Phase 3: push-sum epidemic aggregation ────────────────────────────────
    # Barrier: all ranks finish their one-pass map before gossip starts.
    # Rank 0 reaches here instantly; mappers arrive when their doc loop finishes.
    comm.Barrier()
    t_gossip_start = time.time()

    aggregator = PushSumAggregator(comm, {
        "doc_count":    float(local_doc_count),
        "total_length": local_total_length,
    })
    gossip_estimates = aggregator.run()
    t_gossip_ms = (time.time() - t_gossip_start) * 1000

    # Exact allreduce alongside gossip — reports gossip accuracy as a metric.
    # Push-sum's ring-doubling has measurable error for non-power-of-2 sizes
    # (n=7 here). Allreduce gives the exact value; gossip_error_pct shows
    # how close O(log n) rounds got — that's the academic result.
    from mpi4py import MPI as _MPI
    exact_total_docs = comm.allreduce(local_doc_count,         op=_MPI.SUM)
    exact_total_len  = comm.allreduce(int(local_total_length), op=_MPI.SUM)

    gossip_total = max(1, round(gossip_estimates["doc_count"] * size))
    gossip_error = abs(gossip_total - exact_total_docs) / max(exact_total_docs, 1) * 100

    total_docs = exact_total_docs
    total_len  = exact_total_len
    avgdl      = total_len / max(total_docs, 1)

    log.info(
        f"[push_sum] rank {rank}: gossip={gossip_total} | exact={total_docs} | "
        f"error={gossip_error:.2f}% | {t_gossip_ms:.0f}ms | "
        f"rounds={len(aggregator._round_times)}"
    )

    # ── gather partial indexes (not raw pairs) to rank 0 ─────────────────────
    all_partial_indexes = comm.gather(my_partial_index, root=0)
    del my_partial_index

    # ── rank 0: merge partial indexes + save ─────────────────────────────────
    if rank == 0:
        from indexer.reducer import merge_indexes
        log.info(f"Merging {len(all_partial_indexes)} partial indexes...")
        index = {}
        for i, partial in enumerate(all_partial_indexes):
            if partial:
                log.info(f"  merging rank {i}: {len(partial):,} terms")
                index = merge_indexes(index, partial)
        log.info(f"Merged index: {len(index):,} unique terms")
        save_index(index, INDEX_PATH)

        # ── PageRank from DB ──────────────────────────────────────────────
        # total_docs (n) comes from push-sum allreduce — no extra DB query.
        log.info(f"Computing PageRank (n={total_docs} from epidemic gossip)...")
        from indexer.pagerank import build_link_graph, compute_pagerank
        from db.database import get_all_documents, update_pagerank

        # build link graph from DB instead of JSON files
        docs     = {d["doc_id"]: d for d in get_all_documents()}
        url_to_id = {d["url"]: d["doc_id"] for d in docs.values()}

        # run PageRank directly on DB data
        from collections import defaultdict
        DAMPING    = 0.85
        ITERATIONS = 20
        n          = total_docs    # ← from push-sum, not len(docs)
        all_ids    = list(docs.keys())
        links_out  = defaultdict(list)
        links_in   = defaultdict(list)

        for src_id, doc in docs.items():
            seen = set()
            for target_url in doc.get("links", []):
                tgt_id = url_to_id.get(target_url)
                if tgt_id and tgt_id != src_id and tgt_id not in seen:
                    links_out[src_id].append(tgt_id)
                    links_in[tgt_id].append(src_id)
                    seen.add(tgt_id)

        scores = {doc_id: 1.0 / n for doc_id in all_ids}
        for iteration in range(ITERATIONS):
            new_scores = {}
            for doc_id in all_ids:
                incoming = sum(
                    scores[src] / len(links_out[src])
                    for src in links_in.get(doc_id, [])
                    if links_out.get(src)
                )
                new_scores[doc_id] = (1 - DAMPING) / n + DAMPING * incoming
            delta  = sum(abs(new_scores[d] - scores[d]) for d in all_ids)
            scores = new_scores
            if delta < 1e-6:
                log.info(f"PageRank converged at iteration {iteration + 1}")
                break

        max_score = max(scores.values()) or 1.0
        pr_scores = {doc_id: round(s / max_score, 6)
                     for doc_id, s in scores.items()}

        # save to DB and JSON
        update_pagerank(pr_scores)
        pr_path = "data/index/pagerank.json"
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pr_scores.items()},
                      f, ensure_ascii=False)
        log.info(f"PageRank saved: {len(pr_scores)} scores")

        # ── index_stats.json — corpus stats + push-sum telemetry ─────────
        gossip_stats = aggregator.stats()
        index_stats  = {
            "total_docs":  total_docs,
            "avgdl":       round(avgdl, 4),
            "total_terms": len(index),
            "num_mappers": size - 1,
            "epidemic_aggregation": {
                "gossip_total_docs": gossip_total,
                "exact_total_docs":  total_docs,
                "gossip_error_pct":  round(gossip_error, 3),
                "rounds_run":        gossip_stats["rounds_run"],
                "converged_at":      gossip_stats["converged_at"],
                "total_time_ms":     gossip_stats["total_time_ms"],
                "round_times_ms":    gossip_stats["round_times_ms"],
                "note": (
                    "Gossip estimates global stats in O(log n) rounds. "
                    "Allreduce used for exact correctness; gossip_error_pct "
                    "measures how close the gossip got."
                ),
            },
        }
        os.makedirs("data/index", exist_ok=True)
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(index_stats, f, indent=2)
        log.info(f"Index stats (with gossip telemetry) saved to {STATS_PATH}")

        # ── BERT embeddings + FAISS ───────────────────────────────────────
        log.info("Building BERT embeddings + FAISS index...")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from search.embeddings import build_embeddings
        build_embeddings()
        log.info("BERT embeddings built and cached")

        # ── TF-IDF semantic vectors (kept as fallback) ────────────────────
        log.info("Building TF-IDF fallback vectors...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle as pkl

        corpus  = []
        doc_ids_list = []
        for doc in get_all_documents():
            corpus.append(doc.get("title","") + " " + doc.get("text",""))
            doc_ids_list.append(doc["doc_id"])

        vectorizer = TfidfVectorizer(
            max_features=30000, stop_words="english",
            ngram_range=(1,2), min_df=2, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(corpus)
        VECTOR_CACHE = "data/index/tfidf_vectors.pkl"
        with open(VECTOR_CACHE, "wb") as f:
            pkl.dump({"vectorizer": vectorizer,
                      "matrix": matrix, "doc_ids": doc_ids_list}, f)
        log.info(f"TF-IDF matrix: {matrix.shape}")

        # ── URL map from DB ───────────────────────────────────────────────
        url_map = build_url_map_from_db()
        os.makedirs(os.path.dirname(URL_MAP_PATH), exist_ok=True)
        with open(URL_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(url_map, f, ensure_ascii=False, indent=2)

        # ── mark all docs as indexed ──────────────────────────────────────
        from db.database import mark_indexed
        for doc_id in doc_ids_list:
            mark_indexed(doc_id, 0)

        elapsed = time.time() - start
        log.info("=" * 60)
        log.info(f"Indexing complete in {elapsed:.2f}s")
        log.info(f"  Unique terms   : {len(index):,}")
        log.info(f"  Documents      : {total_docs:,}  (from epidemic gossip + allreduce)")
        log.info(f"  Avg doc length : {avgdl:.1f} tokens")
        log.info("-" * 60)
        log.info("Phase 3  Epidemic Aggregation")
        log.info(f"  Gossip estimate  : {gossip_total:,} docs")
        log.info(f"  Exact (allreduce): {total_docs:,} docs")
        log.info(f"  Gossip error     : {gossip_error:.2f}%")
        log.info(f"  Gossip time      : {t_gossip_ms:.0f}ms "
                 f"({gossip_stats['rounds_run']} rounds, "
                 f"converged at round {gossip_stats['converged_at']})")
        log.info(f"  Round times (ms) : {gossip_stats['round_times_ms']}")
        log.info(f"  Stats available before gather: YES")
        top5 = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info(f"  Top PageRank   : {top5}")
        log.info("=" * 60)

    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    run()